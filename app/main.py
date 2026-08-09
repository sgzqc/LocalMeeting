from __future__ import annotations

import asyncio
import json
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import ROOT_DIR, UPLOAD_DIR, get_settings
from app.services.asr import RecognizerFactory, StreamingAsr, transcribe_audio_file
from app.services.summary import SummaryService
from app.state import JobStore


settings = get_settings()
factory = RecognizerFactory()
summaries = SummaryService(settings)
jobs = JobStore()
STATIC_DIR = ROOT_DIR / "app" / "static"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".webm"}
MAX_UPLOAD_BYTES = 1024 * 1024 * 500


@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Match the X-ASR live demo's ordering: load the shared recognizer before
    # any microphone session can begin. Per-session VAD state is initialized
    # after the WebSocket connects and before the client receives "ready".
    await asyncio.to_thread(factory.preload)
    yield


app = FastAPI(title="会议助手", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "asr_backend": "x_asr",
        "asr_chunk_ms": 960,
        "llm_ready": settings.llm_ready,
        "summary_interval": settings.summary_interval_seconds,
    }


@app.websocket("/ws/live")
async def live_asr(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "status", "state": "initializing"})
    try:
        session = await asyncio.to_thread(StreamingAsr, factory)
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": f"ASR 初始化失败：{exc}"})
        await websocket.close(code=1011)
        return
    await websocket.send_json({"type": "status", "state": "ready"})
    summary = ""
    summarized_parts = 0
    summary_lock = asyncio.Lock()
    input_sample_rate = settings.sample_rate

    async def update_summary(force_full: bool = False) -> None:
        nonlocal summary, summarized_parts
        async with summary_lock:
            transcript = session.transcript
            parts = session.final_parts
            if not transcript or (not force_full and len(parts) == summarized_parts):
                return
            try:
                await websocket.send_json({"type": "summary.started"})
                if force_full:
                    summary = await summaries.summarize_full(transcript)
                else:
                    new_text = "\n".join(parts[summarized_parts:])
                    summary = await summaries.summarize_incremental(summary, new_text)
                summarized_parts = len(parts)
                await websocket.send_json({"type": "summary.updated", "content": summary, "final": force_full})
            except Exception as exc:
                await websocket.send_json({"type": "summary.error", "message": str(exc)})

    async def summary_timer() -> None:
        while True:
            await asyncio.sleep(settings.summary_interval_seconds)
            await update_summary()

    timer = asyncio.create_task(summary_timer())
    try:
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                samples = np.frombuffer(message["bytes"], dtype="<f4").copy()
                partial, final = await asyncio.to_thread(session.accept, samples, input_sample_rate)
                await websocket.send_json({
                    "type": "asr.result", "partial": partial,
                    "final": final, "transcript": session.transcript,
                    "segments": session.segments,
                })
            elif message.get("text"):
                command = json.loads(message["text"])
                if command.get("type") == "audio.config":
                    requested_rate = int(command.get("sample_rate", settings.sample_rate))
                    if not 8000 <= requested_rate <= 192000:
                        await websocket.send_json({"type": "error", "message": "无效的音频采样率"})
                        continue
                    input_sample_rate = requested_rate
                    await websocket.send_json({"type": "audio.configured", "sample_rate": input_sample_rate})
                elif command.get("type") == "stop":
                    final = await asyncio.to_thread(session.finish)
                    await websocket.send_json({
                        "type": "asr.result", "partial": "", "final": final,
                        "transcript": session.transcript, "segments": session.segments,
                    })
                    await update_summary(force_full=True)
                    await websocket.send_json({"type": "meeting.stopped"})
                    break
    except (WebSocketDisconnect, RuntimeError) as exc:
        # Starlette can surface a RuntimeError when a disconnect frame has
        # already been consumed. Treat it like a normal browser disconnect.
        if isinstance(exc, RuntimeError) and "disconnect message" not in str(exc):
            raise
        pass
    finally:
        timer.cancel()
        await asyncio.to_thread(session.close)


def _safe_filename(value: str) -> str:
    name = Path(value).name
    return re.sub(r"[^\w.()\-\u4e00-\u9fff ]", "_", name) or "audio.wav"


@app.post("/api/files")
async def upload_audio(request: Request) -> JSONResponse:
    filename = _safe_filename(unquote(request.headers.get("x-filename", "audio.wav")))
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "不支持的音频格式")
    content_length = int(request.headers.get("content-length", "0") or 0)
    if content_length > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "文件不能超过 500MB")
    job_id = uuid.uuid4().hex
    path = UPLOAD_DIR / f"{job_id}{suffix}"
    size = 0
    with path.open("wb") as target:
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                target.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, "文件不能超过 500MB")
            target.write(chunk)
    if not size:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "文件为空")
    job = jobs.create(filename, path, job_id)
    return JSONResponse(job.public())


def _publish_from_thread(loop: asyncio.AbstractEventLoop, job_id: str, event: dict) -> None:
    job = jobs.get(job_id)
    job.progress = int(event.get("progress", job.progress))
    job.transcript = event.get("transcript", job.transcript)
    job.segments = event.get("segments", job.segments)
    job.partial = event.get("partial", job.partial)
    for queue in list(job.subscribers):
        loop.call_soon_threadsafe(queue.put_nowait, event)


@app.post("/api/files/{job_id}/transcribe")
async def start_transcription(job_id: str) -> dict:
    try:
        job = jobs.get(job_id)
    except KeyError:
        raise HTTPException(404, "任务不存在")
    if job.status not in {"uploaded", "failed"}:
        raise HTTPException(409, "任务已经开始")
    job.status = "transcribing"
    job.error = ""
    loop = asyncio.get_running_loop()

    async def run() -> None:
        try:
            transcript = await asyncio.to_thread(
                transcribe_audio_file, job.path, factory,
                lambda event: _publish_from_thread(loop, job_id, event),
                job.paused.is_set, job.cancelled.is_set,
            )
            job.transcript = transcript
            job.partial = ""
            job.progress = 100
            job.status = "transcribed"
        except Exception as exc:
            job.status = "cancelled" if job.cancelled.is_set() else "failed"
            job.error = str(exc)
        finally:
            _publish_from_thread(loop, job_id, {"type": "job.state", **job.public()})
            if job.cancelled.is_set():
                job.path.unlink(missing_ok=True)
                jobs.jobs.pop(job_id, None)

    asyncio.create_task(run())
    return job.public()


@app.get("/api/files/{job_id}")
async def get_job(job_id: str) -> dict:
    try:
        return jobs.get(job_id).public()
    except KeyError:
        raise HTTPException(404, "任务不存在")


@app.get("/api/files/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    try:
        job = jobs.get(job_id)
    except KeyError:
        raise HTTPException(404, "任务不存在")
    queue: asyncio.Queue = asyncio.Queue()
    job.subscribers.add(queue)

    async def stream():
        try:
            yield f"data: {json.dumps({'type': 'job.state', **job.public()}, ensure_ascii=False)}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), 15)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            job.subscribers.discard(queue)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/files/{job_id}/pause")
async def pause_job(job_id: str) -> dict:
    try:
        job = jobs.get(job_id)
    except KeyError:
        raise HTTPException(404, "任务不存在")
    if job.status not in {"transcribing", "paused"}:
        raise HTTPException(409, "当前任务无法暂停")
    if job.paused.is_set():
        job.paused.clear()
        job.status = "transcribing"
    else:
        job.paused.set()
        job.status = "paused"
    return job.public()


@app.post("/api/files/{job_id}/summary")
async def summarize_job(job_id: str) -> StreamingResponse:
    try:
        job = jobs.get(job_id)
    except KeyError:
        raise HTTPException(404, "任务不存在")
    if job.status not in {"transcribed", "completed"}:
        raise HTTPException(409, "请等待音频识别完成")
    job.status = "summarizing"

    async def stream():
        job.summary = ""
        try:
            yield json.dumps({"type": "summary.started"}, ensure_ascii=False) + "\n"
            async for delta in summaries.stream_full(job.transcript):
                job.summary += delta
                yield json.dumps({"type": "summary.delta", "delta": delta}, ensure_ascii=False) + "\n"
            job.status = "completed"
            yield json.dumps({"type": "summary.done"}, ensure_ascii=False) + "\n"
        except Exception as exc:
            job.status = "transcribed"
            yield json.dumps({"type": "summary.error", "message": str(exc)}, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.delete("/api/files/{job_id}")
async def delete_job(job_id: str) -> dict:
    try:
        job = jobs.get(job_id)
    except KeyError:
        raise HTTPException(404, "任务不存在")
    job.cancelled.set()
    if job.status not in {"transcribing", "paused"}:
        job.path.unlink(missing_ok=True)
        jobs.jobs.pop(job_id, None)
    return {"deleted": True}
