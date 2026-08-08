from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import sherpa_onnx

from app.config import MODEL_DIR


class RecognizerFactory:
    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        self.model_dir = model_dir
        self._recognizer: sherpa_onnx.OnlineRecognizer | None = None
        self._lock = threading.Lock()

    def get(self) -> sherpa_onnx.OnlineRecognizer:
        with self._lock:
            if self._recognizer is None:
                files = {
                    "tokens": self.model_dir / "tokens.txt",
                    "encoder": self.model_dir / "encoder-epoch-99-avg-1.onnx",
                    "decoder": self.model_dir / "decoder-epoch-99-avg-1.onnx",
                    "joiner": self.model_dir / "joiner-epoch-99-avg-1.onnx",
                }
                missing = [str(path) for path in files.values() if not path.is_file()]
                if missing:
                    raise FileNotFoundError("ASR 模型文件缺失: " + ", ".join(missing))
                self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                    **{key: str(value) for key, value in files.items()},
                    num_threads=2,
                    sample_rate=16000,
                    feature_dim=80,
                    enable_endpoint_detection=True,
                    rule1_min_trailing_silence=2.4,
                    rule2_min_trailing_silence=1.2,
                    rule3_min_utterance_length=300,
                    decoding_method="greedy_search",
                    provider="cpu",
                )
            return self._recognizer


class StreamingAsr:
    def __init__(self, factory: RecognizerFactory) -> None:
        self.recognizer = factory.get()
        self.stream = self.recognizer.create_stream()
        self.final_parts: list[str] = []
        self.segments: list[dict[str, float | str]] = []
        self.partial = ""
        self.audio_seconds = 0.0
        self.segment_start_seconds = 0.0
        self.pre_roll_seconds = 0.2
        self._pre_roll = np.empty(0, dtype=np.float32)
        self._lock = threading.Lock()

    def accept(self, samples: np.ndarray, sample_rate: int = 16000) -> tuple[str, str | None]:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not samples.size:
            return self.partial, None
        with self._lock:
            self.stream.accept_waveform(sample_rate, samples)
            self.audio_seconds += samples.size / sample_rate
            pre_roll_size = max(1, int(sample_rate * self.pre_roll_seconds))
            self._pre_roll = np.concatenate((self._pre_roll, samples))[-pre_roll_size:]
            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)
            self.partial = self.recognizer.get_result(self.stream).strip()
            final = None
            if self.recognizer.is_endpoint(self.stream):
                if self.partial:
                    final = self._commit_final(
                        self.partial, self.segment_start_seconds, self.audio_seconds
                    )
                # A browser audio block can contain both the trailing silence
                # that triggers endpointing and the first sound of the next
                # utterance. Start a fully isolated stream and replay a short
                # pre-roll so that boundary audio is not discarded.
                self.stream = self.recognizer.create_stream()
                if self._pre_roll.size:
                    self.stream.accept_waveform(sample_rate, self._pre_roll)
                self.partial = ""
                self.segment_start_seconds = max(
                    0.0, self.audio_seconds - self._pre_roll.size / sample_rate
                )
            return self.partial, final

    def finish(self) -> str | None:
        with self._lock:
            self.stream.input_finished()
            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)
            result = self.recognizer.get_result(self.stream).strip()
            if result:
                result = self._commit_final(
                    result, self.segment_start_seconds, self.audio_seconds
                )
            self.partial = ""
            return result or None

    def _commit_final(self, text: str, start: float, end: float) -> str | None:
        """Append one endpoint-confirmed segment without cross-segment trimming."""
        text = text.strip()
        if not text:
            return None
        self.final_parts.append(text)
        self.segments.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "text": text,
        })
        return text

    @property
    def transcript(self) -> str:
        return "\n".join(self.final_parts)


def transcribe_audio_file(
    path: Path,
    factory: RecognizerFactory,
    on_event: Callable[[dict], None],
    should_pause: Callable[[], bool],
    should_cancel: Callable[[], bool],
) -> str:
    duration = _probe_duration(path)
    command = [
        "ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
        "-acodec", "pcm_f32le", "-ac", "1", "-ar", "16000", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    session = StreamingAsr(factory)
    processed = 0
    chunk_bytes = 16000 * 4 // 2
    try:
        assert process.stdout is not None
        while True:
            if should_cancel():
                process.terminate()
                raise RuntimeError("识别任务已取消")
            if should_pause():
                threading.Event().wait(0.15)
                continue
            raw = process.stdout.read(chunk_bytes)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype=np.float32)
            processed += samples.size
            partial, final = session.accept(samples)
            event = {
                "type": "transcription.progress",
                "partial": partial,
                "transcript": session.transcript,
                "segments": session.segments,
                "progress": min(99, round(processed / (duration * 16000) * 100)) if duration else 0,
            }
            if final:
                event["final"] = final
            on_event(event)
        error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if process.wait() != 0:
            raise RuntimeError(error.strip() or "FFmpeg 无法解码该音频")
        session.finish()
        on_event({"type": "transcription.progress", "transcript": session.transcript, "segments": session.segments, "partial": "", "progress": 100})
        return session.transcript
    finally:
        if process.poll() is None:
            process.kill()


def _probe_duration(path: Path) -> float:
    command = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        return 0
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 0
