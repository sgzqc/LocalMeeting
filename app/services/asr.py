from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import sherpa_onnx

from app.config import ROOT_DIR


SAMPLE_RATE = 16000
VAD_WINDOW = 512
DEFAULT_X_ASR_DEMO = Path(
    r"C:\Code\X-ASR\X-ASR-zh-en\deployment\x-asr-live-demo"
)

_CJK = r"㐀-䶿一-鿿豈-﫿"
_CJK_PUNCT = re.escape("，。！？；：、（）《》〈〉【】「」『』“”‘’")
_ASCII_PUNCT = re.escape(",.!?;:%)]}")


def normalize_cjk(text: str) -> str:
    """Remove Zipformer BPE spaces around CJK text while preserving English spaces."""
    text = re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])", "", text)
    text = re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK_PUNCT}])", "", text)
    text = re.sub(rf"(?<=[{_CJK_PUNCT}])\s+(?=[{_CJK}])", "", text)
    text = re.sub(rf"(?<=[{_CJK_PUNCT}])\s+(?=[{_CJK_PUNCT}])", "", text)
    return re.sub(rf"\s+(?=[{_ASCII_PUNCT}])", "", text)


def _configured_path(name: str, local: Path, demo: Path) -> Path:
    configured = os.getenv(name)
    if configured:
        return Path(configured).expanduser()
    if local.exists():
        return local
    return demo


class FireRedVad:
    """Adapter matching the VAD behavior used by X-ASR's live_asr.py demo."""

    def __init__(
        self,
        model_dir: Path,
        threshold: float = 0.5,
        min_silence: float = 0.7,
        min_speech: float = 0.25,
        chunk_seconds: float = 0.3,
    ) -> None:
        from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig

        frames_per_second = 100
        config = FireRedStreamVadConfig(
            speech_threshold=threshold,
            min_speech_frame=max(1, round(min_speech * frames_per_second)),
            min_silence_frame=max(1, round(min_silence * frames_per_second)),
        )
        self.vad = FireRedStreamVad.from_pretrained(str(model_dir), config)
        self.vad.reset()
        self.chunk_size = int(chunk_seconds * SAMPLE_RATE)
        self._buffer = np.empty(0, dtype=np.float32)
        self.in_speech = False

    def accept_waveform(self, samples: np.ndarray) -> None:
        self._buffer = np.concatenate((self._buffer, np.asarray(samples, dtype=np.float32)))
        while self._buffer.size >= self.chunk_size:
            chunk = self._buffer[: self.chunk_size]
            self._buffer = self._buffer[self.chunk_size :]
            int16 = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
            for result in self.vad.detect_chunk(int16):
                if result.is_speech_start:
                    self.in_speech = True
                if result.is_speech_end:
                    self.in_speech = False

    def is_speech_detected(self) -> bool:
        return self.in_speech


class StreamingLinearResampler:
    """Stateful linear resampler that preserves phase across browser audio blocks."""

    def __init__(self, output_rate: int = SAMPLE_RATE) -> None:
        self.output_rate = output_rate
        self.input_rate: int | None = None
        self._buffer = np.empty(0, dtype=np.float32)
        self._position = 0.0

    def process(self, samples: np.ndarray, input_rate: int) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not samples.size:
            return samples
        if input_rate <= 0:
            raise ValueError("采样率必须大于 0")
        if self.input_rate not in (None, input_rate):
            raise ValueError("识别过程中不能更改输入采样率")
        self.input_rate = input_rate
        if input_rate == self.output_rate:
            return samples.copy()

        self._buffer = np.concatenate((self._buffer, samples))
        step = input_rate / self.output_rate
        available = self._buffer.size - 1 - self._position
        count = max(0, int(np.ceil(available / step)))
        if not count:
            return np.empty(0, dtype=np.float32)
        positions = self._position + np.arange(count, dtype=np.float64) * step
        left = positions.astype(np.int64)
        fraction = positions - left
        output = self._buffer[left] * (1.0 - fraction) + self._buffer[left + 1] * fraction
        self._position = float(positions[-1] + step)
        # Keep the last source sample so interpolation across the next browser
        # block uses the same phase and boundary pair as one continuous buffer.
        drop = min(int(self._position), self._buffer.size - 1)
        if drop:
            self._buffer = self._buffer[drop:]
            self._position -= drop
        return output.astype(np.float32)


class RecognizerFactory:
    def __init__(self, model_dir: Path | None = None, vad_dir: Path | None = None) -> None:
        self.model_dir = model_dir or _configured_path(
            "X_ASR_MODEL_DIR",
            ROOT_DIR / "models" / "x-asr" / "asr",
            DEFAULT_X_ASR_DEMO / "models" / "asr",
        )
        self.vad_dir = vad_dir or _configured_path(
            "X_ASR_VAD_DIR",
            ROOT_DIR / "models" / "x-asr" / "firered_vad",
            DEFAULT_X_ASR_DEMO / "models" / "firered_vad",
        )
        self._recognizer: sherpa_onnx.OnlineRecognizer | None = None
        self._load_lock = threading.Lock()
        self.decode_lock = threading.Lock()

    def get(self) -> sherpa_onnx.OnlineRecognizer:
        with self._load_lock:
            if self._recognizer is None:
                files = self._model_files()
                self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                    **{key: str(value) for key, value in files.items()},
                    num_threads=2,
                    provider="cpu",
                    decoding_method="greedy_search",
                    model_type="zipformer2",
                    enable_endpoint_detection=False,
                )
            return self._recognizer

    def create_vad(self) -> FireRedVad:
        required = [self.vad_dir / "model.pth.tar", self.vad_dir / "cmvn.ark"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("FireRedVAD 模型文件缺失: " + ", ".join(missing))
        return FireRedVad(self.vad_dir)

    def _model_files(self) -> dict[str, Path]:
        files = {
            "tokens": self.model_dir / "tokens.txt",
            "encoder": self.model_dir / "encoder-960ms.onnx",
            "decoder": self.model_dir / "decoder-960ms.onnx",
            "joiner": self.model_dir / "joiner-960ms.onnx",
        }
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("X-ASR 模型文件缺失: " + ", ".join(missing))
        return files


class StreamingAsr:
    """X-ASR + FireRedVAD session compatible with the application's existing API."""

    def __init__(self, factory: RecognizerFactory) -> None:
        self.recognizer = factory.get()
        self.vad = factory.create_vad()
        self.decode_lock = factory.decode_lock
        self.stream = None
        self.active = False
        self.final_parts: list[str] = []
        self.segments: list[dict[str, float | str]] = []
        self.partial = ""
        self.audio_seconds = 0.0
        self.segment_start_seconds = 0.0
        self.preroll_seconds = 0.7
        self.tail_pad_seconds = 1.0
        self._preroll: deque[np.ndarray] = deque(
            maxlen=max(1, int(self.preroll_seconds * SAMPLE_RATE / VAD_WINDOW))
        )
        self._window_buffer = np.empty(0, dtype=np.float32)
        self._resampler = StreamingLinearResampler()
        self._lock = threading.Lock()

    def accept(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple[str, str | None]:
        with self._lock:
            converted = self._resampler.process(samples, sample_rate)
            if converted.size:
                self._window_buffer = np.concatenate((self._window_buffer, converted))
            final = None
            while self._window_buffer.size >= VAD_WINDOW:
                window = self._window_buffer[:VAD_WINDOW]
                self._window_buffer = self._window_buffer[VAD_WINDOW:]
                self.audio_seconds += VAD_WINDOW / SAMPLE_RATE
                committed = self._process_window(window)
                if committed:
                    final = committed
            return self.partial, final

    def _process_window(self, window: np.ndarray) -> str | None:
        self.vad.accept_waveform(window)
        speech = self.vad.is_speech_detected()
        if speech and not self.active:
            self.active = True
            self.stream = self.recognizer.create_stream()
            self.segment_start_seconds = max(
                0.0, self.audio_seconds - (len(self._preroll) + 1) * VAD_WINDOW / SAMPLE_RATE
            )
            for previous in self._preroll:
                self.stream.accept_waveform(SAMPLE_RATE, previous)

        if self.active and self.stream is not None:
            self.stream.accept_waveform(SAMPLE_RATE, window)
            self._decode_ready()
            self.partial = normalize_cjk(self.recognizer.get_result(self.stream).strip())

        final = None
        if self.active and not speech:
            final = self._finalize_active()

        self._preroll.append(window.copy())
        return final

    def _decode_ready(self) -> None:
        assert self.stream is not None
        with self.decode_lock:
            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)

    def _finalize_active(self) -> str | None:
        if not self.active or self.stream is None:
            return None
        self.stream.accept_waveform(
            SAMPLE_RATE, np.zeros(int(self.tail_pad_seconds * SAMPLE_RATE), dtype=np.float32)
        )
        self.stream.input_finished()
        self._decode_ready()
        text = normalize_cjk(self.recognizer.get_result(self.stream).strip())
        final = self._commit_final(text, self.segment_start_seconds, self.audio_seconds)
        self.partial = ""
        self.active = False
        self.stream = None
        return final

    def finish(self) -> str | None:
        with self._lock:
            if self._window_buffer.size:
                self.audio_seconds += self._window_buffer.size / SAMPLE_RATE
                padded = np.pad(self._window_buffer, (0, VAD_WINDOW - self._window_buffer.size))
                self._window_buffer = np.empty(0, dtype=np.float32)
                self._process_window(padded.astype(np.float32))
            silence_windows = int(np.ceil(1.1 * SAMPLE_RATE / VAD_WINDOW))
            final = None
            for _ in range(silence_windows):
                committed = self._process_window(np.zeros(VAD_WINDOW, dtype=np.float32))
                if committed:
                    final = committed
            if self.active:
                final = self._finalize_active()
            self.partial = ""
            return final

    def _commit_final(self, text: str, start: float, end: float) -> str | None:
        text = text.strip()
        if not text:
            return None
        self.final_parts.append(text)
        self.segments.append({"start": round(start, 2), "end": round(end, 2), "text": text})
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
        "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    session = StreamingAsr(factory)
    processed = 0
    chunk_bytes = SAMPLE_RATE * 4 // 2
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
                "progress": min(99, round(processed / (duration * SAMPLE_RATE) * 100)) if duration else 0,
            }
            if final:
                event["final"] = final
            on_event(event)
        error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if process.wait() != 0:
            raise RuntimeError(error.strip() or "FFmpeg 无法解码该音频")
        session.finish()
        on_event({
            "type": "transcription.progress", "transcript": session.transcript,
            "segments": session.segments, "partial": "", "progress": 100,
        })
        return session.transcript
    finally:
        if process.poll() is None:
            process.kill()


def _probe_duration(path: Path) -> float:
    command = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        return 0
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 0
