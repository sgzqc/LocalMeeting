import threading
from collections import deque

from fastapi.testclient import TestClient

import numpy as np

from app.main import app
from app.services.asr import StreamingAsr, StreamingLinearResampler, normalize_cjk


def test_home_and_health():
    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "会议助手" in home.text
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["asr_backend"] == "x_asr"
        assert health.json()["asr_chunk_ms"] == 960


def test_rejects_unsupported_upload():
    with TestClient(app) as client:
        response = client.post(
            "/api/files",
            headers={"x-filename": "notes.txt"},
            content=b"not audio",
        )
        assert response.status_code == 415


def test_live_session_is_ready_before_audio_configuration():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/live") as websocket:
            assert websocket.receive_json() == {"type": "status", "state": "initializing"}
            assert websocket.receive_json() == {"type": "status", "state": "ready"}
            websocket.send_json({"type": "audio.config", "sample_rate": 48000})
            assert websocket.receive_json() == {
                "type": "audio.configured",
                "sample_rate": 48000,
            }


def test_endpoint_segments_keep_natural_repeated_boundary():
    session = StreamingAsr.__new__(StreamingAsr)
    session.final_parts = []
    session.segments = []

    session._commit_final("已经过去了八年", 0.0, 5.0)
    session._commit_final("八年时间了", 6.0, 9.0)

    assert session.final_parts == ["已经过去了八年", "八年时间了"]
    assert session.segments[1]["text"] == "八年时间了"


def test_streaming_resampler_preserves_phase_across_audio_blocks():
    source = np.linspace(-1.0, 1.0, 4800, dtype=np.float32)
    whole = StreamingLinearResampler().process(source, 48000)

    chunked_resampler = StreamingLinearResampler()
    chunked = np.concatenate([
        chunked_resampler.process(source[:1301], 48000),
        chunked_resampler.process(source[1301:3077], 48000),
        chunked_resampler.process(source[3077:], 48000),
    ])

    assert whole.size == 1600
    assert np.allclose(chunked, whole, atol=1e-6)


def test_normalize_cjk_keeps_english_spaces():
    assert normalize_cjk("今 天 是 Monday , weather is good") == "今天是 Monday, weather is good"


def test_vad_falling_edge_commits_and_replays_preroll():
    class Stream:
        def __init__(self):
            self.accepted = []
            self.finished = False

        def accept_waveform(self, sample_rate, samples):
            self.accepted.append((sample_rate, samples.copy()))

        def input_finished(self):
            self.finished = True

    class Recognizer:
        def __init__(self):
            self.streams = []

        def create_stream(self):
            stream = Stream()
            self.streams.append(stream)
            return stream

        def is_ready(self, stream):
            return False

        def get_result(self, stream):
            return "今 天 是 Monday"

    class Vad:
        def __init__(self):
            self.states = iter([False, True, True, False])
            self.speech = False

        def accept_waveform(self, samples):
            self.speech = next(self.states)

        def is_speech_detected(self):
            return self.speech

    session = StreamingAsr.__new__(StreamingAsr)
    session.recognizer = Recognizer()
    session.vad = Vad()
    session.decode_lock = threading.Lock()
    session.stream = None
    session.active = False
    session.final_parts = []
    session.segments = []
    session.partial = ""
    session.audio_seconds = 0.0
    session.segment_start_seconds = 0.0
    session.preroll_seconds = 0.7
    session.tail_pad_seconds = 1.0
    session._preroll = deque(maxlen=21)
    session._window_buffer = np.empty(0, dtype=np.float32)
    session._resampler = StreamingLinearResampler()
    session._lock = threading.Lock()

    samples = np.arange(4 * 512, dtype=np.float32) / 4096
    partial, final = session.accept(samples)

    assert partial == ""
    assert final == "今天是 Monday"
    assert session.transcript == "今天是 Monday"
    assert len(session.recognizer.streams) == 1
    stream = session.recognizer.streams[0]
    assert stream.finished
    assert np.array_equal(stream.accepted[0][1], samples[:512])
    assert stream.accepted[-1][1].size == 16000
