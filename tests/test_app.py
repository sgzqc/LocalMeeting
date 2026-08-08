from fastapi.testclient import TestClient
import threading

import numpy as np

from app.main import app
from app.services.asr import StreamingAsr


def test_home_and_health():
    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "会议助手" in home.text
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"


def test_rejects_unsupported_upload():
    with TestClient(app) as client:
        response = client.post(
            "/api/files",
            headers={"x-filename": "notes.txt"},
            content=b"not audio",
        )
        assert response.status_code == 415


def test_endpoint_segments_keep_natural_repeated_boundary():
    session = StreamingAsr.__new__(StreamingAsr)
    session.final_parts = []
    session.segments = []

    session._commit_final("已经过去了八年", 0.0, 5.0)
    session._commit_final("八年时间了", 6.0, 9.0)

    assert session.final_parts == ["已经过去了八年", "八年时间了"]
    assert session.segments[1]["text"] == "八年时间了"


def test_endpoint_replays_boundary_audio_into_a_fresh_stream():
    class Stream:
        def __init__(self):
            self.accepted = []

        def accept_waveform(self, sample_rate, samples):
            self.accepted.append((sample_rate, samples.copy()))

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
            return "上一句"

        def is_endpoint(self, stream):
            return True

    session = StreamingAsr.__new__(StreamingAsr)
    session.recognizer = Recognizer()
    session.stream = session.recognizer.create_stream()
    session.final_parts = []
    session.segments = []
    session.partial = ""
    session.audio_seconds = 0.0
    session.segment_start_seconds = 0.0
    session.pre_roll_seconds = 0.2
    session._pre_roll = np.empty(0, dtype=np.float32)
    session._lock = threading.Lock()

    samples = np.arange(4096, dtype=np.float32)
    session.accept(samples, sample_rate=48000)

    assert len(session.recognizer.streams) == 2
    replayed = session.recognizer.streams[1].accepted[0]
    assert replayed[0] == 48000
    assert np.array_equal(replayed[1], samples)
