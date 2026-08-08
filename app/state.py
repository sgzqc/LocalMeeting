from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileJob:
    id: str
    filename: str
    path: Path
    status: str = "uploaded"
    progress: int = 0
    transcript: str = ""
    segments: list[dict] = field(default_factory=list)
    partial: str = ""
    summary: str = ""
    error: str = ""
    paused: threading.Event = field(default_factory=threading.Event, repr=False)
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)
    subscribers: set[asyncio.Queue] = field(default_factory=set, repr=False)

    def public(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "progress": self.progress,
            "transcript": self.transcript,
            "segments": self.segments,
            "partial": self.partial,
            "summary": self.summary,
            "error": self.error,
        }


class JobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, FileJob] = {}

    def create(self, filename: str, path: Path, job_id: str | None = None) -> FileJob:
        job = FileJob(id=job_id or uuid.uuid4().hex, filename=filename, path=path)
        self.jobs[job.id] = job
        return job

    def get(self, job_id: str) -> FileJob:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        return job
