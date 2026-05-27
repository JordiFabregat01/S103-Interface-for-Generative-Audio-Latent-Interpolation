"""In-memory job store for async render endpoints.

Caveats:
  * Not durable across restarts.
  * Multi-worker uvicorn deployments will see "job not found" because workers
    do not share this dict. Run a single worker if you rely on this module.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_DONE = "done"
JOB_ERROR = "error"
JOB_CANCELLED = "cancelled"

_TERMINAL_STATUSES = frozenset({JOB_DONE, JOB_ERROR, JOB_CANCELLED})


@dataclass
class JobState:
    """All mutable state for a single render job.

    The ``cancel_event`` is exposed to the runner (e.g. a SCAPES interpolation
    loop that takes ``cancel_event``) so the job can stop cooperatively.
    """

    job_id: str
    status: str = JOB_QUEUED
    progress_done: int = 0
    progress_total: int = 0
    result_path: Optional[Path] = None
    result_bytes: int = 0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    # Guards every read/write of the public fields above. Threads update
    # progress while the HTTP layer reads the snapshot for /jobs/{id}.
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update_progress(self, done: int, total: int) -> None:
        with self._lock:
            self.progress_done = int(done)
            self.progress_total = int(total)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload: dict[str, Any] = {"status": self.status}
            if self.status in (JOB_QUEUED, JOB_RUNNING):
                payload["progress"] = {
                    "done": self.progress_done,
                    "total": self.progress_total,
                }
            elif self.status == JOB_DONE:
                payload["result"] = {
                    "url": f"/jobs/{self.job_id}/result.wav",
                    "bytes": self.result_bytes,
                }
            elif self.status in (JOB_ERROR, JOB_CANCELLED) and self.error:
                payload["error"] = self.error
            return payload


class JobStore:
    """Thread-safe registry of in-flight + recently completed render jobs.

    Each call to :meth:`submit` spawns a daemon thread that runs the supplied
    ``runner(JobState)``. The runner is responsible for setting
    ``state.result_path`` and updating progress via
    ``state.update_progress``; this store handles status transitions and
    cancellation bookkeeping around it.
    """

    def __init__(self, *, jobs_dir: Path, ttl_seconds: float = 30 * 60.0) -> None:
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None

    def start_cleanup(self, interval_seconds: float = 60.0) -> None:
        """Begin periodic eviction of expired jobs. Idempotent."""
        if self._cleanup_thread is not None:
            return
        thread = threading.Thread(
            target=self._cleanup_loop,
            args=(interval_seconds,),
            daemon=True,
            name="jobs-cleanup",
        )
        thread.start()
        self._cleanup_thread = thread

    def stop_cleanup(self) -> None:
        self._stop_event.set()

    def _cleanup_loop(self, interval: float) -> None:
        while not self._stop_event.wait(interval):
            try:
                self.cleanup_expired()
            except Exception:
                logger.exception("error during job cleanup")

    def cleanup_expired(self) -> int:
        """Drop terminal jobs older than ``ttl_seconds`` and unlink their files."""
        now = time.time()
        removed = 0
        with self._lock:
            for job_id, state in list(self._jobs.items()):
                if state.status not in _TERMINAL_STATUSES:
                    continue
                anchor = state.finished_at or state.created_at
                if now - anchor <= self.ttl_seconds:
                    continue
                self._jobs.pop(job_id, None)
                removed += 1
                if state.result_path is not None and state.result_path.exists():
                    try:
                        state.result_path.unlink()
                    except OSError as exc:
                        logger.warning(
                            "failed to delete result file %s: %s",
                            state.result_path,
                            exc,
                        )
        if removed:
            logger.info("evicted %d expired job(s)", removed)
        return removed

    def submit(self, runner: Callable[[JobState], None]) -> JobState:
        """Create a new job, spawn its worker thread, and return the state."""
        job_id = uuid.uuid4().hex
        state = JobState(job_id=job_id)
        with self._lock:
            self._jobs[job_id] = state
        thread = threading.Thread(
            target=self._run,
            args=(state, runner),
            daemon=True,
            name=f"job-{job_id[:8]}",
        )
        thread.start()
        return state

    def _run(self, state: JobState, runner: Callable[[JobState], None]) -> None:
        with state._lock:
            # `cancel` may have flipped us to cancelled while we were still
            # queued; respect that and never start the runner.
            if state.status == JOB_CANCELLED:
                return
            state.status = JOB_RUNNING
            state.started_at = time.time()

        try:
            runner(state)
        except Exception as exc:  # noqa: BLE001 - we want to surface anything
            cancelled = state.cancel_event.is_set()
            if cancelled:
                logger.info("job %s cancelled mid-run", state.job_id)
            else:
                logger.exception("job %s failed", state.job_id)
            with state._lock:
                state.status = JOB_CANCELLED if cancelled else JOB_ERROR
                state.error = "cancelled by client" if cancelled else (
                    str(exc) or exc.__class__.__name__
                )
                state.finished_at = time.time()
            return

        with state._lock:
            if state.cancel_event.is_set():
                state.status = JOB_CANCELLED
                state.error = "cancelled by client"
            else:
                state.status = JOB_DONE
                if (
                    state.result_path is not None
                    and state.result_path.exists()
                ):
                    state.result_bytes = state.result_path.stat().st_size
            state.finished_at = time.time()

    def get(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Signal cancellation. Returns True if the job exists, False otherwise."""
        with self._lock:
            state = self._jobs.get(job_id)
        if state is None:
            return False
        state.cancel_event.set()
        # If the worker hasn't started yet we can short-circuit here so the
        # runner is never invoked.
        with state._lock:
            if state.status == JOB_QUEUED:
                state.status = JOB_CANCELLED
                state.error = "cancelled by client"
                state.finished_at = time.time()
        return True
