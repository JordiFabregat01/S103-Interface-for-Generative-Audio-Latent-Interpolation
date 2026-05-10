from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
import uvicorn
from inference.constants import CACHE_DIR
from inference.methods import (
    greet,
    get_inference_engine,
    render_interpolation_audio,
    render_interpolation_to_file,
)
from inference.models import InterpolationElement
from inference.embeddings import get_sound_layout, resolve_audio_file
from jobs import JobState, JobStore
import logging
import traceback


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


JOBS_DIR = CACHE_DIR / "jobs"
job_store = JobStore(jobs_dir=JOBS_DIR, ttl_seconds=30 * 60.0)


@app.on_event("startup")
def warm_inference_engine() -> None:
    logger.info("STARTING STARTUP EVENT")
    logger.info("Initializing inference engine (this may take a while if models need to be downloaded)...")
    try:
        get_inference_engine()
        logger.info("Done: Inference engine initialized successfully.")
    except Exception as exc:
        logger.critical(f"ERROR during SCAPES inference engine initialization: {exc}")
        logger.critical(traceback.format_exc())
    job_store.start_cleanup(interval_seconds=60.0)
    logger.info("FINISHED STARTUP EVENT")


@app.on_event("shutdown")
def stop_jobs() -> None:
    job_store.stop_cleanup()


@app.get("/")
def root():
    return {"msg": greet()}


@app.get("/sounds")
def list_sounds(refresh: bool = False):
    try:
        layout = get_sound_layout(force=refresh)
    except FileNotFoundError as exc:
        logger.warning(f"Data directory missing for /sounds: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Failed to compute sound layout: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to compute sound layout") from exc
    return [point.__dict__ for point in layout]


@app.get("/sounds/{filename}")
def get_sound_audio(filename: str):
    try:
        path = resolve_audio_file(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.post("/interpolate")
def interpolate(payload: InterpolationElement):
    """Synchronous render. Kept for backwards compatibility; new clients should
    use the async ``/render`` + polling flow."""
    logger.info(
        "Received interpolation request: %s <-> %s "
        "(distance_sec=%.3f, duration_sec=%s, context_mode=%s, nfe=%d)",
        payload.audio1.value,
        payload.audio2.value,
        payload.distance_sec,
        f"{payload.duration_sec:.3f}" if payload.duration_sec is not None else "auto",
        payload.context_mode,
        payload.nfe,
    )
    try:
        audio_bytes = render_interpolation_audio(payload)
        logger.info(f"Successfully generated {len(audio_bytes)} bytes of audio.")
    except FileNotFoundError as exc:
        logger.warning(f"File not found during interpolation: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Unexpected error during interpolation: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error during audio generation") from exc

    return Response(content=audio_bytes, media_type="audio/wav")


def _build_render_runner(payload: InterpolationElement):
    """Create a closure that runs ``render_interpolation_to_file`` for ``payload``."""

    def runner(state: JobState) -> None:
        output_path = JOBS_DIR / f"{state.job_id}.wav"
        state.result_path = output_path

        def on_progress(done: int, total: int) -> None:
            state.update_progress(done, total)

        render_interpolation_to_file(
            payload,
            output_path,
            cancel_event=state.cancel_event,
            progress=on_progress,
        )

    return runner


@app.post("/render", status_code=202)
def render(payload: InterpolationElement):
    """Enqueue an async render. Returns immediately with a ``job_id``.

    For the MVP this accepts a single :class:`InterpolationElement`. A future
    timeline-aware payload (clips + transitions) can plug in here without
    changing the polling/result endpoints.
    """
    logger.info(
        "Render job requested: %s <-> %s (distance_sec=%.3f, duration_sec=%s)",
        payload.audio1.value,
        payload.audio2.value,
        payload.distance_sec,
        f"{payload.duration_sec:.3f}" if payload.duration_sec is not None else "auto",
    )
    state = job_store.submit(_build_render_runner(payload))
    return {"job_id": state.job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    state = job_store.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="job not found")
    return state.snapshot()


@app.get("/jobs/{job_id}/result.wav")
def get_job_result(job_id: str):
    state = job_store.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="job not found")
    snapshot = state.snapshot()
    if snapshot["status"] != "done":
        raise HTTPException(
            status_code=404,
            detail=f"job not ready (status={snapshot['status']})",
        )
    if state.result_path is None or not state.result_path.exists():
        raise HTTPException(status_code=410, detail="result file no longer available")
    return FileResponse(
        state.result_path,
        media_type="audio/wav",
        filename=f"{job_id}.wav",
    )


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not job_store.cancel(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    return {"status": "cancelling"}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
