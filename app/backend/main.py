from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from inference.constants import CACHE_DIR
from inference.methods import (
    greet,
    get_inference_engine,
    render_interpolation_to_file,
    render_timeline_to_file,
)
from inference.models import InterpolationElement, InterpolationSegment, RenderRequest
from inference.embeddings import get_sound_layout, resolve_audio_file
from inference.clap_search import search_by_text, search_similar
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


@app.get("/sounds/search")
def search_sounds(q: str, k: int = 8):
    """Rank library sounds by CLAP cosine similarity to a free-form text query.

    Declared **before** ``/sounds/{filename}`` so FastAPI doesn't treat the
    literal segment ``search`` as a filename.
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="query 'q' must be non-empty")
    if not 1 <= k <= 50:
        raise HTTPException(status_code=400, detail="k must be in [1, 50]")
    try:
        hits = search_by_text(q, k=k)
    except FileNotFoundError as exc:
        logger.warning(f"search cache missing: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"text search failed for query={q!r}: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="text search failed") from exc

    return [
        {
            "id": hit.point.id,
            "name": hit.point.name,
            "filename": hit.point.filename,
            "x": hit.point.x,
            "y": hit.point.y,
            "score": hit.score,
        }
        for hit in hits
    ]


@app.get("/sounds/{filename}/similar")
def similar_sounds(filename: str, k: int = 8):
    """Rank library sounds by CLAP cosine similarity to ``filename``.

    The query sound itself is excluded from the response. Declared **before**
    ``/sounds/{filename}`` so the more-specific ``/similar`` suffix wins the
    path match.
    """
    if not filename.strip():
        raise HTTPException(status_code=400, detail="filename must be non-empty")
    if not 1 <= k <= 50:
        raise HTTPException(status_code=400, detail="k must be in [1, 50]")
    try:
        hits = search_similar(filename, k=k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.warning(f"search cache missing: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"similar search failed for filename={filename!r}: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="similar search failed") from exc

    return [
        {
            "id": hit.point.id,
            "name": hit.point.name,
            "filename": hit.point.filename,
            "x": hit.point.x,
            "y": hit.point.y,
            "score": hit.score,
        }
        for hit in hits
    ]


@app.get("/sounds/{filename}")
def get_sound_audio(filename: str):
    try:
        path = resolve_audio_file(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="audio/wav", filename=path.name)


def _build_timeline_render_runner(payload: RenderRequest):
    """Create a closure that runs ``render_timeline_to_file`` for ``payload``."""

    def runner(state: JobState) -> None:
        output_path = JOBS_DIR / f"{state.job_id}.wav"
        state.result_path = output_path

        def on_progress(done: int, total: int) -> None:
            state.update_progress(done, total)

        render_timeline_to_file(
            payload,
            output_path,
            cancel_event=state.cancel_event,
            progress=on_progress,
        )

    return runner


@app.post("/render", status_code=202)
def render(payload: RenderRequest):
    """Enqueue an async timeline render. Returns immediately with a ``job_id``.

    Poll ``/jobs/{job_id}`` for status and fetch ``/jobs/{job_id}/result.wav``
    once the status is ``done``.
    """
    logger.info(
        "Received render request: %d segments [%s]",
        len(payload.segments),
        ", ".join(seg.type for seg in payload.segments),
    )
    state = job_store.submit(_build_timeline_render_runner(payload))
    return {"job_id": state.job_id}


@app.post("/interpolate", deprecated=True)
def interpolate(payload: InterpolationElement):
    """Deprecated: thin shim that forwards a single interpolation segment to /render.

    Now that /render is async, this also returns ``{"job_id": ...}``. Callers
    should poll ``/jobs/{job_id}`` and fetch ``/jobs/{job_id}/result.wav``.
    """
    logger.info(
        "Received (deprecated) interpolation request: %s <-> %s "
        "(distance_sec=%.3f, duration_sec=%s, context_mode=%s, nfe=%d)",
        payload.audio1.value,
        payload.audio2.value,
        payload.distance_sec,
        f"{payload.duration_sec:.3f}" if payload.duration_sec is not None else "auto",
        payload.context_mode,
        payload.nfe,
    )
    segment = InterpolationSegment(
        audio1=payload.audio1,
        audio2=payload.audio2,
        distance_sec=payload.distance_sec,
        duration_sec=payload.duration_sec,
        a_anchor_sec=payload.a_anchor_sec,
        b_anchor_sec=payload.b_anchor_sec,
        stay_time_sec=payload.stay_time_sec,
        stickyness=payload.stickyness,
        nfe=payload.nfe,
        context_mode=payload.context_mode,
        decode_method=payload.decode_method,
    )
    return render(RenderRequest(segments=[segment]))


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


@app.post("/render/async", status_code=202)
def render_async(payload: InterpolationElement):
    """Enqueue an async render for a single interpolation. Returns ``{job_id}``.

    Prefer :http:post:`/render` for new code — it accepts the full timeline
    payload. This endpoint is kept for callers that still post a bare
    :class:`InterpolationElement`.
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
