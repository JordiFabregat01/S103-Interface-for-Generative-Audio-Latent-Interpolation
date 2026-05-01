from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
import uvicorn
from inference.methods import greet, get_inference_engine, render_interpolation_audio
from inference.models import InterpolationElement
from inference.embeddings import get_sound_layout, resolve_audio_file
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
    logger.info("FINISHED STARTUP EVENT")

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
    logger.info(f"Received interpolation request: {payload.audio1.value} <-> {payload.audio2.value} (Timeline: {payload.timeline_size}, NFE: {payload.NFE})")
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

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
