from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import uvicorn
from inference.methods import greet, get_inference_engine, render_interpolation_audio
from inference.models import InterpolationElement
import logging
import traceback


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


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