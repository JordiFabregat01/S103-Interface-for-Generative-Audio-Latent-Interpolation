from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import uvicorn
from inference.methods import greet, get_inference_engine, render_interpolation_audio
from inference.models import InterpolationElement

app = FastAPI()

@app.on_event("startup")
def warm_inference_engine() -> None:
    try:
        get_inference_engine()
    except FileNotFoundError as exc:
        print(f"SCAPES inference engine not warmed at startup: {exc}")

@app.get("/")
def root():
    return {"msg": greet()}


@app.post("/interpolate")
def interpolate(payload: InterpolationElement):
    try:
        audio_bytes = render_interpolation_audio(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(content=audio_bytes, media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)