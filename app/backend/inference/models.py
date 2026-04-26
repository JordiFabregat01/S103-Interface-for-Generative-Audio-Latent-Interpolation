from pydantic import BaseModel
from enum import Enum


class AudioElement(Enum, str):
    CAMPFIRE = "campfire"
    KEYBOARD = "keyboard"

class InterpolationElement(BaseModel):
    audio1: AudioElement
    audio2: AudioElement
    timeline_size: int
    stay_time: int
    stickyness: float
    play: bool
    NFE: int
    context_static: bool