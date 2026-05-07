from enum import Enum

from pydantic import BaseModel


class AudioElement(str, Enum):
    CAMPFIRE = "camp_fire"
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