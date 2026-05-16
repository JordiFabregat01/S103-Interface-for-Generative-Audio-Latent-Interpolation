from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class AudioElement(str, Enum):
    ARTIC_WIND = "ArticWind"
    BEES = "Bees"
    BIRD_AMBIENCE = "BirdAmbience"
    BREAKING_WATER = "BreakingWater"
    BREEZE = "Breeze"
    CAMPFIRE = "camp_fire"
    CICADAS_CLEAN = "CicadasClean"
    CORNFIELD_WIND = "CornfieldWind"
    CRICKETS = "crickets"
    FOOTSTEPS = "footsteps"
    ICE_STORM = "IceStorm"
    INTENSE_BREEZE = "IntenseBreeze"
    KEYBOARD = "keyboard"
    LOON_CALL = "LoonCall"
    RAIN = "rain"
    RAIN_ON_LEAVES = "RainOnLeaves"
    SEA_WAVES = "sea_waves"
    SEAGULLS = "Seagulls"
    SLOW_RIVER = "SlowRiver"
    SNOW_STEPS = "SnowSteps"
    THUNDER_STORM = "ThunderStorm"
    UNDERWATER_FLOW = "UnderwaterFlow"
    WATERFALL = "waterfall"
    WATER_ON_ROCKS = "WaterOnRocks"
    WIND_AND_RAIN = "Wind&Rain"


ContextModeLiteral = Literal["auto", "static_first", "static_at_anchor", "dynamic"]


class InterpolationElement(BaseModel):
    """Clip-aware interpolation request.

    `distance_sec` encodes the geometric relationship between two clips:
      * `< 0` -> overlap of `|distance_sec|`. Default context_mode: `dynamic`.
      * `> 0` -> gap of `distance_sec`. Default context_mode: `static_at_anchor`.
      * `== 0` -> adjacent. `duration_sec` is required and used as the fade.

    `context_mode` defaults to `"auto"` so the typical drag-and-drop request
    gets the right mode for free; pass an explicit value to override geometry.
    """

    audio1: AudioElement
    audio2: AudioElement

    distance_sec: float = 0.0
    duration_sec: Optional[float] = Field(
        default=None,
        description="Required when distance_sec == 0; ignored otherwise.",
    )

    a_anchor_sec: float = 0.0
    b_anchor_sec: float = 0.0

    stay_time_sec: float = 0.0
    stickyness: float = 1.0
    nfe: int = 8
    context_mode: ContextModeLiteral = "auto"
    decode_method: str = "ola_smooth"

    @model_validator(mode="after")
    def _validate(self) -> "InterpolationElement":
        if self.distance_sec == 0 and (
            self.duration_sec is None or self.duration_sec <= 0
        ):
            raise ValueError(
                "adjacent clips (distance_sec == 0) need a positive `duration_sec`"
            )
        if self.nfe <= 0:
            raise ValueError("nfe must be > 0")
        if self.stickyness <= 0:
            raise ValueError("stickyness must be > 0")
        if self.stay_time_sec < 0:
            raise ValueError("stay_time_sec must be non-negative")
        return self
