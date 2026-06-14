from __future__ import annotations

from enum import Enum
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class AudioElement(str, Enum):
    ARTIC_WIND = "artic_wind"
    BEES = "bees"
    BEES_ANGRY = "bees_angry"
    BIRDS_1 = "Birds_1"
    BIRDS_2 = "Birds_2"
    BREAKING_WATER = "BreakingWater"
    BREEZE = "breeze"
    CAMPFIRE_HARD_CRACKLE = "camp_fire_hard_crackle"
    CAMPFIRE_SOFT_CRACKLE = "camp_fire_soft_crackle"
    CICADAS = "Cicadas"
    CORNFIELD_WIND = "CornfieldWind"
    FOOTSTEPS = "footsteps"
    ICE_STORM = "IceStorm"
    INTENSE_BREEZE_2 = "IntenseBreeze_2"
    KEYBOARD = "Keyboard"
    RAIN = "Rain"
    RAIN_ON_LEAVES = "RainOnLeaves"
    SEAGULLS_1 = "Seagulls_1"
    SEAGULLS_2 = "Seagulls_2"
    SEA_WAVES = "SeaWaves"
    SLOW_RIVER = "SlowRiver"
    SNOW_STEPS = "snow_steps"
    THUNDER_STORM_BIG = "thunder_storm_big_thunder"
    THUNDER_STORM_FAR = "thunder_storm_far_thunder"
    UNDERWATER_FLOW = "UnderwaterFlow"
    WATERFALL = "waterfall"
    WATER_ON_ROCKS = "WaterOnRocks"
    WIND_AND_RAIN = "wind&rain"
    CRICKETS = "crickets"


ContextModeLiteral = Literal["auto", "static_first", "static_at_anchor", "dynamic"]

# Which variant of a sound to use: "short" (clean cherrypick clip) or "long"
# (original longer recording). Disambiguates the stem collisions between the
# assets/short/ and assets/long/ folders. Defaults to "long" so older callers
# keep their original behavior.
KindLiteral = Literal["short", "long"]


class InterpolationElement(BaseModel):
    """Clip-aware interpolation request.

    `distance_sec` encodes the geometric relationship between two clips:
      * `< 0` -> overlap of `|distance_sec|`. Default context_mode: `dynamic`.
      * `> 0` -> gap of `distance_sec`. Default context_mode: `static_at_anchor`.
      * `== 0` -> adjacent. `duration_sec` is required and used as the fade.

    `context_mode` defaults to `"auto"` so the typical drag-and-drop request
    gets the right mode for free; pass an explicit value to override geometry.
    """

    audio1: str
    audio2: str
    audio1_kind: KindLiteral = "long"
    audio2_kind: KindLiteral = "long"

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


# --------------------------------------------------------------------------- #
# Timeline render: a composition of N clips, N-1 gaps (silence or interpolation)
# --------------------------------------------------------------------------- #


class ClipSegment(BaseModel):
    """A source WAV played for `duration` seconds.

    `filename` is the on-disk asset name as returned by `/sounds` (e.g.
    `"ArticWind.wav"`). It is resolved against the assets directory; the clip is
    trimmed to `duration` (and zero-padded if the file is shorter).
    """

    type: Literal["clip"] = "clip"
    filename: str
    kind: KindLiteral = "long"
    duration: float = Field(gt=0, description="Seconds of audio to emit.")


class SilenceSegment(BaseModel):
    """A gap rendered as true silence (zero samples).

    SCAPES still generates atoms across the gap to keep its autoregressive past
    buffer warm — the output is hard-zeroed after decoding — and the context fed
    to those atoms is split between the preceding and following sounds so the
    next clip starts on its own timbre. See ``build_unified_timeline_schedule``.
    """

    type: Literal["silence"] = "silence"
    duration: float = Field(gt=0, description="Seconds of silence to emit.")


class InterpolationSegment(BaseModel):
    """A gap rendered by interpolating between two clips.

    Carries the same knobs as `InterpolationElement`; `to_element()` adapts it
    so the existing `render_interpolation_audio()` path can be reused verbatim.
    """

    type: Literal["interpolation"] = "interpolation"

    audio1: str
    audio2: str
    audio1_kind: KindLiteral = "long"
    audio2_kind: KindLiteral = "long"

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

    def to_element(self) -> InterpolationElement:
        return InterpolationElement(
            audio1=self.audio1,
            audio2=self.audio2,
            audio1_kind=self.audio1_kind,
            audio2_kind=self.audio2_kind,
            distance_sec=self.distance_sec,
            duration_sec=self.duration_sec,
            a_anchor_sec=self.a_anchor_sec,
            b_anchor_sec=self.b_anchor_sec,
            stay_time_sec=self.stay_time_sec,
            stickyness=self.stickyness,
            nfe=self.nfe,
            context_mode=self.context_mode,
            decode_method=self.decode_method,
        )

    @classmethod
    def from_element(cls, element: InterpolationElement) -> "InterpolationSegment":
        return cls(
            audio1=element.audio1,
            audio2=element.audio2,
            audio1_kind=element.audio1_kind,
            audio2_kind=element.audio2_kind,
            distance_sec=element.distance_sec,
            duration_sec=element.duration_sec,
            a_anchor_sec=element.a_anchor_sec,
            b_anchor_sec=element.b_anchor_sec,
            stay_time_sec=element.stay_time_sec,
            stickyness=element.stickyness,
            nfe=element.nfe,
            context_mode=element.context_mode,
            decode_method=element.decode_method,
        )


Segment = Annotated[
    Union[ClipSegment, SilenceSegment, InterpolationSegment],
    Field(discriminator="type"),
]


class RenderRequest(BaseModel):
    """A full timeline composition: N clips with N-1 gaps between them."""

    segments: List[Segment]

    @model_validator(mode="after")
    def _validate(self) -> "RenderRequest":
        if not self.segments:
            raise ValueError("segments must contain at least one entry")
        return self
