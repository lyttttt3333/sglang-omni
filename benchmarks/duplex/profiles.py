# SPDX-License-Identifier: Apache-2.0
"""Model-specific contracts for the shared duplex session benchmark."""

from dataclasses import dataclass
from typing import Literal

ProfileName = Literal[
    "nemotron-voicechat-pr2188",
    "minicpmo-native-pr2377",
    "qwen3-omni-half-duplex",
    "turn-based",
    "openai-realtime",
    "openai-live",
]
DEFAULT_PROFILE: ProfileName = "nemotron-voicechat-pr2188"
# note (luojiaxuan): "native" is the full-duplex session protocol with sglang.*
# receipts; "legacy" is the turn-based conversation protocol, with `modalities` and a
# top-level `turn_detection`; "turn_based" is the session protocol with sglang.*
# receipts whose server takes turns by server VAD and answers each one.
Protocol = Literal["native", "legacy", "turn_based"]
# note (wenyao): protocol is the conversation semantics downstream replay branches on;
# wire is the dialect on the socket, so a commercial API can share a protocol.
Wire = Literal["sglang", "openai-realtime", "openai-live"]
# note (wenyao): The received events at which a zero-buffer client stops its playout;
# speech_started counts only while turn_detection.interrupt_response is not false and
# response.done only with status cancelled.
PlayoutCut = Literal[
    "output_audio_buffer.cleared",
    "input_audio_buffer.speech_started",
    "response.done",
]


@dataclass(frozen=True, kw_only=True)
class DuplexProfile:
    native_unit_ms: int
    output_sample_rate: int
    stop_requires_eos: bool
    continuous_output: bool
    protocol: Protocol
    wire: Wire = "sglang"
    playout_cuts: frozenset[PlayoutCut] = frozenset()


PROFILES: dict[ProfileName, DuplexProfile] = {
    "nemotron-voicechat-pr2188": DuplexProfile(
        native_unit_ms=80,
        output_sample_rate=22050,
        stop_requires_eos=True,
        continuous_output=True,
        protocol="native",
    ),
    "minicpmo-native-pr2377": DuplexProfile(
        native_unit_ms=1000,
        output_sample_rate=24000,
        stop_requires_eos=False,
        continuous_output=False,
        protocol="native",
    ),
    "qwen3-omni-half-duplex": DuplexProfile(
        native_unit_ms=20,
        output_sample_rate=24000,
        stop_requires_eos=False,
        continuous_output=False,
        protocol="legacy",
        playout_cuts=frozenset(
            {"output_audio_buffer.cleared", "input_audio_buffer.speech_started"}
        ),
    ),
    "turn-based": DuplexProfile(
        native_unit_ms=200,
        output_sample_rate=24000,
        stop_requires_eos=False,
        continuous_output=False,
        protocol="turn_based",
        playout_cuts=frozenset({"response.done"}),
    ),
    "openai-realtime": DuplexProfile(
        native_unit_ms=200,
        output_sample_rate=24000,
        stop_requires_eos=False,
        continuous_output=False,
        protocol="turn_based",
        wire="openai-realtime",
        # note (wenyao): GA streams a reply faster than real time, so most barge-ins
        # land after its response.done and only speech_started stops the queued audio.
        playout_cuts=frozenset({"input_audio_buffer.speech_started", "response.done"}),
    ),
    "openai-live": DuplexProfile(
        native_unit_ms=1000,
        output_sample_rate=16000,
        stop_requires_eos=False,
        continuous_output=True,
        protocol="native",
        wire="openai-live",
    ),
}
