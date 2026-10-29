# SPDX-License-Identifier: Apache-2.0
"""Record OpenAI Realtime, OpenAI Live and Gemini Live sessions in the trace vocabulary
of the sglang wire."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import numpy as np
import websockets
from pydantic import JsonValue
from scipy.signal import resample_poly

from benchmarks.duplex.client import (
    FRAME_EXCERPT_CHARS,
    MAX_MESSAGE_BYTES,
    PACKET_BYTES,
    POST_CLOSE_SECONDS,
    SAMPLE_RATE,
    SEND_RECEIPTS_FILE,
    SendReceipt,
    TurnDetection,
    ZeroBufferPlayer,
    profile_protocol,
    session_cuts,
)
from benchmarks.duplex.profiles import PROFILES, ProfileName, Wire

ProviderWire = Literal["openai-realtime", "openai-live", "gemini-live"]
WIRE_FILE = "wire.jsonl"
RTT_FILE = "rtt.json"
PING_INTERVAL_S = 1.0
LIVE_QUIET_S = 2.0
# note (wenyao): Live streams output audio continuously, digital silence between
# replies (int16 RMS 0 to 89 in a real session, speech 125 and up), so only a delta
# above this RMS counts as speech for the end-of-input wait.
LIVE_SPEECH_RMS = 100.0
LIVE_CLOSE_WAIT_S = 15.0
DEFAULT_MODELS: dict[ProviderWire, str] = {
    "openai-realtime": "gpt-realtime-2.1",
    "openai-live": "gpt-live-1",
    "gemini-live": "gemini-3.8-live",
}
DEFAULT_VOICES: dict[ProviderWire, str] = {
    "openai-realtime": "marin",
    "openai-live": "marin",
    "gemini-live": "Puck",
}
ENV_PREFIXES: dict[ProviderWire, str] = {
    "openai-realtime": "OPENAI",
    "openai-live": "OPENAI",
    "gemini-live": "GEMINI",
}
END_OF_INPUT_RULES: dict[ProviderWire, str] = {
    "openai-realtime": "after the input end, at most end_wait_s until every "
    "response.created has its response.done and no speech_started is open without "
    "speech_stopped, then a WebSocket close with code 1000",
    "openai-live": f"after the input end, at most end_wait_s until no speech output "
    f"audio (a delta with int16 RMS above {LIVE_SPEECH_RMS}) arrived for "
    f"{LIVE_QUIET_S}s, then session.close and at most {LIVE_CLOSE_WAIT_S}s for "
    "session.closed",
    "gemini-live": "after the input end, at most end_wait_s until every model turn has "
    "ended by turnComplete or interrupted and no voiceActivity ACTIVITY_START is open "
    "without ACTIVITY_END, then a WebSocket close with code 1000",
}
GEMINI_INPUT_MIME_TYPE = f"audio/pcm;rate={SAMPLE_RATE}"
GEMINI_OUTPUT_RATE = 24000
GEMINI_OUTPUT_MIME_TYPE = f"audio/pcm;rate={GEMINI_OUTPUT_RATE}"
GEMINI_VAD_FIELDS = {
    "silence_duration_ms": "silenceDurationMs",
    "prefix_padding_ms": "prefixPaddingMs",
}
GEMINI_ACTIVITY_EVENTS = {
    "ACTIVITY_START": "input_audio_buffer.speech_started",
    "ACTIVITY_END": "input_audio_buffer.speech_stopped",
}


@dataclass(frozen=True)
class Provider:
    """Provider settings read from the environment, OPENAI_* or GEMINI_* by wire; the
    key reaches only the socket and the model probe."""

    api_key: str = field(repr=False)
    model: str
    voice: str
    instructions: str | None

    @classmethod
    def from_env(cls, wire: ProviderWire) -> Provider:
        prefix = ENV_PREFIXES[wire]
        api_key = os.environ.get(f"{prefix}_API_KEY")
        if not api_key:
            raise ValueError(f"{wire} sessions require {prefix}_API_KEY")
        else:
            pass
        instructions_file = os.environ.get(f"{prefix}_DUPLEX_INSTRUCTIONS_FILE")
        return cls(
            api_key=api_key,
            model=os.environ.get(f"{prefix}_DUPLEX_MODEL") or DEFAULT_MODELS[wire],
            voice=os.environ.get(f"{prefix}_DUPLEX_VOICE") or DEFAULT_VOICES[wire],
            instructions=(
                Path(instructions_file).read_text(encoding="utf-8")
                if instructions_file
                else None
            ),
        )


def provider_manifest(profile: ProfileName, url: str) -> dict[str, JsonValue] | None:
    """What a run manifest records about the provider; None for the sglang wire."""
    wire = PROFILES[profile].wire
    if wire == "sglang":
        return None
    else:
        provider = Provider.from_env(wire)
        return {
            "wire": wire,
            "url": DIALECTS[wire].connect_url(url, provider),
            "model": provider.model,
            "voice": provider.voice,
            "instructions_sha256": (
                hashlib.sha256(provider.instructions.encode()).hexdigest()
                if provider.instructions is not None
                else None
            ),
            "ping_interval_s": PING_INTERVAL_S,
            "admission_retries": 0,
            "end_of_input": END_OF_INPUT_RULES[wire],
        }


def served_model(profile: ProfileName, url: str) -> dict[str, JsonValue]:
    """GET the model card with the key: the HTTP status and the id it answers, which
    show that the account serves the model without recording its card."""
    wire = PROFILES[profile].wire
    assert wire != "sglang"
    dialect = DIALECTS[wire]
    provider = Provider.from_env(wire)
    parts = urlsplit(url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    request = urllib.request.Request(
        urlunsplit(
            (
                scheme,
                parts.netloc,
                dialect.model_path.format(model=quote(provider.model, safe="")),
                "",
                "",
            )
        ),
        headers=dialect.auth_headers(provider),
    )
    # note (wenyao): A failed probe is recorded; it must not block the benchmark.
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            card = json.load(response)
        return {
            "status": response.status,
            "id": card.get(dialect.model_card_id) if isinstance(card, dict) else None,
        }
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "id": None}
    except (OSError, ValueError) as exc:
        return {"status": None, "id": None, "error": f"{type(exc).__name__}: {exc}"}


@dataclass
class SessionState:
    """What the receive loop has observed, for the driver's waits."""

    changed: asyncio.Condition = field(default_factory=asyncio.Condition)
    latest: dict[str, dict[str, JsonValue]] = field(default_factory=dict)
    open_responses: set[str] = field(default_factory=set)
    speech_open: bool = False
    last_speech_s: float = -math.inf
    closing: bool = False

    async def until(self, ready: Callable[[], bool]) -> None:
        async with self.changed:
            await self.changed.wait_for(ready)


def realtime_url(url: str, provider: Provider) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query.setdefault("model", provider.model)
    return urlunsplit(parts._replace(query=urlencode(query)))


def realtime_configure(
    provider: Provider, turn_detection: TurnDetection | None
) -> dict[str, JsonValue]:
    assert turn_detection is not None
    pcm24 = {"type": "audio/pcm", "rate": 24000}
    return {
        "type": "realtime",
        "output_modalities": ["audio"],
        **(
            {"instructions": provider.instructions}
            if provider.instructions is not None
            else {}
        ),
        "audio": {
            "input": {
                "format": pcm24,
                "turn_detection": turn_detection.model_dump(mode="json"),
            },
            "output": {"format": pcm24, "voice": provider.voice},
        },
    }


def live_configure(
    provider: Provider, turn_detection: TurnDetection | None
) -> dict[str, JsonValue]:
    # note (wenyao): The delegation guide documents no default, so client mode (no
    # backend, no tools) is chosen explicitly at creation.
    return {
        "model": provider.model,
        "delegation": {"type": "client"},
        **(
            {"instructions": provider.instructions}
            if provider.instructions is not None
            else {}
        ),
        "audio": {
            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
            "output": {"voice": provider.voice},
        },
    }


async def responses_settled(state: SessionState) -> None:
    await state.until(lambda: not state.open_responses and not state.speech_open)


async def speech_quiet(state: SessionState) -> None:
    while (remaining_s := state.last_speech_s + LIVE_QUIET_S - time.perf_counter()) > 0:
        await asyncio.sleep(remaining_s)


LIVE_CANONICAL_TYPES = {
    "session.output_audio.delta": "response.output_audio.delta",
    "session.output_transcript.delta": "response.output_audio_transcript.delta",
}


def live_translate(event: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    event_type = event.get("type")
    canonical_type = (
        LIVE_CANONICAL_TYPES.get(event_type) if isinstance(event_type, str) else None
    )
    if canonical_type is None:
        return [event]
    else:
        return [
            {
                **event,
                "type": canonical_type,
                "response_id": None,
                "item_id": None,
                "wire_type": event["type"],
            }
        ]


def gemini_configure(
    provider: Provider, turn_detection: TurnDetection | None
) -> dict[str, JsonValue]:
    """The BidiGenerateContent setup: audio out with both transcriptions, and the
    server_vad fields that automatic activity detection has."""
    assert turn_detection is not None
    settings = turn_detection.model_dump(mode="json")
    if settings.pop("type") != "server_vad":
        raise ValueError("gemini-live takes turns by server_vad only")
    else:
        pass
    detection = {
        GEMINI_VAD_FIELDS[name]: settings.pop(name)
        for name in [*settings]
        if name in GEMINI_VAD_FIELDS
    }
    interrupts = settings.pop("interrupt_response", True)
    if settings:
        raise ValueError(f"gemini-live has no turn_detection fields {sorted(settings)}")
    else:
        pass
    return {
        "model": f"models/{provider.model}",
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": provider.voice}}
            },
        },
        **(
            {"systemInstruction": {"parts": [{"text": provider.instructions}]}}
            if provider.instructions is not None
            else {}
        ),
        "realtimeInputConfig": {
            "automaticActivityDetection": detection,
            "activityHandling": (
                "START_OF_ACTIVITY_INTERRUPTS" if interrupts else "NO_INTERRUPTION"
            ),
        },
        "inputAudioTranscription": {},
        "outputAudioTranscription": {},
    }


def gemini_frame(event: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """The Gemini frame for a canonical client event; Gemini rejects unknown fields,
    so the event_id stays in the canonical trace."""
    if event["type"] == "session.update":
        return {"setup": event["session"]}
    elif event["type"] == "input_audio_buffer.append":
        return {
            "realtimeInput": {
                "audio": {"data": event["audio"], "mimeType": GEMINI_INPUT_MIME_TYPE}
            }
        }
    else:
        raise ValueError(f"gemini-live has no client event for {event['type']}")


def text_field(value: JsonValue) -> str | None:
    text = value.get("text") if isinstance(value, dict) else None
    return text if isinstance(text, str) and text else None


@dataclass
class GeminiTurns:
    """Gemini frames as canonical events: each model turn is one response that opens
    at its first audio and ends cancelled at interrupted or completed at turnComplete.
    Frames and flags with no canonical event pass through as gemini.<field>."""

    response_id: str | None = None
    interrupted_id: str | None = None
    # note (jeffro): outputTranscription can lead the turn's first audio part, so
    # text that arrives before a response opens joins the next one.
    unopened_transcript: str = ""
    transcripts: dict[str, str] = field(default_factory=dict)

    def done(
        self, response_id: str, status: str, usage: JsonValue
    ) -> dict[str, JsonValue]:
        """A response.done whose output carries the turn's transcript the way an
        OpenAI response does."""
        return {
            "type": "response.done",
            "response": {
                "id": response_id,
                "status": status,
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_audio",
                                "transcript": self.transcripts[response_id],
                            }
                        ],
                    }
                ],
                **({"usage": usage} if usage is not None else {}),
            },
        }

    def translate(self, frame: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
        events: list[dict[str, JsonValue]] = []
        usage = frame.get("usageMetadata")
        for name, value in frame.items():
            activity = value.get("type") if isinstance(value, dict) else None
            if name == "setupComplete":
                # note (jeffro): setupComplete is empty and Live output is fixed at
                # 24 kHz PCM16, so the acknowledgment declares that rate and every
                # audio part is held to it.
                events.append(
                    {
                        "type": "session.updated",
                        "session": {
                            "audio": {
                                "output": {
                                    "format": {
                                        "type": "audio/pcm",
                                        "rate": GEMINI_OUTPUT_RATE,
                                    }
                                }
                            }
                        },
                    }
                )
            elif name == "voiceActivity" and activity in GEMINI_ACTIVITY_EVENTS:
                events.append(
                    {
                        "type": GEMINI_ACTIVITY_EVENTS[activity],
                        "audio_offset": value.get("audioOffset"),
                    }
                )
            elif name == "serverContent" and isinstance(value, dict):
                events.extend(self.server_content(value, usage))
            else:
                events.append({"type": f"gemini.{name}", name: value})
        return [{**event, "event_id": uuid.uuid4().hex} for event in events]

    def server_content(
        self, content: dict[str, JsonValue], usage: JsonValue
    ) -> list[dict[str, JsonValue]]:
        events: list[dict[str, JsonValue]] = []
        model_turn = content.get("modelTurn")
        parts = model_turn.get("parts") if isinstance(model_turn, dict) else None
        for name, value in content.items():
            if name == "inputTranscription" and text_field(value) is not None:
                events.append(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "delta": text_field(value),
                    }
                )
            elif name == "modelTurn":
                for part in parts if isinstance(parts, list) else []:
                    inline = part.get("inlineData") if isinstance(part, dict) else None
                    audio = inline.get("data") if isinstance(inline, dict) else None
                    if isinstance(audio, str) and audio:
                        if inline.get("mimeType") != GEMINI_OUTPUT_MIME_TYPE:
                            raise ValueError(
                                f"audio part {inline.get('mimeType')!r} is not "
                                f"{GEMINI_OUTPUT_MIME_TYPE}"
                            )
                        else:
                            pass
                        if self.response_id is None:
                            self.response_id = uuid.uuid4().hex
                            self.interrupted_id = None
                            self.transcripts[self.response_id] = (
                                self.unopened_transcript
                            )
                            self.unopened_transcript = ""
                            events.append(
                                {
                                    "type": "response.created",
                                    "response": {
                                        "id": self.response_id,
                                        "status": "in_progress",
                                    },
                                }
                            )
                        else:
                            pass
                        events.append(
                            {
                                "type": "response.output_audio.delta",
                                "response_id": self.response_id,
                                "item_id": self.response_id,
                                "delta": audio,
                            }
                        )
                    else:
                        events.append({"type": "gemini.modelTurn.part", "part": part})
            elif name == "outputTranscription" and text_field(value) is not None:
                if self.response_id is None:
                    self.unopened_transcript += text_field(value)
                else:
                    self.transcripts[self.response_id] += text_field(value)
                events.append(
                    {
                        "type": "response.output_audio_transcript.delta",
                        "response_id": self.response_id,
                        "delta": text_field(value),
                    }
                )
            elif name == "interrupted" and value is True and self.response_id:
                events.append(self.done(self.response_id, "cancelled", None))
                self.response_id, self.interrupted_id = None, self.response_id
            elif name == "turnComplete" and value is True and self.response_id:
                events.append(self.done(self.response_id, "completed", usage))
                self.response_id = None
            elif name == "turnComplete" and value is True and self.interrupted_id:
                # note (jeffro): The usage of an interrupted turn arrives with its
                # turnComplete, so the cancelled response.done is reported again.
                events.append(self.done(self.interrupted_id, "cancelled", usage))
                self.interrupted_id = None
            else:
                events.append({"type": f"gemini.{name}", name: value})
        return events


@dataclass(frozen=True)
class Dialect:
    """How one provider wire speaks the session the sglang wire records."""

    input_rate: int
    connect_url: Callable[[str, Provider], str]
    auth_headers: Callable[[Provider], dict[str, str]]
    model_path: str
    model_card_id: str
    # note (wenyao): Realtime announces session.created before it accepts
    # session.update; Live takes session.start as the first frame.
    greeting: str | None
    configure_type: str
    configure: Callable[[Provider, TurnDetection | None], dict[str, JsonValue]]
    acknowledgment: str
    # note (wenyao): The configured settings the acknowledgment must echo: audio
    # format, and the turn detector (Realtime) or client delegation (Live).
    echoed: tuple[tuple[str, ...], ...]
    append_type: str
    # note (jeffro): The frame on the socket for each canonical client event.
    wire_frame: Callable[[dict[str, JsonValue]], dict[str, JsonValue]]
    settle: Callable[[SessionState], Awaitable[None]]
    close_type: str | None
    closed_type: str | None
    # note (jeffro): A new translator per session; one wire frame can carry several
    # canonical events, and Gemini numbers its model turns itself.
    translator: Callable[
        [], Callable[[dict[str, JsonValue]], list[dict[str, JsonValue]]]
    ]


DIALECTS: dict[ProviderWire, Dialect] = {
    "openai-realtime": Dialect(
        input_rate=24000,
        connect_url=realtime_url,
        auth_headers=lambda provider: {"Authorization": f"Bearer {provider.api_key}"},
        model_path="/v1/models/{model}",
        model_card_id="id",
        greeting="session.created",
        configure_type="session.update",
        configure=realtime_configure,
        acknowledgment="session.updated",
        echoed=(
            ("audio", "input", "format", "type"),
            ("audio", "input", "format", "rate"),
            ("audio", "output", "format", "type"),
            ("audio", "output", "format", "rate"),
            ("audio", "input", "turn_detection", "type"),
        ),
        append_type="input_audio_buffer.append",
        wire_frame=lambda event: event,
        settle=responses_settled,
        close_type=None,
        closed_type=None,
        translator=lambda: lambda event: [event],
    ),
    "openai-live": Dialect(
        input_rate=SAMPLE_RATE,
        connect_url=lambda url, provider: url,
        auth_headers=lambda provider: {"Authorization": f"Bearer {provider.api_key}"},
        model_path="/v1/models/{model}",
        model_card_id="id",
        greeting=None,
        configure_type="session.start",
        configure=live_configure,
        acknowledgment="session.started",
        echoed=(
            ("audio", "format", "type"),
            ("audio", "format", "rate"),
            ("delegation", "type"),
        ),
        append_type="session.input_audio.append",
        wire_frame=lambda event: event,
        settle=speech_quiet,
        close_type="session.close",
        closed_type="session.closed",
        translator=lambda: live_translate,
    ),
    "gemini-live": Dialect(
        input_rate=SAMPLE_RATE,
        connect_url=lambda url, provider: url,
        auth_headers=lambda provider: {"x-goog-api-key": provider.api_key},
        model_path="/v1beta/models/{model}",
        model_card_id="name",
        greeting=None,
        configure_type="session.update",
        configure=gemini_configure,
        acknowledgment="session.updated",
        # note (jeffro): setupComplete is empty, so it can echo nothing.
        echoed=(),
        append_type="input_audio_buffer.append",
        wire_frame=gemini_frame,
        settle=responses_settled,
        close_type=None,
        closed_type=None,
        translator=lambda: GeminiTurns().translate,
    ),
}


def echo_mismatches(
    dialect: Dialect, requested: JsonValue, echoed: JsonValue
) -> list[str]:
    """The session settings the request set that the acknowledgment does not echo."""

    def at(session: JsonValue, path: tuple[str, ...]) -> JsonValue:
        for key in path:
            session = session.get(key) if isinstance(session, dict) else None
        return session

    return [
        f"{'.'.join(path)} echoed {at(echoed, path)!r}, requested {at(requested, path)!r}"
        for path in dialect.echoed
        if at(requested, path) is not None and at(echoed, path) != at(requested, path)
    ]


def pcm_rms(audio: str) -> float:
    samples = np.frombuffer(base64.b64decode(audio), dtype="<i2").astype(np.float64)
    return float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0


def wire_slices(stream: bytes, bounds: list[tuple[int, int]], rate: int) -> list[bytes]:
    """The [start, end) sample slices of 16 kHz PCM16 at the wire rate, resampling the
    whole stream once so packet edges add no filter transients."""
    if rate == SAMPLE_RATE:
        return [stream[2 * start : 2 * end] for start, end in bounds]
    else:
        divisor = math.gcd(rate, SAMPLE_RATE)
        up, down = rate // divisor, SAMPLE_RATE // divisor
        resampled = resample_poly(np.frombuffer(stream, dtype="<i2"), up, down)
        pcm = np.clip(np.rint(resampled), -32768, 32767).astype("<i2").tobytes()
        return [
            pcm[2 * -(-start * up // down) : 2 * -(-end * up // down)]
            for start, end in bounds
        ]


async def run_openai_session(
    url: str,
    pcm: bytes,
    *,
    trace_path: Path,
    timeout_s: float,
    profile: ProfileName,
    turn_detection: TurnDetection | None,
    report_playback: bool,
    tail_silence_s: float,
    end_wait_s: float | None,
) -> None:
    """run_session for a provider wire: the canonical trace, plus wire.jsonl with every
    frame as sent or received and rtt.json with WebSocket ping round trips."""
    if not pcm or len(pcm) % 2:
        raise ValueError("Input must be nonempty PCM16")
    else:
        pass
    profile_protocol(profile, turn_detection)
    wire: Wire = PROFILES[profile].wire
    assert wire != "sglang"
    if report_playback and wire == "gemini-live":
        raise ValueError(
            "gemini-live has no truncate; its server cuts its own playout clock"
        )
    else:
        pass
    dialect = DIALECTS[wire]
    provider = Provider.from_env(wire)
    tail_bytes = round(tail_silence_s * SAMPLE_RATE) * 2
    input_s = (len(pcm) + tail_bytes) / (2 * SAMPLE_RATE)
    stream = pcm + bytes(tail_bytes)
    packet_offsets = [*range(0, len(pcm), PACKET_BYTES)] + [
        *range(len(pcm), len(stream), PACKET_BYTES)
    ]
    bounds = [
        (
            offset // 2,
            min(offset + PACKET_BYTES, len(pcm) if offset < len(pcm) else len(stream))
            // 2,
        )
        for offset in packet_offsets
    ]
    canonical_audio, wire_audio = (
        [
            base64.b64encode(chunk).decode("ascii")
            for chunk in wire_slices(stream, bounds, rate)
        ]
        for rate in (SAMPLE_RATE, dialect.input_rate)
    )
    output_rate = PROFILES[profile].output_sample_rate

    receipts: list[SendReceipt] = []
    rtt_samples_s: list[float] = []
    with (
        trace_path.open("x", encoding="utf-8", buffering=1) as trace_file,
        trace_path.with_name(WIRE_FILE).open(
            "x", encoding="utf-8", buffering=1
        ) as wire_file,
    ):

        def record(
            direction: Literal["send", "receive", "error"],
            event: dict[str, JsonValue],
            wire_event: dict[str, JsonValue] | None = None,
        ) -> float:
            observed_time_s = time.perf_counter()
            trace_file.write(
                json.dumps(
                    {"direction": direction, "time_s": observed_time_s, "event": event},
                    allow_nan=False,
                )
                + "\n"
            )
            if wire_event is not None:
                wire_file.write(
                    json.dumps(
                        {
                            "direction": direction,
                            "time_s": observed_time_s,
                            "event": wire_event,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
            else:
                pass
            return observed_time_s

        async def exchange() -> None:
            async with websockets.connect(
                dialect.connect_url(url, provider),
                additional_headers=dialect.auth_headers(provider),
                max_size=MAX_MESSAGE_BYTES,
                compression=None,
                ping_interval=None,
            ) as websocket:
                state = SessionState()
                translate = dialect.translator()
                player = ZeroBufferPlayer(
                    output_rate,
                    session_cuts(
                        PROFILES[profile].playout_cuts,
                        (
                            turn_detection.model_dump(mode="json")
                            if turn_detection is not None
                            else None
                        ),
                    ),
                )

                async def send(event: dict[str, JsonValue]) -> None:
                    event = {**event, "event_id": uuid.uuid4().hex}
                    wire_event = dialect.wire_frame(event)
                    record("send", event, wire_event)
                    await websocket.send(json.dumps(wire_event))

                # note (wenyao): Until the session is configured an error answers the
                # awaited event; later errors are recorded and the session goes on.
                async def until_received(event_type: str) -> dict[str, JsonValue]:
                    await state.until(
                        lambda: event_type in state.latest or "error" in state.latest
                    )
                    if event_type not in state.latest:
                        raise RuntimeError(
                            f"error before {event_type}: {state.latest['error']}"
                        )
                    else:
                        return state.latest[event_type]

                async def receive() -> None:
                    async for frame in websocket:
                        try:
                            # note (jeffro): Gemini sends its JSON in binary frames.
                            wire_event = json.loads(frame)
                            if not isinstance(wire_event, dict):
                                raise ValueError("server event must be a JSON object")
                            else:
                                pass
                            events = translate(wire_event)
                            if not all(
                                isinstance(event.get("type"), str) for event in events
                            ):
                                raise ValueError("server event type must be a string")
                            else:
                                pass
                        except ValueError as exc:
                            raise ValueError(
                                "malformed server frame "
                                f"{frame[:FRAME_EXCERPT_CHARS]!r}: "
                                f"{type(exc).__name__}: {exc}"
                            ) from exc
                        for index, event in enumerate(events):
                            received_s = record(
                                "receive", event, wire_event if index == 0 else None
                            )
                            event_type = event["type"]
                            response_id = (event.get("response") or {}).get("id")
                            if event_type == "response.created":
                                state.open_responses.add(response_id)
                            elif event_type == "response.done":
                                state.open_responses.discard(response_id)
                            elif event_type == "input_audio_buffer.speech_started":
                                state.speech_open = True
                            elif event_type == "input_audio_buffer.speech_stopped":
                                state.speech_open = False
                            elif (
                                event_type == "response.output_audio.delta"
                                and pcm_rms(event["delta"]) > LIVE_SPEECH_RMS
                            ):
                                state.last_speech_s = received_s
                            else:
                                pass
                            for truncate in (
                                player.played(event, received_s)
                                if report_playback
                                else []
                            ):
                                await send(
                                    {"type": "conversation.item.truncate", **truncate}
                                )
                            async with state.changed:
                                state.latest[event_type] = event
                                state.changed.notify_all()
                    if not state.closing:
                        raise RuntimeError(
                            "server closed the connection before the end of input "
                            f"(code {websocket.close_code}: {websocket.close_reason!r})"
                        )
                    else:
                        pass

                async def drive() -> None:
                    if dialect.greeting is not None:
                        await until_received(dialect.greeting)
                    else:
                        pass
                    session = dialect.configure(provider, turn_detection)
                    await send({"type": dialect.configure_type, "session": session})
                    acknowledgment = await until_received(dialect.acknowledgment)
                    mismatches = echo_mismatches(
                        dialect, session, acknowledgment.get("session")
                    )
                    if mismatches:
                        raise RuntimeError(
                            f"{dialect.acknowledgment}: {'; '.join(mismatches)}"
                        )
                    else:
                        pass
                    start_s = time.perf_counter()
                    for sequence, byte_offset in enumerate(packet_offsets):
                        await asyncio.sleep(
                            max(
                                0.0,
                                start_s
                                + byte_offset / (2 * SAMPLE_RATE)
                                - time.perf_counter(),
                            )
                        )
                        event_id = uuid.uuid4().hex
                        wire_event = dialect.wire_frame(
                            {
                                "type": dialect.append_type,
                                "event_id": event_id,
                                "audio": wire_audio[sequence],
                            }
                        )
                        send_started_s = record(
                            "send",
                            {
                                "type": "input_audio_buffer.append",
                                "event_id": event_id,
                                "audio": canonical_audio[sequence],
                                "sglang": {
                                    "seq": sequence,
                                    "t_start_ms": byte_offset / 2 / SAMPLE_RATE * 1000,
                                },
                            },
                            wire_event,
                        )
                        await websocket.send(json.dumps(wire_event))
                        receipts.append(
                            {
                                "event_id": event_id,
                                "seq": sequence,
                                "start_s": send_started_s,
                                "completed_s": time.perf_counter(),
                            }
                        )
                    await asyncio.sleep(
                        max(0.0, receipts[0]["start_s"] + input_s - time.perf_counter())
                    )
                    try:
                        await asyncio.wait_for(dialect.settle(state), end_wait_s)
                    except asyncio.TimeoutError:
                        raise RuntimeError(
                            f"{wire} session not finished within {end_wait_s}s of the "
                            "input end"
                        ) from None
                    state.closing = True
                    if dialect.close_type is None:
                        await websocket.close(code=1000)
                    else:
                        await send({"type": dialect.close_type})
                        try:
                            await asyncio.wait_for(
                                state.until(
                                    lambda: dialect.closed_type in state.latest
                                ),
                                LIVE_CLOSE_WAIT_S,
                            )
                        except asyncio.TimeoutError:
                            raise RuntimeError(
                                f"no {dialect.closed_type} within "
                                f"{LIVE_CLOSE_WAIT_S}s of {dialect.close_type}"
                            ) from None

                async def ping() -> None:
                    while True:
                        try:
                            pong = await websocket.ping()
                        except websockets.ConnectionClosed:
                            return
                        # note (jeffro): A pong is in the trace so a silent provider
                        # still proves its receiver alive after the input window.
                        pong.add_done_callback(
                            lambda done: (
                                (
                                    rtt_samples_s.append(done.result()),
                                    record(
                                        "receive",
                                        {
                                            "type": "transport.pong",
                                            "event_id": uuid.uuid4().hex,
                                            "rtt_s": done.result(),
                                        },
                                    ),
                                )
                                if not done.cancelled() and done.exception() is None
                                else None
                            )
                        )
                        await asyncio.sleep(PING_INTERVAL_S)

                receiver = asyncio.create_task(receive())
                driver = asyncio.create_task(drive())
                pinger = asyncio.create_task(ping())
                try:
                    done, _ = await asyncio.wait(
                        {receiver, driver}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if receiver in done:
                        await receiver
                        try:
                            await asyncio.wait_for(driver, POST_CLOSE_SECONDS)
                        except asyncio.TimeoutError:
                            record(
                                "error",
                                {"message": "driver stalled after the receive loop"},
                            )
                    else:
                        await driver
                        try:
                            await asyncio.wait_for(receiver, POST_CLOSE_SECONDS)
                        except asyncio.TimeoutError:
                            pass
                finally:
                    for task in (receiver, driver, pinger):
                        task.cancel()
                    await asyncio.gather(
                        receiver, driver, pinger, return_exceptions=True
                    )

        try:
            await asyncio.wait_for(exchange(), timeout=timeout_s)
        except asyncio.TimeoutError:
            record("error", {"message": f"Session timeout after {timeout_s}s"})
        except (
            OSError,
            websockets.WebSocketException,
            ValueError,
            RuntimeError,
        ) as exc:
            record("error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            trace_path.with_name(SEND_RECEIPTS_FILE).write_text(
                json.dumps({"appends": receipts}, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            trace_path.with_name(RTT_FILE).write_text(
                json.dumps(
                    {
                        "samples_s": rtt_samples_s,
                        **{
                            f"p{percent}_s": (
                                float(np.percentile(rtt_samples_s, percent))
                                if rtt_samples_s
                                else None
                            )
                            for percent in (50, 90)
                        },
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
