# SPDX-License-Identifier: Apache-2.0
"""Record OpenAI Realtime and Live sessions in the trace vocabulary of the sglang wire."""

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

OpenAIWire = Literal["openai-realtime", "openai-live"]
WIRE_FILE = "wire.jsonl"
RTT_FILE = "rtt.json"
PING_INTERVAL_S = 1.0
LIVE_QUIET_S = 2.0
# note (wenyao): Live streams output audio continuously, digital silence between
# replies (int16 RMS 0 to 89 in a real session, speech 125 and up), so only a delta
# above this RMS counts as speech for the end-of-input wait.
LIVE_SPEECH_RMS = 100.0
LIVE_CLOSE_WAIT_S = 15.0
DEFAULT_MODELS: dict[OpenAIWire, str] = {
    "openai-realtime": "gpt-realtime-2.1",
    "openai-live": "gpt-live-1",
}
DEFAULT_VOICE = "marin"
END_OF_INPUT_RULES: dict[OpenAIWire, str] = {
    "openai-realtime": "after the input end, at most end_wait_s until every "
    "response.created has its response.done and no speech_started is open without "
    "speech_stopped, then a WebSocket close with code 1000",
    "openai-live": f"after the input end, at most end_wait_s until no speech output "
    f"audio (a delta with int16 RMS above {LIVE_SPEECH_RMS}) arrived for "
    f"{LIVE_QUIET_S}s, then session.close and at most {LIVE_CLOSE_WAIT_S}s for "
    "session.closed",
}


@dataclass(frozen=True)
class Provider:
    """OpenAI settings read from the environment; the key reaches only the socket."""

    api_key: str = field(repr=False)
    model: str
    voice: str
    instructions: str | None

    @classmethod
    def from_env(cls, wire: OpenAIWire) -> Provider:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"{wire} sessions require OPENAI_API_KEY")
        else:
            pass
        instructions_file = os.environ.get("OPENAI_DUPLEX_INSTRUCTIONS_FILE")
        return cls(
            api_key=api_key,
            model=os.environ.get("OPENAI_DUPLEX_MODEL") or DEFAULT_MODELS[wire],
            voice=os.environ.get("OPENAI_DUPLEX_VOICE") or DEFAULT_VOICE,
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
    """GET /v1/models/{model} with the key: the HTTP status and the id it answers,
    which show that the account serves the model without recording its card."""
    wire = PROFILES[profile].wire
    assert wire != "sglang"
    provider = Provider.from_env(wire)
    parts = urlsplit(url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    request = urllib.request.Request(
        urlunsplit(
            (
                scheme,
                parts.netloc,
                f"/v1/models/{quote(provider.model, safe='')}",
                "",
                "",
            )
        ),
        headers={"Authorization": f"Bearer {provider.api_key}"},
    )
    # note (wenyao): A failed probe is recorded; it must not block the benchmark.
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            card = json.load(response)
        return {
            "status": response.status,
            "id": card.get("id") if isinstance(card, dict) else None,
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


def live_translate(event: dict[str, JsonValue]) -> dict[str, JsonValue]:
    canonical_type = LIVE_CANONICAL_TYPES.get(event["type"])
    if canonical_type is None:
        return event
    else:
        return {
            **event,
            "type": canonical_type,
            "response_id": None,
            "item_id": None,
            "wire_type": event["type"],
        }


@dataclass(frozen=True)
class Dialect:
    """How one OpenAI wire speaks the session the sglang wire records."""

    input_rate: int
    connect_url: Callable[[str, Provider], str]
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
    settle: Callable[[SessionState], Awaitable[None]]
    close_type: str | None
    closed_type: str | None
    translate: Callable[[dict[str, JsonValue]], dict[str, JsonValue]]


DIALECTS: dict[OpenAIWire, Dialect] = {
    "openai-realtime": Dialect(
        input_rate=24000,
        connect_url=realtime_url,
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
        settle=responses_settled,
        close_type=None,
        closed_type=None,
        translate=lambda event: event,
    ),
    "openai-live": Dialect(
        input_rate=SAMPLE_RATE,
        connect_url=lambda url, provider: url,
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
        settle=speech_quiet,
        close_type="session.close",
        closed_type="session.closed",
        translate=live_translate,
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
    """run_session for an OpenAI wire: the canonical trace, plus wire.jsonl with every
    frame as sent or received and rtt.json with WebSocket ping round trips."""
    if not pcm or len(pcm) % 2:
        raise ValueError("Input must be nonempty PCM16")
    else:
        pass
    profile_protocol(profile, turn_detection)
    wire: Wire = PROFILES[profile].wire
    assert wire != "sglang"
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
                additional_headers={"Authorization": f"Bearer {provider.api_key}"},
                max_size=MAX_MESSAGE_BYTES,
                compression=None,
                ping_interval=None,
            ) as websocket:
                state = SessionState()
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
                    record("send", event, event)
                    await websocket.send(json.dumps(event))

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
                            wire_event = json.loads(frame)
                            if not isinstance(wire_event, dict):
                                raise ValueError("server event must be a JSON object")
                            elif not isinstance(wire_event.get("type"), str):
                                raise ValueError("server event type must be a string")
                            else:
                                pass
                        except ValueError as exc:
                            raise ValueError(
                                "malformed server frame "
                                f"{frame[:FRAME_EXCERPT_CHARS]!r}: "
                                f"{type(exc).__name__}: {exc}"
                            ) from exc
                        event = dialect.translate(wire_event)
                        received_s = record("receive", event, wire_event)
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
                            player.played(event, received_s) if report_playback else []
                        ):
                            await send(
                                {"type": "conversation.item.truncate", **truncate}
                            )
                        async with state.changed:
                            state.latest[event_type] = event
                            state.changed.notify_all()
                    if not state.closing:
                        raise RuntimeError(
                            "server closed the connection before the end of input"
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
                        wire_event = {
                            "type": dialect.append_type,
                            "event_id": event_id,
                            "audio": wire_audio[sequence],
                        }
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
                        pong.add_done_callback(
                            lambda done: (
                                rtt_samples_s.append(done.result())
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
