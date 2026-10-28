# SPDX-License-Identifier: Apache-2.0
"""In-process OpenAI Realtime (GA) and Live WebSocket servers that hold clients to the docs."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Annotated, Literal
from urllib.parse import parse_qs, urlsplit

import numpy as np
import websockets
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
)
from websockets.asyncio.server import ServerConnection
from websockets.http11 import Request, Response

REALTIME_PATH = "/v1/realtime"
LIVE_PATH = "/v1/live/sessions"
REALTIME_SAMPLE_RATE = 24000
LIVE_SAMPLE_RATES = (16000, 24000)
SERVER_VAD_DEFAULTS: dict[str, JsonValue] = {
    "type": "server_vad",
    "threshold": 0.5,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 500,
    "idle_timeout_ms": None,
    "create_response": True,
    "interrupt_response": True,
}
# note (wenyao): the docs give the server_vad threshold as a speech probability; the
# fakes gate on PCM16 frame RMS instead, so only silence_duration_ms is honored.
SPEECH_RMS = 500.0
VAD_FRAME_MS = 10
VOICE_AMPLITUDE = 6000
# note (wenyao): a real Live session streams near-silent frames (int16 RMS 0 to 89)
# between replies; a 40-peak tone has RMS 28.
QUIET_AMPLITUDE = 40
CHUNK_S = 0.1
REPLY_WORDS = ("sure", "here", "is", "a", "short", "answer")
MAX_MESSAGE_BYTES = 8 * 1024 * 1024


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ClientEvent(StrictModel):
    event_id: Annotated[str, Field(max_length=512)] | None = None


class AudioFormat(StrictModel):
    type: Literal["audio/pcm", "audio/pcmu", "audio/pcma"]
    rate: int | None = None


class ServerVad(StrictModel):
    type: Literal["server_vad"]
    threshold: float | None = None
    prefix_padding_ms: int | None = None
    silence_duration_ms: int | None = None
    idle_timeout_ms: int | None = None
    create_response: bool | None = None
    interrupt_response: bool | None = None


class SemanticVad(StrictModel):
    type: Literal["semantic_vad"]
    eagerness: Literal["low", "medium", "high", "auto"] | None = None
    create_response: bool | None = None
    interrupt_response: bool | None = None


class RealtimeAudioInput(StrictModel):
    format: AudioFormat | None = None
    noise_reduction: JsonValue = None
    transcription: JsonValue = None
    turn_detection: (
        Annotated[ServerVad | SemanticVad, Field(discriminator="type")] | None
    ) = None


class RealtimeAudioOutput(StrictModel):
    format: AudioFormat | None = None
    voice: str | dict[str, str] | None = None
    speed: float | None = None


class RealtimeAudio(StrictModel):
    input: RealtimeAudioInput | None = None
    output: RealtimeAudioOutput | None = None


class RealtimeSessionConfig(StrictModel):
    type: Literal["realtime"]
    output_modalities: list[Literal["text", "audio"]] | None = None
    instructions: str | None = None
    audio: RealtimeAudio | None = None
    model: str | None = None
    include: JsonValue = None
    max_output_tokens: int | Literal["inf"] | None = None
    parallel_tool_calls: bool | None = None
    reasoning: JsonValue = None
    prompt: JsonValue = None
    tool_choice: JsonValue = None
    tools: JsonValue = None
    tracing: JsonValue = None
    truncation: JsonValue = None


class SessionUpdate(ClientEvent):
    type: Literal["session.update"]
    session: RealtimeSessionConfig


class InputAudioBufferAppend(ClientEvent):
    type: Literal["input_audio_buffer.append"]
    audio: str


class ConversationItemTruncate(ClientEvent):
    type: Literal["conversation.item.truncate"]
    item_id: str
    content_index: int
    audio_end_ms: int


class LiveAudioFormat(StrictModel):
    type: Literal["audio/pcm", "audio/pcmu", "audio/pcma"]
    rate: int


class LiveAudioOutput(StrictModel):
    voice: str | dict[str, str]


class LiveAudio(StrictModel):
    format: LiveAudioFormat | None = None
    output: LiveAudioOutput | None = None


class LiveDelegation(StrictModel):
    type: Literal["client", "responses"]
    responses: JsonValue = None


class LiveSessionConfig(StrictModel):
    model: str
    instructions: str | None = None
    audio: LiveAudio | None = None
    delegation: LiveDelegation | None = None
    store: bool | None = None


class SessionStart(ClientEvent):
    type: Literal["session.start"]
    session: LiveSessionConfig


class SessionInputAudioAppend(ClientEvent):
    type: Literal["session.input_audio.append"]
    audio: str


class SessionClose(ClientEvent):
    type: Literal["session.close"]


REALTIME_EVENTS: TypeAdapter[
    SessionUpdate | InputAudioBufferAppend | ConversationItemTruncate
] = TypeAdapter(
    Annotated[
        SessionUpdate | InputAudioBufferAppend | ConversationItemTruncate,
        Field(discriminator="type"),
    ]
)
LIVE_EVENTS: TypeAdapter[SessionStart | SessionInputAudioAppend | SessionClose] = (
    TypeAdapter(
        Annotated[
            SessionStart | SessionInputAudioAppend | SessionClose,
            Field(discriminator="type"),
        ]
    )
)


def tone_pcm(
    seconds: float,
    sample_rate: int,
    frequency_hz: float = 220.0,
    amplitude: int = VOICE_AMPLITUDE,
) -> bytes:
    times_s = np.arange(round(seconds * sample_rate)) / sample_rate
    return (
        (amplitude * np.sin(2 * math.pi * frequency_hz * times_s))
        .astype("<i2")
        .tobytes()
    )


def utterance_pcm(segments: list[tuple[str, float]], sample_rate: int) -> bytes:
    """PCM16 of ("voice" | "silence", seconds) segments; voice clears the VAD gate."""
    return b"".join(
        (
            tone_pcm(seconds, sample_rate)
            if kind == "voice"
            else bytes(2 * round(seconds * sample_rate))
        )
        for kind, seconds in segments
    )


@dataclass(kw_only=True)
class EnergyVad:
    """Speech start at the first voiced frame; stop after silence_ms of unvoiced media."""

    sample_rate: int
    silence_ms: float
    is_speaking: bool = False
    media_ms: float = 0.0
    voice_end_ms: float = 0.0

    def feed(self, pcm: bytes) -> list[tuple[Literal["started", "stopped"], float]]:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
        frame_samples = self.sample_rate * VAD_FRAME_MS // 1000
        transitions: list[tuple[Literal["started", "stopped"], float]] = []
        for start in range(0, len(samples), frame_samples):
            frame = samples[start : start + frame_samples]
            frame_ms = len(frame) * 1000 / self.sample_rate
            if math.sqrt(float(np.mean(frame**2))) >= SPEECH_RMS:
                if not self.is_speaking:
                    transitions.append(("started", self.media_ms))
                    self.is_speaking = True
                self.voice_end_ms = self.media_ms + frame_ms
            elif (
                self.is_speaking
                and self.media_ms + frame_ms - self.voice_end_ms >= self.silence_ms
            ):
                transitions.append(("stopped", self.voice_end_ms))
                self.is_speaking = False
            self.media_ms += frame_ms
        return transitions


@dataclass(kw_only=True)
class FakeConnection:
    """One accepted socket: the handshake it presented and every frame both ways."""

    websocket: ServerConnection
    path: str
    authorization: str | None
    received: list[dict[str, JsonValue]] = field(default_factory=list)
    sent: list[dict[str, JsonValue]] = field(default_factory=list)
    input_pcm: bytearray = field(default_factory=bytearray)

    async def emit(self, event_type: str, **fields: JsonValue) -> None:
        event = {"type": event_type, "event_id": f"event_{uuid.uuid4().hex[:20]}"}
        event.update(fields)
        self.sent.append(event)
        await self.websocket.send(json.dumps(event))

    def errors(self) -> list[dict[str, JsonValue]]:
        return [event for event in self.sent if event["type"] == "error"]


def error_fields(
    code: str,
    message: str,
    param: str | None,
    client_event_id: str | None,
    id_field: str,
) -> dict[str, JsonValue]:
    return {
        "error": {
            "type": "invalid_request_error",
            "code": code,
            "message": message,
            "param": param,
            id_field: client_event_id,
        }
    }


def validation_error_fields(
    exc: ValidationError, raw: dict[str, JsonValue], id_field: str
) -> dict[str, JsonValue]:
    first = exc.errors(include_input=False)[0]
    code = {
        "extra_forbidden": "unknown_parameter",
        "missing": "missing_required_parameter",
        "union_tag_invalid": "invalid_event",
        "union_tag_not_found": "invalid_event",
    }.get(first["type"], "invalid_value")
    event_type = raw.get("type")
    param = ".".join(str(part) for part in first["loc"] if part != event_type)
    event_id = raw.get("event_id")
    return error_fields(
        code,
        first["msg"],
        param or None,
        event_id if isinstance(event_id, str) else None,
        id_field,
    )


@dataclass(kw_only=True)
class StrictFake:
    """Shared handshake: the documented path (404), the exact bearer key (401) and,
    for Realtime, the model query (400)."""

    api_key: str
    path: str
    connections: list[FakeConnection] = field(default_factory=list)

    def process_request(
        self, connection: ServerConnection, request: Request
    ) -> Response | None:
        target = urlsplit(request.path)
        if target.path != self.path:
            return connection.respond(HTTPStatus.NOT_FOUND, "unknown endpoint\n")
        elif request.headers.get("Authorization") != f"Bearer {self.api_key}":
            return connection.respond(HTTPStatus.UNAUTHORIZED, "invalid api key\n")
        elif self.path == REALTIME_PATH and not parse_qs(target.query).get("model"):
            return connection.respond(HTTPStatus.BAD_REQUEST, "model is required\n")
        else:
            return None

    def accept(self, websocket: ServerConnection) -> FakeConnection:
        connection = FakeConnection(
            websocket=websocket,
            path=websocket.request.path,
            authorization=websocket.request.headers.get("Authorization"),
        )
        self.connections.append(connection)
        return connection

    async def handler(self, websocket: ServerConnection) -> None:
        raise NotImplementedError

    @contextlib.asynccontextmanager
    async def serve(self) -> AsyncIterator[str]:
        """Yield the base ws:// URL of this fake's documented endpoint."""
        async with websockets.serve(
            self.handler,
            "127.0.0.1",
            0,
            process_request=self.process_request,
            max_size=MAX_MESSAGE_BYTES,
        ) as server:
            port = server.sockets[0].getsockname()[1]
            yield f"ws://127.0.0.1:{port}{self.path}"


@dataclass(kw_only=True)
class RealtimeReply:
    response_id: str
    item_id: str
    status: Literal["in_progress", "completed", "cancelled"] = "in_progress"
    delivered_s: float = 0.0
    audio_started_s: float | None = None
    task: asyncio.Task[None] | None = None

    def audible(self, now_s: float) -> bool:
        """Whether a client playing from the first delta at real time still plays."""
        return (
            self.audio_started_s is not None
            and now_s < self.audio_started_s + self.delivered_s
        )


@dataclass(kw_only=True)
class RealtimeFake(StrictFake):
    """Server VAD turns at 24 kHz; output_speed is reply audio seconds per wall second.

    As the real API does, a barge-in also cancels a completed reply whose audio a
    real-time player would still be playing: a second response.done, cancelled, with
    the same id and usage."""

    path: str = REALTIME_PATH
    reply_s: float = 1.0
    output_speed: float = 5.0
    first_audio_delay_s: float = 0.05

    async def handler(self, websocket: ServerConnection) -> None:
        connection = self.accept(websocket)
        model = parse_qs(urlsplit(connection.path).query)["model"][0]
        session: dict[str, JsonValue] = {
            "type": "realtime",
            "object": "realtime.session",
            "id": f"sess_{uuid.uuid4().hex[:20]}",
            "model": model,
            "output_modalities": ["audio"],
            "instructions": "",
            "max_output_tokens": "inf",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": REALTIME_SAMPLE_RATE},
                    "transcription": None,
                    "noise_reduction": None,
                    "turn_detection": dict(SERVER_VAD_DEFAULTS),
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": REALTIME_SAMPLE_RATE},
                    "voice": "alloy",
                    "speed": 1,
                },
            },
        }
        vad = EnergyVad(
            sample_rate=REALTIME_SAMPLE_RATE,
            silence_ms=SERVER_VAD_DEFAULTS["silence_duration_ms"],
        )
        replies: dict[str, RealtimeReply] = {}
        active: RealtimeReply | None = None
        user_item_id: str | None = None
        last_item_id: str | None = None

        async def finish(
            reply: RealtimeReply, reason: Literal["turn_detected"] | None
        ) -> None:
            ids = {"response_id": reply.response_id, "item_id": reply.item_id}
            await connection.emit(
                "response.output_audio.done", **ids, output_index=0, content_index=0
            )
            await connection.emit(
                "response.output_item.done",
                response_id=reply.response_id,
                output_index=0,
                item={
                    "id": reply.item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "status": "completed" if reason is None else "incomplete",
                    "role": "assistant",
                    "content": [{"type": "output_audio", "transcript": ""}],
                },
            )
            await done(reply, reason)

        async def done(
            reply: RealtimeReply, reason: Literal["turn_detected"] | None
        ) -> None:
            output_audio_tokens = round(reply.delivered_s * 20)
            await connection.emit(
                "response.done",
                response={
                    "object": "realtime.response",
                    "id": reply.response_id,
                    "status": reply.status,
                    "status_details": (
                        None
                        if reason is None
                        else {"type": reply.status, "reason": reason}
                    ),
                    "output": [],
                    "output_modalities": ["audio"],
                    "usage": {
                        "total_tokens": 100 + output_audio_tokens,
                        "input_tokens": 100,
                        "output_tokens": output_audio_tokens,
                        "input_token_details": {
                            "text_tokens": 60,
                            "audio_tokens": 40,
                            "image_tokens": 0,
                            "cached_tokens": 0,
                            "cached_tokens_details": {
                                "text_tokens": 0,
                                "audio_tokens": 0,
                                "image_tokens": 0,
                            },
                        },
                        "output_token_details": {
                            "text_tokens": 0,
                            "audio_tokens": output_audio_tokens,
                        },
                    },
                    "metadata": None,
                },
            )

        async def generate(reply: RealtimeReply, previous_item_id: str | None) -> None:
            ids = {"response_id": reply.response_id, "item_id": reply.item_id}
            item = {
                "id": reply.item_id,
                "object": "realtime.item",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            }
            await connection.emit(
                "response.output_item.added",
                response_id=reply.response_id,
                output_index=0,
                item=item,
            )
            await connection.emit(
                "conversation.item.added", previous_item_id=previous_item_id, item=item
            )
            await asyncio.sleep(self.first_audio_delay_s)
            pcm = tone_pcm(self.reply_s, REALTIME_SAMPLE_RATE, frequency_hz=440.0)
            chunk_bytes = round(CHUNK_S * REALTIME_SAMPLE_RATE) * 2
            started_s = time.perf_counter()
            for index, offset in enumerate(range(0, len(pcm), chunk_bytes)):
                await asyncio.sleep(
                    max(
                        0.0,
                        started_s
                        + index * CHUNK_S / self.output_speed
                        - time.perf_counter(),
                    )
                )
                chunk = pcm[offset : offset + chunk_bytes]
                if reply.audio_started_s is None:
                    reply.audio_started_s = time.perf_counter()
                else:
                    pass
                reply.delivered_s += len(chunk) / 2 / REALTIME_SAMPLE_RATE
                await connection.emit(
                    "response.output_audio.delta",
                    **ids,
                    output_index=0,
                    content_index=0,
                    delta=base64.b64encode(chunk).decode("ascii"),
                )
                if index < len(REPLY_WORDS):
                    await connection.emit(
                        "response.output_audio_transcript.delta",
                        **ids,
                        output_index=0,
                        content_index=0,
                        delta=REPLY_WORDS[index] + " ",
                    )
            reply.status = "completed"
            await finish(reply, None)

        async def reject(
            code: str, message: str, param: str | None, raw: dict[str, JsonValue]
        ) -> None:
            event_id = raw.get("event_id")
            await connection.emit(
                "error",
                **error_fields(
                    code,
                    message,
                    param,
                    event_id if isinstance(event_id, str) else None,
                    "event_id",
                ),
            )

        await connection.emit("session.created", session=session)
        try:
            async for frame in websocket:
                raw = json.loads(frame)
                connection.received.append(raw)
                try:
                    event = REALTIME_EVENTS.validate_python(raw)
                except ValidationError as exc:
                    await connection.emit(
                        "error", **validation_error_fields(exc, raw, "event_id")
                    )
                    continue
                if isinstance(event, SessionUpdate):
                    audio = event.session.audio
                    formats = {
                        f"session.audio.{direction}.format": (
                            getattr(audio, direction).format
                            if audio is not None
                            and getattr(audio, direction) is not None
                            else None
                        )
                        for direction in ("input", "output")
                    }
                    wrong_format = next(
                        (
                            param
                            for param, audio_format in formats.items()
                            if audio_format is not None
                            and (
                                audio_format.type != "audio/pcm"
                                or audio_format.rate not in (None, REALTIME_SAMPLE_RATE)
                            )
                        ),
                        None,
                    )
                    if wrong_format is not None:
                        await reject(
                            "invalid_value",
                            f"only audio/pcm at {REALTIME_SAMPLE_RATE} Hz is supported",
                            wrong_format,
                            raw,
                        )
                        continue
                    elif event.session.model not in (None, session["model"]):
                        await reject(
                            "invalid_value",
                            "model cannot be changed by session.update",
                            "session.model",
                            raw,
                        )
                        continue
                    update = event.session.model_dump(exclude_unset=True)
                    update_audio = update.pop("audio", {})
                    session.update(update)
                    for direction, settings in update_audio.items():
                        session["audio"][direction].update(settings)
                    turn_detection = session["audio"]["input"]["turn_detection"]
                    if (
                        audio is not None
                        and audio.input is not None
                        and "turn_detection" in audio.input.model_fields_set
                        and turn_detection is not None
                    ):
                        defaults = (
                            SERVER_VAD_DEFAULTS
                            if turn_detection["type"] == "server_vad"
                            else {
                                "eagerness": "auto",
                                "create_response": True,
                                "interrupt_response": True,
                            }
                        )
                        turn_detection = {
                            **defaults,
                            **{
                                name: value
                                for name, value in turn_detection.items()
                                if value is not None
                            },
                        }
                        session["audio"]["input"]["turn_detection"] = turn_detection
                        vad.silence_ms = turn_detection.get("silence_duration_ms", 500)
                    await connection.emit(
                        "session.updated",
                        session=json.loads(json.dumps(session)),
                    )
                elif isinstance(event, InputAudioBufferAppend):
                    pcm = base64.b64decode(event.audio, validate=True)
                    if len(pcm) % 2:
                        await reject(
                            "invalid_value", "PCM16 needs whole samples", "audio", raw
                        )
                        continue
                    connection.input_pcm.extend(pcm)
                    turn_detection = session["audio"]["input"]["turn_detection"]
                    if turn_detection is None:
                        continue
                    for transition, media_ms in vad.feed(pcm):
                        if transition == "started":
                            user_item_id = f"item_{uuid.uuid4().hex[:20]}"
                            await connection.emit(
                                "input_audio_buffer.speech_started",
                                audio_start_ms=max(
                                    0,
                                    round(media_ms)
                                    - turn_detection.get("prefix_padding_ms", 0),
                                ),
                                item_id=user_item_id,
                            )
                            if (
                                active is None
                                or not turn_detection["interrupt_response"]
                            ):
                                pass
                            elif active.status == "in_progress":
                                assert active.task is not None
                                active.task.cancel()
                                await asyncio.gather(
                                    active.task, return_exceptions=True
                                )
                                active.status = "cancelled"
                                await finish(active, "turn_detected")
                            elif active.status == "completed" and active.audible(
                                time.perf_counter()
                            ):
                                active.status = "cancelled"
                                await done(active, "turn_detected")
                            else:
                                pass
                        else:
                            assert user_item_id is not None
                            await connection.emit(
                                "input_audio_buffer.speech_stopped",
                                audio_end_ms=round(media_ms),
                                item_id=user_item_id,
                            )
                            await connection.emit(
                                "input_audio_buffer.committed",
                                previous_item_id=last_item_id,
                                item_id=user_item_id,
                            )
                            await connection.emit(
                                "conversation.item.added",
                                previous_item_id=last_item_id,
                                item={
                                    "id": user_item_id,
                                    "object": "realtime.item",
                                    "type": "message",
                                    "status": "completed",
                                    "role": "user",
                                    "content": [
                                        {"type": "input_audio", "transcript": None}
                                    ],
                                },
                            )
                            last_item_id = user_item_id
                            if turn_detection["create_response"] and (
                                active is None or active.status != "in_progress"
                            ):
                                active = RealtimeReply(
                                    response_id=f"resp_{uuid.uuid4().hex[:20]}",
                                    item_id=f"item_{uuid.uuid4().hex[:20]}",
                                )
                                await connection.emit(
                                    "rate_limits.updated",
                                    rate_limits=[
                                        {
                                            "name": "requests",
                                            "limit": 500,
                                            "remaining": 499,
                                            "reset_seconds": 30,
                                        },
                                        {
                                            "name": "tokens",
                                            "limit": 20000,
                                            "remaining": 19000,
                                            "reset_seconds": 30,
                                        },
                                    ],
                                )
                                await connection.emit(
                                    "response.created",
                                    response={
                                        "object": "realtime.response",
                                        "id": active.response_id,
                                        "status": "in_progress",
                                        "status_details": None,
                                        "output": [],
                                        "output_modalities": ["audio"],
                                        "usage": None,
                                    },
                                )
                                replies[active.item_id] = active
                                active.task = asyncio.create_task(
                                    generate(active, user_item_id)
                                )
                elif event.item_id not in replies:
                    await reject(
                        "item_not_found",
                        f"no assistant item {event.item_id}",
                        "item_id",
                        raw,
                    )
                elif (
                    event.content_index != 0
                    or event.audio_end_ms
                    > replies[event.item_id].delivered_s * 1000 + 1
                ):
                    await reject(
                        "invalid_value",
                        "audio_end_ms is past the item's audio",
                        "audio_end_ms",
                        raw,
                    )
                else:
                    await connection.emit(
                        "conversation.item.truncated",
                        item_id=event.item_id,
                        content_index=0,
                        audio_end_ms=event.audio_end_ms,
                    )
        finally:
            for reply in replies.values():
                if reply.task is not None:
                    reply.task.cancel()
            await asyncio.gather(
                *(reply.task for reply in replies.values() if reply.task is not None),
                return_exceptions=True,
            )


@dataclass(kw_only=True)
class LiveFake(StrictFake):
    """Full duplex: replies after quiet_s of input silence and yields yield_s after the
    caller speaks over it; output_speed is reply audio seconds per wall second.
    Between replies it streams one quiet frame per CHUNK_S of wall time, as the real
    API does."""

    path: str = LIVE_PATH
    reply_s: float = 1.0
    output_speed: float = 1.0
    quiet_s: float = 0.5
    yield_s: float = 0.2

    async def handler(self, websocket: ServerConnection) -> None:
        connection = self.accept(websocket)
        opened_s = time.perf_counter()
        snapshot: dict[str, JsonValue] | None = None
        sample_rate = 24000
        vad: EnergyVad | None = None
        reply_task: asyncio.Task[None] | None = None
        quiet_task: asyncio.Task[None] | None = None
        stop_at_s: float | None = None
        output_media_ms = 0.0
        sent_response_event = False

        def usage() -> dict[str, JsonValue]:
            return {"seconds": math.ceil(time.perf_counter() - opened_s)}

        async def speak() -> None:
            nonlocal output_media_ms, stop_at_s, sent_response_event
            pcm = tone_pcm(self.reply_s, sample_rate, frequency_hz=440.0)
            chunk_bytes = round(CHUNK_S * sample_rate) * 2
            started_s = time.perf_counter()
            for index, offset in enumerate(range(0, len(pcm), chunk_bytes)):
                await asyncio.sleep(
                    max(
                        0.0,
                        started_s
                        + index * CHUNK_S / self.output_speed
                        - time.perf_counter(),
                    )
                )
                if stop_at_s is not None and time.perf_counter() >= stop_at_s:
                    break
                chunk = pcm[offset : offset + chunk_bytes]
                chunk_ms = len(chunk) / 2 / sample_rate * 1000
                await connection.emit(
                    "session.output_audio.delta",
                    delta=base64.b64encode(chunk).decode("ascii"),
                )
                if index < len(REPLY_WORDS):
                    await connection.emit(
                        "session.output_transcript.delta",
                        delta=REPLY_WORDS[index] + " ",
                        start_ms=round(output_media_ms),
                        end_ms=round(output_media_ms + chunk_ms),
                    )
                output_media_ms += chunk_ms
            stop_at_s = None
            await connection.emit(
                "session.usage.updated",
                usage=usage(),
                context_window={"usage_ratio": 0.01},
            )
            if not sent_response_event:
                sent_response_event = True
                await connection.emit(
                    "response.event",
                    delegation_id=None,
                    event={
                        "type": "response.output_text.delta",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": "noted",
                    },
                )

        async def stream_quiet() -> None:
            frame = base64.b64encode(
                tone_pcm(CHUNK_S, sample_rate, 110.0, amplitude=QUIET_AMPLITUDE)
            ).decode("ascii")
            next_s = time.perf_counter()
            while True:
                if reply_task is None or reply_task.done():
                    await connection.emit("session.output_audio.delta", delta=frame)
                else:
                    pass
                next_s += CHUNK_S
                await asyncio.sleep(max(0.0, next_s - time.perf_counter()))

        async def reject(
            code: str, message: str, param: str | None, raw: dict[str, JsonValue]
        ) -> None:
            event_id = raw.get("event_id")
            await connection.emit(
                "error",
                **error_fields(
                    code,
                    message,
                    param,
                    event_id if isinstance(event_id, str) else None,
                    "client_event_id",
                ),
            )

        try:
            async for frame in websocket:
                raw = json.loads(frame)
                connection.received.append(raw)
                try:
                    event = LIVE_EVENTS.validate_python(raw)
                except ValidationError as exc:
                    await connection.emit(
                        "error", **validation_error_fields(exc, raw, "client_event_id")
                    )
                    continue
                if isinstance(event, SessionStart):
                    audio_format = (
                        event.session.audio.format
                        if event.session.audio is not None
                        else None
                    )
                    if snapshot is not None:
                        await reject(
                            "session_already_started",
                            "session.start was already sent",
                            None,
                            raw,
                        )
                    elif audio_format is not None and (
                        audio_format.type != "audio/pcm"
                        or audio_format.rate not in LIVE_SAMPLE_RATES
                    ):
                        await reject(
                            "invalid_value",
                            f"audio/pcm rate must be one of {LIVE_SAMPLE_RATES}",
                            "session.audio.format",
                            raw,
                        )
                    else:
                        sample_rate = audio_format.rate if audio_format else 24000
                        vad = EnergyVad(
                            sample_rate=sample_rate, silence_ms=self.quiet_s * 1000
                        )
                        # note (wenyao): shaped like a real session.started: client
                        # delegation unless asked otherwise, and the start's id.
                        snapshot = {
                            "id": f"live_{uuid.uuid4().hex[:20]}",
                            "expires_at": int(time.time()) + 3600,
                            "delegation": {"type": "client"},
                            **event.session.model_dump(mode="json", exclude_none=True),
                            "status": "active",
                            "input": [],
                        }
                        await connection.emit(
                            "session.started",
                            client_event_id=raw.get("event_id"),
                            session=snapshot,
                        )
                        quiet_task = asyncio.create_task(stream_quiet())
                elif snapshot is None or vad is None:
                    await reject(
                        "session_not_started", "send session.start first", None, raw
                    )
                elif isinstance(event, SessionInputAudioAppend):
                    pcm = base64.b64decode(event.audio, validate=True)
                    if len(pcm) % 2:
                        await reject(
                            "invalid_value",
                            "PCM16 needs an even byte count",
                            "audio",
                            raw,
                        )
                        continue
                    connection.input_pcm.extend(pcm)
                    for transition, media_ms in vad.feed(pcm):
                        if transition == "started":
                            if reply_task is not None and not reply_task.done():
                                stop_at_s = time.perf_counter() + self.yield_s
                        else:
                            await connection.emit(
                                "session.input_transcript.delta",
                                delta="hello ",
                                start_ms=0,
                                end_ms=round(media_ms),
                            )
                            if reply_task is None or reply_task.done():
                                reply_task = asyncio.create_task(speak())
                else:
                    for task in (reply_task, quiet_task):
                        if task is not None:
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                        else:
                            pass
                    await connection.emit(
                        "session.closed",
                        reason="close_requested",
                        usage=usage(),
                        session=snapshot,
                    )
                    await websocket.close(1000)
                    break
        finally:
            for task in (reply_task, quiet_task):
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                else:
                    pass
