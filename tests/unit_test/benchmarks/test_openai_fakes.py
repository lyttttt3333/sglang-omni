# SPDX-License-Identifier: Apache-2.0
"""The OpenAI fakes driven by a raw WebSocket client, so they are known-good alone."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
import time
import uuid
from collections.abc import AsyncIterator
from http import HTTPStatus

import numpy as np
import pytest
import websockets
from pydantic import JsonValue

from tests.unit_test.benchmarks.openai_fakes import (
    QUIET_AMPLITUDE,
    REALTIME_SAMPLE_RATE,
    REPLY_WORDS,
    SERVER_VAD_DEFAULTS,
    LiveFake,
    RealtimeFake,
    StrictFake,
    utterance_pcm,
)

KEY = "sk-test-fake-key"
PACKET_S = 0.08
REALTIME_MODEL_QUERY = "?model=gpt-realtime-2.1"
REALTIME_UPDATE = {
    "type": "realtime",
    "output_modalities": ["audio"],
    "audio": {
        "input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "turn_detection": {"type": "server_vad", "silence_duration_ms": 800},
        },
        "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "marin"},
    },
}
LIVE_START = {
    "model": "gpt-live-1",
    "instructions": "be brief",
    "audio": {
        "format": {"type": "audio/pcm", "rate": 16000},
        "output": {"voice": "marin"},
    },
    "delegation": {"type": "client"},
}


class Peer:
    """A raw client that stamps every server event with its arrival time."""

    def __init__(self, websocket: websockets.ClientConnection) -> None:
        self.websocket = websocket
        self.events: list[dict[str, JsonValue]] = []

    async def pump(self) -> None:
        async for frame in self.websocket:
            self.events.append({**json.loads(frame), "at_s": time.perf_counter()})

    async def send(self, **event: JsonValue) -> str:
        event_id = uuid.uuid4().hex
        await self.websocket.send(json.dumps({"event_id": event_id, **event}))
        return event_id

    async def wait_for(
        self, event_type: str, count: int = 1, timeout_s: float = 5.0
    ) -> list[dict[str, JsonValue]]:
        async with asyncio.timeout(timeout_s):
            while len(self.of(event_type)) < count:
                await asyncio.sleep(0.005)
        return self.of(event_type)

    def of(self, event_type: str) -> list[dict[str, JsonValue]]:
        return [event for event in self.events if event["type"] == event_type]

    async def stream(
        self, append_type: str, pcm: bytes, sample_rate: int, *, paced: bool
    ) -> list[float]:
        """Send PACKET_S packets and return each packet's send time."""
        packet_bytes = round(PACKET_S * sample_rate) * 2
        started_s = time.perf_counter()
        sent_s = []
        for index, offset in enumerate(range(0, len(pcm), packet_bytes)):
            if paced:
                await asyncio.sleep(
                    max(0.0, started_s + index * PACKET_S - time.perf_counter())
                )
            sent_s.append(time.perf_counter())
            await self.send(
                type=append_type,
                audio=base64.b64encode(pcm[offset : offset + packet_bytes]).decode(),
            )
        return sent_s


@contextlib.asynccontextmanager
async def peer_of(fake: StrictFake, query: str = "") -> AsyncIterator[Peer]:
    async with (
        fake.serve() as url,
        websockets.connect(
            url + query, additional_headers={"Authorization": f"Bearer {KEY}"}
        ) as websocket,
    ):
        peer = Peer(websocket)
        pump = asyncio.create_task(peer.pump())
        try:
            yield peer
        finally:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)


def audio_s(events: list[dict[str, JsonValue]], sample_rate: int) -> float:
    return sum(
        len(base64.b64decode(event["delta"])) / 2 / sample_rate for event in events
    )


def rms(event: dict[str, JsonValue]) -> float:
    samples = np.frombuffer(base64.b64decode(event["delta"]), "<i2").astype(float)
    return float(np.sqrt(np.mean(np.square(samples))))


def speech(deltas: list[dict[str, JsonValue]]) -> list[dict[str, JsonValue]]:
    return [event for event in deltas if rms(event) > QUIET_AMPLITUDE]


@pytest.mark.parametrize("fake_type", [RealtimeFake, LiveFake])
@pytest.mark.parametrize("key", [None, "sk-wrong"])
def test_handshake_without_the_key_is_refused_with_401(
    fake_type: type[StrictFake], key: str | None
) -> None:
    async def scenario() -> None:
        fake = fake_type(api_key=KEY)
        async with fake.serve() as url:
            with pytest.raises(websockets.InvalidStatus) as refused:
                async with websockets.connect(
                    url + REALTIME_MODEL_QUERY,
                    additional_headers=(
                        {"Authorization": f"Bearer {key}"} if key else None
                    ),
                ):
                    pass
        assert refused.value.response.status_code == HTTPStatus.UNAUTHORIZED
        assert fake.connections == []

    asyncio.run(scenario())


def test_realtime_requires_the_model_query() -> None:
    async def scenario() -> None:
        async with RealtimeFake(api_key=KEY).serve() as url:
            with pytest.raises(websockets.InvalidStatus) as refused:
                async with websockets.connect(
                    url, additional_headers={"Authorization": f"Bearer {KEY}"}
                ):
                    pass
        assert refused.value.response.status_code == HTTPStatus.BAD_REQUEST

    asyncio.run(scenario())


def test_realtime_rejects_unknown_fields_and_non_24k_audio() -> None:
    fake = RealtimeFake(api_key=KEY)

    async def scenario() -> tuple[str, str, list[dict[str, JsonValue]]]:
        async with peer_of(fake, REALTIME_MODEL_QUERY) as peer:
            await peer.wait_for("session.created")
            append_id = await peer.send(
                type="input_audio_buffer.append",
                audio=base64.b64encode(bytes(3840)).decode(),
                sglang={"seq": 0, "t_start_ms": 0.0},
            )
            update_id = await peer.send(
                type="session.update",
                session={
                    "type": "realtime",
                    "audio": {
                        "input": {"format": {"type": "audio/pcm", "rate": 16000}}
                    },
                },
            )
            return append_id, update_id, await peer.wait_for("error", count=2)

    append_id, update_id, (unknown, wrong_rate) = asyncio.run(scenario())

    assert unknown["error"]["code"] == "unknown_parameter"
    assert unknown["error"]["param"] == "sglang"
    assert unknown["error"]["event_id"] == append_id
    assert wrong_rate["error"]["param"] == "session.audio.input.format"
    assert wrong_rate["error"]["event_id"] == update_id
    assert fake.connections[0].input_pcm == b""
    assert not [e for e in fake.connections[0].sent if e["type"] == "session.updated"]


async def realtime_turns(
    fake: RealtimeFake, pcm: bytes, silence_duration_ms: int, listen_s: float
) -> list[dict[str, JsonValue]]:
    update = json.loads(json.dumps(REALTIME_UPDATE))
    update["audio"]["input"]["turn_detection"][
        "silence_duration_ms"
    ] = silence_duration_ms
    async with peer_of(fake, REALTIME_MODEL_QUERY) as peer:
        await peer.wait_for("session.created")
        await peer.send(type="session.update", session=update)
        await peer.wait_for("session.updated")
        await peer.stream(
            "input_audio_buffer.append", pcm, REALTIME_SAMPLE_RATE, paced=False
        )
        await asyncio.sleep(listen_s)
        return peer.events


def test_realtime_turns_follow_the_configured_silence_duration() -> None:
    # note (wenyao): one 600 ms pause splits the speech into two turns at 500 ms only.
    pcm = utterance_pcm(
        [("voice", 0.4), ("silence", 0.6), ("voice", 0.4), ("silence", 1.0)],
        REALTIME_SAMPLE_RATE,
    )

    def turns(silence_duration_ms: int) -> list[dict[str, JsonValue]]:
        return asyncio.run(
            realtime_turns(
                RealtimeFake(api_key=KEY, reply_s=0.3), pcm, silence_duration_ms, 1.5
            )
        )

    long_pause, short_pause = turns(800), turns(500)

    def stopped_ms(events: list[dict[str, JsonValue]]) -> list[int]:
        return [
            event["audio_end_ms"]
            for event in events
            if event["type"] == "input_audio_buffer.speech_stopped"
        ]

    (updated,) = [e for e in long_pause if e["type"] == "session.updated"]
    assert updated["session"]["audio"]["input"]["turn_detection"] == {
        **SERVER_VAD_DEFAULTS,
        "silence_duration_ms": 800,
    }
    assert stopped_ms(long_pause) == [1400]
    assert stopped_ms(short_pause) == [400, 1400]
    assert [e["type"] for e in long_pause].count("response.created") == 1
    assert [e["type"] for e in short_pause].count("response.created") == 2


def test_realtime_turn_emits_the_documented_sequence_faster_than_real_time() -> None:
    fake = RealtimeFake(api_key=KEY, reply_s=1.0, output_speed=5.0)
    pcm = utterance_pcm([("voice", 0.4), ("silence", 0.9)], REALTIME_SAMPLE_RATE)
    events = asyncio.run(realtime_turns(fake, pcm, 800, 1.0))
    types = [event["type"] for event in events]
    deltas = [e for e in events if e["type"] == "response.output_audio.delta"]
    (created,) = [e for e in events if e["type"] == "response.created"]
    (done,) = [e for e in events if e["type"] == "response.done"]
    order = [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "response.created",
        "response.output_audio.delta",
        "response.done",
    ]

    assert [types.index(t) for t in order] == sorted(types.index(t) for t in order)
    assert {
        "rate_limits.updated",
        "conversation.item.added",
        "response.output_item.added",
    } <= set(types)
    assert audio_s(deltas, 24000) == pytest.approx(1.0)
    assert {len(base64.b64decode(e["delta"])) for e in deltas} == {4800}
    assert {e["response_id"] for e in deltas} == {created["response"]["id"]}
    assert deltas[-1]["at_s"] - deltas[0]["at_s"] == pytest.approx(0.9 / 5, abs=0.06)
    assert "".join(
        e["delta"]
        for e in events
        if e["type"] == "response.output_audio_transcript.delta"
    ).split() == list(REPLY_WORDS)
    assert done["response"]["status"] == "completed"
    assert done["response"]["usage"]["output_token_details"]["audio_tokens"] > 0
    assert fake.connections[0].errors() == []


def test_realtime_barge_in_cancels_and_truncate_checks_the_item() -> None:
    fake = RealtimeFake(api_key=KEY, reply_s=5.0, output_speed=2.0)

    async def scenario() -> list[dict[str, JsonValue]]:
        async with peer_of(fake, REALTIME_MODEL_QUERY) as peer:
            await peer.wait_for("session.created")
            await peer.send(type="session.update", session=REALTIME_UPDATE)
            await peer.wait_for("session.updated")
            speech = utterance_pcm([("voice", 0.4), ("silence", 0.9)], 24000)
            await peer.stream("input_audio_buffer.append", speech, 24000, paced=False)
            await peer.wait_for("response.output_audio.delta", count=3)
            await peer.stream(
                "input_audio_buffer.append",
                utterance_pcm([("voice", 0.2)], 24000),
                24000,
                paced=False,
            )
            await peer.wait_for("response.done")
            deltas = peer.of("response.output_audio.delta")
            item_id = deltas[0]["item_id"]
            delivered_ms = round(audio_s(deltas, 24000) * 1000)
            for target, end_ms in (
                (item_id, 50),
                ("item_unknown", 50),
                (item_id, delivered_ms + 500),
            ):
                await peer.send(
                    type="conversation.item.truncate",
                    item_id=target,
                    content_index=0,
                    audio_end_ms=end_ms,
                )
            await peer.wait_for("error", count=2)
            await peer.wait_for("conversation.item.truncated")
            return peer.events

    events = asyncio.run(scenario())
    (done,) = [e for e in events if e["type"] == "response.done"]
    truncated, unknown, too_long = [
        e for e in events if e["type"] in ("conversation.item.truncated", "error")
    ]
    speech_started = [
        e for e in events if e["type"] == "input_audio_buffer.speech_started"
    ]

    assert done["response"]["status"] == "cancelled"
    assert done["response"]["status_details"] == {
        "type": "cancelled",
        "reason": "turn_detected",
    }
    assert events.index(speech_started[1]) < events.index(done)
    assert (
        audio_s(
            [e for e in events if e["type"] == "response.output_audio.delta"], 24000
        )
        < 5.0
    )
    assert truncated["audio_end_ms"] == 50
    assert unknown["error"]["code"] == "item_not_found"
    assert too_long["error"]["param"] == "audio_end_ms"


def test_live_rejects_audio_before_start_unknown_fields_and_bad_rates() -> None:
    async def scenario() -> tuple[str, list[dict[str, JsonValue]]]:
        async with peer_of(LiveFake(api_key=KEY)) as peer:
            early_id = await peer.send(type="session.input_audio.append", audio="AAAA")
            await peer.send(
                type="session.start",
                session={
                    "model": "gpt-live-1",
                    "audio": {"format": {"type": "audio/pcm", "rate": 22050}},
                },
            )
            await peer.send(
                type="session.start", session={"model": "gpt-live-1", "sglang": {}}
            )
            return early_id, await peer.wait_for("error", count=3)

    early_id, (early, bad_rate, unknown) = asyncio.run(scenario())

    assert early["error"]["code"] == "session_not_started"
    assert early["error"]["client_event_id"] == early_id
    assert bad_rate["error"]["param"] == "session.audio.format"
    assert unknown["error"]["code"] == "unknown_parameter"
    assert unknown["error"]["param"] == "session.sglang"


async def live_session(
    fake: LiveFake, pcm: bytes, listen_s: float
) -> tuple[list[dict[str, JsonValue]], list[float]]:
    async with peer_of(fake) as peer:
        await peer.send(type="session.start", session=LIVE_START)
        await peer.wait_for("session.started")
        sent_s = await peer.stream("session.input_audio.append", pcm, 16000, paced=True)
        await asyncio.sleep(listen_s)
        await peer.send(type="session.close")
        await peer.wait_for("session.closed")
        await asyncio.wait_for(peer.websocket.wait_closed(), 1.0)
        assert peer.websocket.close_code == 1000
        return peer.events, sent_s


@pytest.mark.parametrize("output_speed", [1.0, 4.0])
def test_live_replies_after_quiet_paced_by_output_speed(output_speed: float) -> None:
    fake = LiveFake(api_key=KEY, reply_s=1.5, output_speed=output_speed)
    pcm = utterance_pcm([("voice", 0.4), ("silence", 0.8)], 16000)
    events, _ = asyncio.run(live_session(fake, pcm, 2.0))
    deltas = speech([e for e in events if e["type"] == "session.output_audio.delta"])

    assert events[0]["session"]["audio"]["format"] == {
        "type": "audio/pcm",
        "rate": 16000,
    }
    assert audio_s(deltas, 16000) == pytest.approx(1.5)
    assert {len(base64.b64decode(e["delta"])) for e in deltas} == {3200}
    assert deltas[-1]["at_s"] - deltas[0]["at_s"] == pytest.approx(
        1.4 / output_speed, abs=0.08
    )
    assert {
        "session.output_transcript.delta",
        "session.usage.updated",
        "response.event",
    } <= {event["type"] for event in events}
    assert events[-1]["type"] == "session.closed"
    assert events[-1]["reason"] == "close_requested"
    assert events[-1]["usage"]["seconds"] >= 1
    assert fake.connections[0].errors() == []
    assert bytes(fake.connections[0].input_pcm) == pcm


def test_live_stops_talking_soon_after_the_caller_speaks_over_it() -> None:
    fake = LiveFake(api_key=KEY, reply_s=4.0, yield_s=0.2)
    pcm = utterance_pcm(
        [("voice", 0.4), ("silence", 1.3), ("voice", 0.4), ("silence", 0.4)], 16000
    )
    events, sent_s = asyncio.run(live_session(fake, pcm, 1.5))
    deltas = speech([e for e in events if e["type"] == "session.output_audio.delta"])
    barge_in_s = sent_s[round(1.7 / PACKET_S)]

    assert deltas[0]["at_s"] < barge_in_s
    assert 0.1 <= deltas[-1]["at_s"] - barge_in_s <= 0.2 + 0.15
    assert audio_s(deltas, 16000) < 2.0


def test_live_streams_quiet_frames_at_real_time_between_replies() -> None:
    fake = LiveFake(api_key=KEY, reply_s=0.5, output_speed=4.0)
    pcm = utterance_pcm([("voice", 0.4), ("silence", 1.2)], 16000)
    events, _ = asyncio.run(live_session(fake, pcm, 1.0))
    deltas = [e for e in events if e["type"] == "session.output_audio.delta"]
    quiet = [e for e in deltas if rms(e) <= QUIET_AMPLITUDE]
    started_s = events[0]["at_s"]
    closed_s = events[-1]["at_s"]

    assert events[0]["type"] == "session.started"
    assert events[0]["session"]["delegation"] == {"type": "client"}
    assert audio_s(speech(deltas), 16000) == pytest.approx(0.5)
    assert audio_s(quiet, 16000) == pytest.approx(
        closed_s - started_s - 0.5 / 4.0, abs=0.25
    )
    assert max(b["at_s"] - a["at_s"] for a, b in zip(deltas, deltas[1:])) < 0.2
    assert [rms(e) for e in quiet] == pytest.approx(
        [QUIET_AMPLITUDE / math.sqrt(2)] * len(quiet), rel=0.05
    )
