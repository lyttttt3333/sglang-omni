# SPDX-License-Identifier: Apache-2.0
"""OpenAI Realtime and Live sessions recorded by the duplex client against the fakes,
then replayed by the oracle, reconstructed and scored like any other arm."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile
from pydantic import JsonValue
from scipy.signal import resample_poly

from benchmarks.duplex import serving, v10_dataset
from benchmarks.duplex.client import (
    OPENAI_REALTIME_PLAYBACK_REPORT,
    TurnDetection,
    run_session,
)
from benchmarks.duplex.openai_client import LIVE_QUIET_S, LIVE_SPEECH_RMS, pcm_rms
from benchmarks.duplex.oracle import evaluate_trace
from benchmarks.duplex.reference_capture import TraceFormat
from benchmarks.duplex.reference_export import export_runs
from benchmarks.duplex.reference_v10 import export as export_v10
from benchmarks.duplex.v15_audio import OPENAI_REALTIME_BARGE_IN_CUT, reconstruct_output
from benchmarks.duplex.v15_runner import run_samples
from tests.unit_test.benchmarks.openai_fakes import (
    LiveFake,
    RealtimeFake,
    StrictFake,
    utterance_pcm,
)
from tests.unit_test.fixtures.duplex_fakes import (
    SERVER,
    write_dataset,
    write_v10_dataset,
)

SENTINEL_KEY = "sk-sentinel-must-not-leak-4f1c"
INSTRUCTIONS = "Answer in one short sentence."
PACKET_BYTES = 2560
TAIL_SILENCE_S = 0.48
SERVER_VAD = TurnDetection(type="server_vad", silence_duration_ms=800)
# note (wenyao): every segment is a whole number of 80 ms packets, so packet k of the
# stream is exactly bytes [k * 2560, (k + 1) * 2560).
CLEAN_TURN = [("voice", 0.48), ("silence", 1.12)]
BARGE_IN = [("voice", 0.48), ("silence", 1.2), ("voice", 0.48), ("silence", 0.96)]
# note (wenyao): the reply ends inside the input, so Realtime stays silent through
# the 10 s tail and only the transport pongs show its receiver alive after it.
REPLY_IN_WINDOW = [("voice", 0.48), ("silence", 2.88)]


@pytest.fixture
def provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """OpenAI settings in the environment; returns the run directory to search."""
    instructions = tmp_path / "instructions.txt"
    instructions.write_text(INSTRUCTIONS, encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL_KEY)
    monkeypatch.setenv("OPENAI_DUPLEX_INSTRUCTIONS_FILE", str(instructions))
    for name in ("OPENAI_DUPLEX_MODEL", "OPENAI_DUPLEX_VOICE"):
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return run_dir


def record(
    fake: StrictFake,
    variant_dir: Path,
    pcm: bytes,
    *,
    profile: str,
    turn_detection: TurnDetection | None,
    report_playback: bool = False,
) -> list[dict[str, JsonValue]]:
    async def scenario() -> None:
        async with fake.serve() as url:
            await run_session(
                url,
                pcm,
                trace_path=variant_dir / "continuous.jsonl",
                timeout_s=15.0,
                profile=profile,
                turn_detection=turn_detection,
                report_playback=report_playback,
                tail_silence_s=TAIL_SILENCE_S,
                end_wait_s=8.0,
            )

    variant_dir.mkdir()
    (variant_dir / "input.pcm").write_bytes(pcm)
    asyncio.run(scenario())
    return read_jsonl(variant_dir / "continuous.jsonl")


def read_jsonl(path: Path) -> list[dict[str, JsonValue]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def sent(
    records: list[dict[str, JsonValue]], event_type: str
) -> list[dict[str, JsonValue]]:
    return [
        r["event"]
        for r in records
        if r["direction"] == "send" and r["event"]["type"] == event_type
    ]


def received(
    records: list[dict[str, JsonValue]], event_type: str
) -> list[dict[str, JsonValue]]:
    return [
        r
        for r in records
        if r["direction"] == "receive" and r["event"]["type"] == event_type
    ]


def assert_canonical_appends(
    records: list[dict[str, JsonValue]], stream: bytes
) -> None:
    appends = sent(records, "input_audio_buffer.append")
    assert [base64.b64decode(a["audio"]) for a in appends] == [
        stream[offset : offset + PACKET_BYTES]
        for offset in range(0, len(stream), PACKET_BYTES)
    ]
    assert [a["sglang"]["seq"] for a in appends] == list(range(len(appends)))
    assert [a["sglang"]["t_start_ms"] for a in appends] == pytest.approx(
        [index * 80.0 for index in range(len(appends))]
    )


def assert_no_key_on_disk(run_dir: Path) -> None:
    leaked = [
        path
        for path in run_dir.rglob("*")
        if path.is_file() and SENTINEL_KEY.encode() in path.read_bytes()
    ]
    assert leaked == []


def assert_rtt_samples(variant_dir: Path) -> None:
    rtt = json.loads((variant_dir / "rtt.json").read_text())
    assert rtt["samples_s"] and all(0 <= sample < 1 for sample in rtt["samples_s"])
    assert rtt["p50_s"] == pytest.approx(float(np.percentile(rtt["samples_s"], 50)))
    assert rtt["p90_s"] == pytest.approx(float(np.percentile(rtt["samples_s"], 90)))


def test_realtime_clean_session_replays_reconstructs_and_scores(provider: Path) -> None:
    pcm = utterance_pcm(CLEAN_TURN, 16000)
    stream = pcm + bytes(round(TAIL_SILENCE_S * 16000) * 2)
    fake = RealtimeFake(api_key=SENTINEL_KEY, reply_s=1.0, output_speed=5.0)
    variant_dir = provider / "realtime"
    records = record(
        fake, variant_dir, pcm, profile="openai-realtime", turn_detection=SERVER_VAD
    )
    (connection,) = fake.connections
    (update,) = [e for e in connection.received if e["type"] == "session.update"]
    wire = read_jsonl(variant_dir / "wire.jsonl")
    wire_appends = sent(wire, "input_audio_buffer.append")
    wire_pcm = b"".join(base64.b64decode(a["audio"]) for a in wire_appends)
    expected_24k = np.clip(
        resample_poly(np.frombuffer(stream, dtype="<i2"), 3, 2), -32768, 32767
    )

    assert evaluate_trace(records, profile="openai-realtime")["status"] == "pass"
    assert connection.authorization == f"Bearer {SENTINEL_KEY}"
    assert "model=gpt-realtime-2.1" in connection.path
    assert connection.errors() == []
    assert update["session"]["audio"]["input"][
        "turn_detection"
    ] == SERVER_VAD.model_dump(mode="json")
    assert update["session"]["audio"]["output"]["voice"] == "marin"
    assert update["session"]["instructions"] == INSTRUCTIONS
    assert_canonical_appends(records, stream)
    assert all("sglang" not in a for a in wire_appends)
    assert len(wire_appends) == len(sent(records, "input_audio_buffer.append"))
    assert {len(base64.b64decode(a["audio"])) for a in wire_appends[:-1]} == {3840}
    assert wire_pcm == bytes(connection.input_pcm)
    assert len(wire_pcm) // 2 == len(expected_24k)
    assert np.abs(np.frombuffer(wire_pcm, dtype="<i2") - expected_24k).max() <= 1

    playout = reconstruct_output(variant_dir, profile="openai-realtime")
    assert playout["errors"] == []
    assert playout["media_samples"] == 24000
    assert playout["barge_in"]["cuts"] == 0

    (reply,) = serving.score_session(variant_dir, "openai-realtime", 0.0)["replies"]
    assert fake.first_audio_delay_s <= reply["first_audio_latency_s"] < 0.5
    assert reply["audio_s"] == pytest.approx(1.0)
    assert reply["real_time_factor"] == pytest.approx(1 / 5.0, abs=0.08)

    assert_rtt_samples(variant_dir)
    assert_no_key_on_disk(provider)


def test_realtime_barge_in_cuts_playout_and_truncates_what_played(
    provider: Path,
) -> None:
    pcm = utterance_pcm(BARGE_IN, 16000)
    fake = RealtimeFake(api_key=SENTINEL_KEY, reply_s=6.0, output_speed=5.0)
    variant_dir = provider / "realtime"
    records = record(
        fake,
        variant_dir,
        pcm,
        profile="openai-realtime",
        turn_detection=SERVER_VAD,
        report_playback=True,
    )
    cancelled = [
        r
        for r in received(records, "response.done")
        if r["event"]["response"]["status"] == "cancelled"
    ]
    (cancel,) = cancelled
    cancelled_id = cancel["event"]["response"]["id"]
    barge_in = received(records, "input_audio_buffer.speech_started")[1]
    (truncate,) = sent(records, "conversation.item.truncate")
    origin_s = next(
        r["time_s"]
        for r in records
        if r["direction"] == "send"
        and r["event"]["type"] == "input_audio_buffer.append"
    )

    assert evaluate_trace(records, profile="openai-realtime")["status"] == "pass"
    assert [
        r["event"]["response"]["status"] for r in received(records, "response.done")
    ] == ["cancelled", "completed"]
    assert fake.connections[0].errors() == []
    assert [e["type"] for e in fake.connections[0].sent].count(
        "conversation.item.truncated"
    ) == 1

    reconstruct_output(variant_dir, profile="openai-realtime")
    playout = json.loads((variant_dir / "playout.json").read_text())
    (cut,) = playout["cuts"]
    assert cut["type"] == "input_audio_buffer.speech_started"
    assert cut["receipt_s"] == pytest.approx(barge_in["time_s"] - origin_s)
    assert cut["dropped_samples"] > 0
    assert barge_in["time_s"] < cancel["time_s"]

    cancelled_chunks = [
        c for c in playout["chunks"] if c["response_id"] == cancelled_id
    ]
    played_samples = sum(
        min(chunk["samples"], max(0, cut["cut_sample"] - chunk["playout_start_sample"]))
        for chunk in cancelled_chunks
    )
    delivered_ms = sum(chunk["samples"] for chunk in cancelled_chunks) / 24
    assert (
        truncate["item_id"]
        == received(records, "response.output_audio.delta")[0]["event"]["item_id"]
    )
    assert truncate["content_index"] == 0
    assert truncate["audio_end_ms"] == pytest.approx(played_samples / 24, abs=2)
    assert 0 < truncate["audio_end_ms"] < delivered_ms


@pytest.mark.parametrize(
    ("reply_s", "interrupt_response", "statuses"),
    [
        (6.0, None, ["completed", "cancelled", "completed"]),
        (0.2, None, ["completed", "completed"]),
        (6.0, False, ["completed", "completed"]),
    ],
    ids=["audio-queued", "audio-played-out", "interrupt-response-false"],
)
def test_realtime_barge_in_after_a_completed_reply_cuts_only_queued_audio(
    provider: Path,
    reply_s: float,
    interrupt_response: bool | None,
    statuses: list[str],
) -> None:
    turn_detection = TurnDetection(
        type="server_vad",
        silence_duration_ms=800,
        **({} if interrupt_response is None else {"interrupt_response": False}),
    )
    fake = RealtimeFake(api_key=SENTINEL_KEY, reply_s=reply_s, output_speed=30.0)
    variant_dir = provider / "realtime"
    records = record(
        fake,
        variant_dir,
        utterance_pcm(BARGE_IN, 16000),
        profile="openai-realtime",
        turn_detection=turn_detection,
        report_playback=True,
    )
    dones = received(records, "response.done")
    barge_in = received(records, "input_audio_buffer.speech_started")[1]
    first_deltas = [
        r
        for r in received(records, "response.output_audio.delta")
        if r["event"]["response_id"] == dones[0]["event"]["response"]["id"]
    ]
    truncates = [
        r
        for r in records
        if r["direction"] == "send"
        and r["event"]["type"] == "conversation.item.truncate"
    ]
    origin_s = next(
        r["time_s"]
        for r in records
        if r["direction"] == "send"
        and r["event"]["type"] == "input_audio_buffer.append"
    )
    report = evaluate_trace(records, profile="openai-realtime")
    playout = reconstruct_output(variant_dir, profile="openai-realtime")
    cuts = json.loads((variant_dir / "playout.json").read_text())["cuts"]
    usage_by_response = {
        r["event"]["response"]["id"]: r["event"]["response"]["usage"] for r in dones
    }

    assert [r["event"]["response"]["status"] for r in dones] == statuses
    assert dones[0]["time_s"] < barge_in["time_s"]
    assert fake.connections[0].errors() == []
    assert report["status"] == "pass", report["violations"]
    assert report["metrics"]["usage"]["total_tokens"] == sum(
        usage["total_tokens"] for usage in usage_by_response.values()
    )
    queued = "cancelled" in statuses
    assert playout["barge_in"]["cuts"] == len(cuts) == int(queued)
    if queued:
        completed, cancelled = dones[0], dones[1]
        (truncate,) = truncates
        (cut,) = cuts
        played_ms = (barge_in["time_s"] - first_deltas[0]["time_s"]) * 1000
        assert cancelled["event"]["response"]["id"] == (
            completed["event"]["response"]["id"]
        )
        assert cancelled["event"]["response"]["usage"] == (
            completed["event"]["response"]["usage"]
        )
        assert barge_in["time_s"] <= truncate["time_s"] <= cancelled["time_s"]
        assert truncate["event"]["item_id"] == first_deltas[0]["event"]["item_id"]
        assert truncate["event"]["audio_end_ms"] == pytest.approx(played_ms, abs=2)
        assert 0 < truncate["event"]["audio_end_ms"] < reply_s * 1000
        assert cut["type"] == "input_audio_buffer.speech_started"
        assert cut["receipt_s"] == pytest.approx(barge_in["time_s"] - origin_s)
        assert cut["dropped_samples"] > 0
        assert [e["type"] for e in fake.connections[0].sent].count(
            "conversation.item.truncated"
        ) == 1
    else:
        assert truncates == []


@pytest.mark.parametrize(
    ("reply_s", "output_speed"),
    [(1.0, 1.0), (1.0, 5.0), (3.0, 1.0)],
    ids=["real-time", "fast", "speech-past-the-input-end"],
)
def test_live_session_replays_reconstructs_and_refuses_serving(
    provider: Path, reply_s: float, output_speed: float
) -> None:
    pcm = utterance_pcm(CLEAN_TURN, 16000)
    stream = pcm + bytes(round(TAIL_SILENCE_S * 16000) * 2)
    fake = LiveFake(api_key=SENTINEL_KEY, reply_s=reply_s, output_speed=output_speed)
    variant_dir = provider / "live"
    records = record(fake, variant_dir, pcm, profile="openai-live", turn_detection=None)
    (connection,) = fake.connections
    (start,) = [e for e in connection.received if e["type"] == "session.start"]
    wire_appends = sent(
        read_jsonl(variant_dir / "wire.jsonl"), "session.input_audio.append"
    )
    deltas = received(records, "response.output_audio.delta")
    fake_deltas = [
        e for e in connection.sent if e["type"] == "session.output_audio.delta"
    ]

    assert evaluate_trace(records, profile="openai-live")["status"] == "pass"
    assert connection.authorization == f"Bearer {SENTINEL_KEY}"
    assert connection.errors() == []
    assert start["session"]["model"] == "gpt-live-1"
    assert start["session"]["delegation"] == {"type": "client"}
    assert start["session"]["audio"] == {
        "format": {"type": "audio/pcm", "rate": 16000},
        "output": {"voice": "marin"},
    }
    assert start["session"]["instructions"] == INSTRUCTIONS
    assert_canonical_appends(records, stream)
    assert [base64.b64decode(a["audio"]) for a in wire_appends] == [
        base64.b64decode(a["audio"]) for a in sent(records, "input_audio_buffer.append")
    ]
    assert all("sglang" not in a for a in wire_appends)
    assert bytes(connection.input_pcm) == stream

    assert [d["event"]["delta"] for d in deltas] == [e["delta"] for e in fake_deltas]
    assert {d["event"]["wire_type"] for d in deltas} == {"session.output_audio.delta"}
    assert {d["event"]["response_id"] for d in deltas} == {None}
    for verbatim in ("session.usage.updated", "response.event", "session.closed"):
        assert [r["event"] for r in received(records, verbatim)] == [
            e for e in connection.sent if e["type"] == verbatim
        ]
    speech = [d for d in deltas if pcm_rms(d["event"]["delta"]) > LIVE_SPEECH_RMS]
    assert len(speech) * 0.1 == pytest.approx(reply_s)
    assert speech[-1]["time_s"] - speech[0]["time_s"] == pytest.approx(
        (reply_s - 0.1) / output_speed, abs=0.08
    )
    # note (wenyao): quiet frames never pause for LIVE_QUIET_S, so the session ends
    # LIVE_QUIET_S after the last speech, or at the input end if that is later.
    assert max(b["time_s"] - a["time_s"] for a, b in zip(deltas, deltas[1:])) < 0.2
    input_end_s = next(
        r["time_s"]
        for r in records
        if r["direction"] == "send"
        and r["event"]["type"] == "input_audio_buffer.append"
    ) + len(stream) / (2 * 16000)
    (close,) = [
        r
        for r in records
        if r["direction"] == "send" and r["event"]["type"] == "session.close"
    ]
    assert close["time_s"] == pytest.approx(
        max(input_end_s, speech[-1]["time_s"] + LIVE_QUIET_S), abs=0.15
    )

    playout = reconstruct_output(variant_dir, profile="openai-live")
    assert playout["errors"] == []
    assert playout["sample_rate"] == 16000
    assert playout["media_samples"] == 1600 * len(deltas)

    with pytest.raises(ValueError):
        serving.score_session(variant_dir, "openai-live", 0.0)
    assert_rtt_samples(variant_dir)
    assert_no_key_on_disk(provider)


def write_speech_sample(dataset: Path) -> None:
    write_dataset(dataset)
    speech = np.frombuffer(utterance_pcm(CLEAN_TURN, 16000), dtype="<i2")
    for filename in ("input.wav", "clean_input.wav"):
        soundfile.write(
            str(dataset / "user_interruption" / "1" / filename),
            speech,
            16000,
            subtype="PCM_16",
        )


@pytest.mark.parametrize(
    ("profile", "fake_type", "turn_detection"),
    [
        ("openai-realtime", RealtimeFake, SERVER_VAD),
        ("openai-live", LiveFake, None),
    ],
)
def test_run_samples_records_an_openai_wire_end_to_end(
    provider: Path,
    profile: str,
    fake_type: type[StrictFake],
    turn_detection: TurnDetection | None,
) -> None:
    dataset, output = provider.parent / "data", provider / "samples-run"
    write_speech_sample(dataset)
    fake = fake_type(api_key=SENTINEL_KEY, reply_s=1.0)

    async def scenario() -> dict[str, JsonValue]:
        async with fake.serve() as url:
            return await run_samples(
                dataset,
                url=url,
                output=output,
                server=SERVER,
                dataset_revision="fixture",
                timeout_s=20.0,
                sample_ids=["user_interruption/1"],
                profile=profile,
                turn_detection=turn_detection,
                report_playback=turn_detection is not None,
                tail_silence_s=TAIL_SILENCE_S,
                end_wait_s=8.0,
            )

    result = asyncio.run(scenario())
    (sample,) = result["samples"]
    manifest_text = (output / "manifest.json").read_text()

    assert result["status"] == "complete"
    assert len(fake.connections) == 2
    for variant in sample["variants"].values():
        variant_dir = output / variant["directory"]
        (case,) = json.loads((variant_dir / "report.json").read_text())["cases"]
        assert variant["status"] == "pass" and variant["qualified"] is True, variant
        assert case["status"] == "pass"
        assert {"continuous.jsonl", "wire.jsonl", "rtt.json"} <= set(variant["files"])
        deltas = received(
            read_jsonl(variant_dir / "continuous.jsonl"), "response.output_audio.delta"
        )
        speech = [d for d in deltas if pcm_rms(d["event"]["delta"]) > LIVE_SPEECH_RMS]
        assert len(speech) * 0.1 == pytest.approx(1.0)
        assert variant["output"]["media_duration_s"] == pytest.approx(0.1 * len(deltas))
    assert hashlib.sha256(INSTRUCTIONS.encode()).hexdigest() in manifest_text
    if profile == "openai-realtime":
        turn_based = json.loads(manifest_text)["config"]["turn_based"]
        assert turn_based["barge_in_cut"] == OPENAI_REALTIME_BARGE_IN_CUT
        assert turn_based["playback_report"] == OPENAI_REALTIME_PLAYBACK_REPORT
    else:
        pass
    assert ("gpt-realtime-2.1" if profile == "openai-realtime" else "gpt-live-1") in (
        manifest_text
    )
    assert_no_key_on_disk(provider)


@pytest.mark.parametrize(
    ("profile", "fake_type", "turn_detection", "trace_format"),
    [
        (
            "openai-realtime",
            RealtimeFake,
            SERVER_VAD,
            TraceFormat.TURN_BASED_PCM16,
        ),
        ("openai-live", LiveFake, None, TraceFormat.PCM16),
    ],
)
def test_reference_exports_take_sessions_with_a_silent_tail(
    provider: Path,
    profile: str,
    fake_type: type[StrictFake],
    turn_detection: TurnDetection | None,
    trace_format: TraceFormat,
) -> None:
    v15_data, v10_data = provider.parent / "v15-data", provider.parent / "v10-data"
    write_dataset(v15_data)
    write_v10_dataset(v10_data)
    speech = np.frombuffer(utterance_pcm(REPLY_IN_WINDOW, 16000), dtype="<i2")
    for source in (
        v15_data / "user_interruption/1/input.wav",
        v15_data / "user_interruption/1/clean_input.wav",
        v10_data / "synthetic_user_interruption/1/input.wav",
    ):
        soundfile.write(str(source), speech, 16000, subtype="PCM_16")
    fake = fake_type(api_key=SENTINEL_KEY, reply_s=1.0)
    settings: dict = dict(
        server=SERVER,
        dataset_revision="fixture",
        timeout_s=30.0,
        profile=profile,
        concurrency=2,
        turn_detection=turn_detection,
        report_playback=turn_detection is not None,
        tail_silence_s=10.0,
        end_wait_s=8.0,
    )

    async def scenario() -> None:
        async with fake.serve() as url:
            await asyncio.gather(
                run_samples(
                    v15_data,
                    url=url,
                    output=provider / "v15",
                    sample_ids=["user_interruption/1"],
                    **settings,
                ),
                run_samples(
                    v10_data,
                    url=url,
                    output=provider / "v10",
                    sample_ids=["synthetic_user_interruption/1"],
                    dataset=v10_dataset,
                    variants={"input": "input"},
                    kind=v10_dataset.RUN_KIND,
                    **settings,
                ),
            )

    asyncio.run(scenario())
    v15 = export_runs(
        [provider / "v15"],
        provider.parent / "v15-export",
        "fake",
        dataset_root=v15_data,
        sample_ids=["user_interruption/1"],
        trace_format=trace_format.value,
    )
    v10 = export_v10(
        [provider / "v10"], v10_data, provider.parent / "v10-export", trace_format.value
    )

    ((v15_sample,), (v10_sample,)) = v15["samples"], v10["samples"]
    assert {
        name: variant["reasons"] for name, variant in v15_sample["variants"].items()
    } == {"overlap": [], "clean": []}
    assert v10_sample["reasons"] == []
    for output in (
        provider.parent / "v15-export/user_interruption/1/output.wav",
        provider.parent / "v15-export/user_interruption/1/clean_output.wav",
        provider.parent / "v10-export/synthetic_user_interruption/1/output.wav",
    ):
        audio, rate = soundfile.read(str(output), dtype="int16")
        assert rate == 16000 and len(audio) == len(speech)
        assert np.abs(audio).max() > 1000
