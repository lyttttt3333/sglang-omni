# SPDX-License-Identifier: Apache-2.0
"""Prepares a CI data root: pinned model checkpoints, the frozen corpus and the
voices the speech models clone.

Every file is checked against its recorded SHA-256, so a runner's cache is
reused only when it matches what the golden files were made from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from math import gcd
from pathlib import Path

import soundfile
from scipy.signal import resample_poly

CI_DIRECTORY = Path(__file__).resolve().parent
CORPUS_SOURCES = {
    "librispeech-test-clean.tar.gz": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "fleurs-cmn-test.tar.gz": "https://huggingface.co/datasets/google/fleurs/resolve/main/data/cmn_hans_cn/audio/test.tar.gz",
    "fleurs-cmn-test.tsv": "https://huggingface.co/datasets/google/fleurs/resolve/main/data/cmn_hans_cn/test.tsv",
    "ascend-test.parquet": "https://huggingface.co/datasets/CAiRE/ASCEND/resolve/main/main/test-00000-of-00001.parquet",
}
DOWNLOAD_CHUNK_BYTES = 1 << 20
# The voice Drama 3's goldens and API tests clone: the Common Voice clip (CC0)
# the CUDA cookbook clones, resampled to the Fish S1 DAC codec's 44.1 kHz.
VOICE_SOURCE = "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/0df0b57b45253d5a9fa0b0e98846642f235f828e/en/prompt-wavs/common_voice_en_10119832.wav"
VOICE_SOURCE_SHA256 = "af3ac928e98cdb171c9abdc611813f2cc40c52b71167790b785628104f81f389"
VOICE_SHA256 = "df436ec78ec5b99cb16d4ca91057148bb490ac58715da6e49ce22ffa070f643b"
VOICE_SAMPLE_RATE = 44100
VOICE_DIRECTORY = "voices"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_hashes(path: Path) -> dict[str, str]:
    hashes = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            hashes[name.strip()] = digest
        else:
            pass
    return hashes


def download(url: str, destination: Path, expected_sha256: str) -> None:
    """Downloads url to destination unless a file with the expected hash is there."""
    if destination.exists() and sha256(destination) == expected_sha256:
        return
    else:
        pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    print(f"downloading {url}", flush=True)
    with (
        urllib.request.urlopen(url, timeout=600) as response,
        partial.open("wb") as out,
    ):
        shutil.copyfileobj(response, out, DOWNLOAD_CHUNK_BYTES)
    actual = sha256(partial)
    if actual != expected_sha256:
        partial.unlink()
        raise SystemExit(
            f"{url}: SHA-256 {actual} does not match the pinned {expected_sha256}"
        )
    else:
        pass
    partial.rename(destination)


def provision_model(data_root: Path, repo: str, pin: dict) -> Path:
    model_directory = data_root / "models" / repo.replace("/", "_")
    for name, digest in pin["files"].items():
        url = f"https://huggingface.co/{repo}/resolve/{pin['revision']}/{name}"
        download(url, model_directory / name, digest)
    return model_directory


def corpus_is_complete(corpus_directory: Path, hashes: dict[str, str]) -> bool:
    return all(
        (corpus_directory / name).exists() and sha256(corpus_directory / name) == digest
        for name, digest in hashes.items()
    )


def provision_corpus(data_root: Path) -> Path:
    corpus_root = data_root / "corpus"
    corpus_directory = corpus_root / "v1"
    clip_hashes = read_hashes(CI_DIRECTORY / "corpus" / "v1.sha256")
    if corpus_is_complete(corpus_directory, clip_hashes):
        return corpus_directory
    else:
        pass
    raw_hashes = read_hashes(CI_DIRECTORY / "corpus" / "raw.sha256")
    for name, url in CORPUS_SOURCES.items():
        download(url, corpus_root / "raw" / name, raw_hashes[name])
    extracted = corpus_root / "extracted"
    if not (extracted / "LibriSpeech" / "test-clean").exists():
        with tarfile.open(
            corpus_root / "raw" / "librispeech-test-clean.tar.gz"
        ) as archive:
            archive.extractall(extracted, filter="data")
    else:
        pass
    if not (extracted / "test").exists():
        with tarfile.open(corpus_root / "raw" / "fleurs-cmn-test.tar.gz") as archive:
            archive.extractall(extracted, filter="data")
    else:
        pass
    subprocess.run(
        [
            sys.executable,
            str(CI_DIRECTORY / "corpus" / "build_corpus.py"),
            str(data_root),
        ],
        check=True,
    )
    if not corpus_is_complete(corpus_directory, clip_hashes):
        raise SystemExit("the rebuilt corpus does not match corpus/v1.sha256")
    else:
        pass
    # Note (Jiaxin Deng): later runs read only the clips; the archives and their
    # extracted audio (about 2 GB) are only needed to rebuild them.
    shutil.rmtree(corpus_root / "raw", ignore_errors=True)
    shutil.rmtree(extracted, ignore_errors=True)
    return corpus_directory


def provision_voice(data_root: Path) -> Path:
    voice = data_root / VOICE_DIRECTORY / "fishaudio_drama3_voice.wav"
    if voice.exists() and sha256(voice) == VOICE_SHA256:
        return voice
    else:
        pass
    source = data_root / VOICE_DIRECTORY / "common_voice_en_10119832.wav"
    download(VOICE_SOURCE, source, VOICE_SOURCE_SHA256)
    samples, source_rate = soundfile.read(source, dtype="float64")
    step = gcd(VOICE_SAMPLE_RATE, source_rate)
    soundfile.write(
        voice,
        resample_poly(samples, VOICE_SAMPLE_RATE // step, source_rate // step),
        VOICE_SAMPLE_RATE,
        subtype="PCM_16",
    )
    if sha256(voice) != VOICE_SHA256:
        raise SystemExit(f"{voice} does not match the pinned {VOICE_SHA256}")
    else:
        pass
    return voice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument(
        "--model", action="append", help="repo to provision (default: all pinned)"
    )
    arguments = parser.parse_args()
    pins = json.loads((CI_DIRECTORY / "models.json").read_text())
    for repo in arguments.model or list(pins):
        print(
            f"model {repo}: {provision_model(arguments.data_root, repo, pins[repo])}",
            flush=True,
        )
    print(f"corpus: {provision_corpus(arguments.data_root)}", flush=True)
    print(f"voice: {provision_voice(arguments.data_root)}", flush=True)


if __name__ == "__main__":
    main()
