# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Inputs, reference format and metrics for the generate API GPU tests.

The Gemma 4 tests run a real checkpoint through ``AutoModel`` and compare
its next-token distributions with Hugging Face references. The references
and the dogs image live in the :data:`GENERATE_API_DATA` tarball on the
public S3 bucket, not in the repo, and ``regenerate_references.py`` writes
a new tarball. The FLUX.2-dev test runs ``AutoPipeline`` and compares its
images with diffusers images also stored in S3. Both sides read the inputs
from here, so the references and the tests cannot drift apart.

A reference keeps each step's greedy token and its top :data:`TOP_K` log
probabilities. KL divergence is measured over those tokens plus one bucket
for the rest of the vocabulary, a lower bound on the full-vocabulary KL that
stays close to it for a peaked distribution.
"""

from __future__ import annotations

import dataclasses
import functools
import gzip
import hashlib
import io
import json
import os
import tarfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
import numpy.typing as npt
from huggingface_hub import snapshot_download
from max.driver import Accelerator
from max.pipelines import AutoModel
from max.pipelines.lib import PIPELINE_REGISTRY
from max.pipelines.modeling.types import PipelineTask
from PIL import Image, ImageDraw
from test_common.numpy_encoder import NumpyDecoder
from test_common.storage import load_from_s3, load_from_tar

GENERATE_API_DATA = (
    "s3://modular-bazel-artifacts-public/artifacts/generate_api_references/1/"
    "4a2db75d4be3e35c399ea229c3a9df5b166afccbc376e2a25d58dacc6873ef1c/"
    "generate_api_references.tar.gz"
)
"""The tarball with ``inputs/dogs_512x384.jpg`` and
``references/<gpu>/<run>.json``. The path's second-to-last part is the
tarball's sha256."""

DATA_TARBALL_NAME = "generate_api_references.tar.gz"

DATA_TARBALL_MTIME = 1767225600
"""2026-01-01 UTC, the mtime of every file in a data tarball."""

GEMMA4 = "google/gemma-4-31B-it"
FLUX2_DEV = "black-forest-labs/FLUX.2-dev"

TOP_K = 32
"""The log probabilities a reference keeps per step."""

TEXT_STEPS = 10
"""Greedy steps per text and multimodal prompt."""

GEMMA4_MAX_LENGTH = 4096
"""Keeps the KV cache small, since every prompt is short."""

NEAR_TIE_NATS = 0.5
"""The largest gap between the reference's top two log probabilities at a
step where MAX may pick the other token."""

INFRA_MARKER = "GENERATE_API_INFRA:"
"""Printed before a failure that comes from the runner rather than the
model, so the scheduled workflow records it as infra."""

CONFIDENT_PROBABILITY = 0.9
"""A step whose reference top-1 probability is at least this is confident,
and gets the tighter KL bound."""

TEXT_PROMPTS = [
    "Explain in one sentence why the sky is blue.",
    "Write a short Python function that returns the nth Fibonacci number.",
    "Translate to French: The weather is nice today, so we will walk to the"
    " market.",
]


class Step(TypedDict):
    """One greedy step of a reference."""

    token: int
    ids: list[int]
    logprobs: list[float]


class _RowBase(TypedDict):
    prompt_ids: list[int]
    steps: list[Step]


class TextRow(_RowBase):
    """One text prompt of a reference."""

    prompt: str


class ConversationRow(_RowBase):
    """One conversation of a reference."""

    image: str
    question: str


Row = TextRow | ConversationRow


class Reference(TypedDict):
    model: str
    revision: str
    device: str
    torch: str
    transformers: str
    attn_implementation: str
    top_k: int
    rows: list[Row]


def text_messages(prompt: str) -> list[dict[str, str]]:
    """Returns a one-turn text conversation, which builds the prompt ids."""
    return [{"role": "user", "content": prompt}]


def shapes_image() -> Image.Image:
    """Draws a red circle and a blue square on white."""
    image = Image.new("RGB", (384, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 58, 180, 198), fill=(220, 30, 30))
    draw.rectangle((214, 58, 354, 198), fill=(30, 60, 220))
    return image


def _dogs_image() -> Image.Image:
    # Downscaled to 512x384 from max/examples/custom_ops/dogs.jpg (1920x1440).
    path = data_dir(GENERATE_API_DATA) / "inputs" / "dogs_512x384.jpg"
    return Image.open(path).convert("RGB")


IMAGES: Mapping[str, Callable[[], Image.Image]] = {
    "dogs": _dogs_image,
    "shapes": shapes_image,
}


@dataclasses.dataclass(frozen=True)
class Conversation:
    image: str
    """A key of :data:`IMAGES`."""

    question: str

    def messages(self) -> list[dict[str, object]]:
        """Returns the conversation in Hugging Face format."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": IMAGES[self.image]()},
                    {"type": "text", "text": self.question},
                ],
            }
        ]


CONVERSATIONS = [
    Conversation("dogs", "Describe this image in one sentence."),
    Conversation(
        "shapes", "What shapes are in this picture, and what color is each?"
    ),
]


def log_softmax(logits: npt.ArrayLike) -> npt.NDArray[np.float64]:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max()
    return shifted - np.log(np.exp(shifted).sum())


def top_k_step(logits: npt.ArrayLike) -> Step:
    """Returns the greedy token and the top-k log probabilities of a step."""
    logprobs = log_softmax(logits)
    ids = np.argsort(-logprobs, kind="stable")[:TOP_K]
    return {
        "token": int(ids[0]),
        "ids": [int(i) for i in ids],
        "logprobs": [round(float(logprobs[i]), 6) for i in ids],
    }


@dataclasses.dataclass(frozen=True)
class StepMetrics:
    kl: float
    """KL(reference || MAX) over the reference's top-k tokens and one bucket
    for the rest of the vocabulary."""

    cos: float
    """The cosine distance between the two models' probabilities over the
    same buckets."""

    reference_top1: float
    """The reference's top-1 probability."""


def step_metrics(reference: Step, logits: npt.ArrayLike) -> StepMetrics:
    ids = np.asarray(reference["ids"])
    ref_lp = np.asarray(reference["logprobs"], dtype=np.float64)
    max_lp = log_softmax(logits)[ids]
    ref_p = np.exp(ref_lp)
    max_p = np.exp(max_lp)
    ref_buckets = np.append(ref_p, max(1.0 - ref_p.sum(), 1e-12))
    max_buckets = np.append(max_p, max(1.0 - max_p.sum(), 1e-12))
    kl = float((ref_buckets * np.log(ref_buckets / max_buckets)).sum())
    cos = 1.0 - float(
        ref_buckets
        @ max_buckets
        / (np.linalg.norm(ref_buckets) * np.linalg.norm(max_buckets))
    )
    return StepMetrics(max(kl, 0.0), max(cos, 0.0), float(ref_p[0]))


def check_near_tie(step: Step, max_token: int, label: str, index: int) -> None:
    """Fails unless the token MAX picked is a near tie in the reference.

    MAX's token has to be one of the reference's top-k tokens, with a log
    probability within :data:`NEAR_TIE_NATS` of the reference's top token.
    """
    where = f"{label}, step {index}"
    assert max_token in step["ids"], (
        f"{where}: MAX picked token {max_token}, which is not in the"
        f" reference's top {len(step['ids'])}."
    )
    gap = step["logprobs"][0] - step["logprobs"][step["ids"].index(max_token)]
    assert gap < NEAR_TIE_NATS, (
        f"{where}: MAX picked token {max_token}, which the reference puts"
        f" {gap:.2f} nats below its top token, more than the"
        f" {NEAR_TIE_NATS} near-tie bound."
    )


def first_divergence(
    tokens: Sequence[int], reference: Sequence[int]
) -> int | None:
    for index, (token, expected) in enumerate(
        zip(tokens, reference, strict=True)
    ):
        if int(token) != int(expected):
            return index
    return None


class _GenerateOutput(Protocol):
    @property
    def logits(self) -> Sequence[npt.NDArray[np.float32]] | None: ...


class _Generator(Protocol):
    def generate(
        self,
        input_ids: Sequence[Sequence[int]],
        *,
        max_new_tokens: int,
        top_k: int,
        ignore_eos: bool,
        output_logits: bool,
    ) -> _GenerateOutput: ...


def prefill_reference_steps(
    model: _Generator,
    prompt_ids: Sequence[int],
    reference_tokens: Sequence[int],
    logits: list[npt.NDArray[np.float32]],
    divergence: int,
) -> None:
    """Replaces each step after ``divergence`` with a prefilled one.

    Each later step prefills the prompt and the reference's tokens so far,
    so it sees the reference's context.
    """
    for step in range(divergence + 1, len(reference_tokens)):
        forced = model.generate(
            [[*prompt_ids, *reference_tokens[:step]]],
            max_new_tokens=1,
            top_k=1,
            ignore_eos=True,
            output_logits=True,
        )
        assert forced.logits is not None
        logits[step] = forced.logits[0][0]


def gpu_kind() -> str:
    """Returns the references' GPU directory for the first GPU."""
    name = Accelerator().model_name
    if "B200" in name:
        return "b200"
    if "MI355" in name:
        return "mi355"
    message = f"The generate API tests have no references for {name}."
    print(f"{INFRA_MARKER} {message}")
    raise ValueError(message)


def device_memory_used_gib() -> float | None:
    stats = Accelerator().stats
    if "total_memory" not in stats:
        return None
    return round((stats["total_memory"] - stats["free_memory"]) / 2**30, 1)


def check_module_v3(model: AutoModel, task: PipelineTask) -> str:
    """Returns the resolved architecture name, failing unless it is ModuleV3."""
    arch = PIPELINE_REGISTRY.architecture_for_config(
        model.pipeline_config, task=task
    )
    assert arch is not None, "The config resolved to no architecture."
    print(f"Resolved architecture: {arch.name}")
    assert arch.name.endswith("_ModuleV3"), (
        f"Expected a ModuleV3 architecture, but the config resolved to"
        f" {arch.name}."
    )
    return arch.name


def load_gemma4(revision: str, max_batch_size: int = 1) -> AutoModel:
    return AutoModel(
        GEMMA4,
        max_batch_size=max_batch_size,
        prefer_module_v3=True,
        max_length=GEMMA4_MAX_LENGTH,
        quantization_encoding="bfloat16",
        huggingface_model_revision=revision,
        huggingface_weight_revision=revision,
    )


def check_sha256(path: Path, s3_path: str) -> None:
    """Fails unless ``path`` has the sha256 named in ``s3_path``."""
    expected = s3_path.split("/")[-2]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, (
        f"{s3_path} downloaded with sha256 {actual}, not {expected}."
    )


@functools.cache
def data_dir(s3_path: str) -> Path:
    """Downloads and extracts a data tarball, after checking its sha256."""
    try:
        path = Path(load_from_s3(s3_path))
    except Exception as error:
        message = f"Downloading {s3_path} failed: {error}"
        print(f"{INFRA_MARKER} {message}")
        raise
    check_sha256(path, s3_path)
    return Path(load_from_tar(str(path)))


def references_dir() -> Path:
    """Returns this GPU's references, since each runner pool caches its own
    checkpoint revisions."""
    return data_dir(GENERATE_API_DATA) / "references" / gpu_kind()


def write_data_tarball(files: Mapping[str, bytes], path: Path) -> str:
    """Writes ``files`` to a reproducible tarball and returns its sha256.

    Names are sorted, the format is ustar, every file has mode 0644, owner
    0:0 with empty names and :data:`DATA_TARBALL_MTIME`, and the gzip stream
    is level 9 with no file name or time in its header.
    """
    archive = io.BytesIO()
    with tarfile.open(
        fileobj=archive, mode="w", format=tarfile.USTAR_FORMAT
    ) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size = len(files[name])
            info.mtime = DATA_TARBALL_MTIME
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(files[name]))
    data = gzip.compress(archive.getvalue(), compresslevel=9, mtime=0)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def check_revision(repo: str, revision: str) -> None:
    """Fails unless the cache holds the checkpoint revision the reference
    was made from."""
    try:
        snapshot_download(repo, revision=revision, local_files_only=True)
    except Exception as error:
        message = (
            f"The Hugging Face cache has no {repo} at {revision}, the revision"
            f" the {gpu_kind()} references were made from. Regenerate them"
            " with regenerate_references.py."
        )
        print(f"{INFRA_MARKER} {message}")
        raise AssertionError(message) from error


def load_reference(name: str) -> Reference:
    reference: Reference = json.loads((references_dir() / name).read_text())
    return reference


def dump_reference(reference: Reference) -> str:
    """Returns the reference as JSON with one step per line."""
    header = {key: value for key, value in reference.items() if key != "rows"}
    lines = ["{"]
    lines += [
        f" {json.dumps(key)}: {json.dumps(value)},"
        for key, value in header.items()
    ]
    lines.append(' "rows": [')
    rows = []
    for row in reference["rows"]:
        fields = ", ".join(
            f"{json.dumps(key)}: {json.dumps(value)}"
            for key, value in row.items()
            if key != "steps"
        )
        steps = ",\n".join(f"   {json.dumps(step)}" for step in row["steps"])
        rows.append(f'  {{{fields}, "steps": [\n{steps}\n  ]}}')
    lines.append(",\n".join(rows))
    lines += [" ]", "}", ""]
    return "\n".join(lines)


# FLUX.2-dev's references are the diffusers images logit verification stores
# for these prompts and settings, made with an earlier transformers and
# diffusers. transformers 5.12.1 with diffusers 0.37.0 makes different
# images, so regenerate_references.py --run flux2_dev does not reproduce them.
FLUX2_GOLDEN_TAR = (
    "s3://modular-bazel-artifacts-public/artifacts/"
    "torch_black-forest-labs_FLUX.2-dev-t2i_golden/1/"
    "09e2330e3ea8016979b145072037df30ac8a7c6a425c00d64ae340cd52455295/"
    "torch_black-forest-labs_FLUX.2-dev-t2i_golden.tar.gz"
)
FLUX2_GOLDEN_JSON = (
    "torch_black-forest-labs_FLUX.2-dev-t2i_bfloat16_golden.json"
)
FLUX2_REVISION = "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c"
FLUX2_HEIGHT = 1024
FLUX2_WIDTH = 1024
FLUX2_STEPS = 50
FLUX2_GUIDANCE = 3.5
FLUX2_SEED = 42


@dataclasses.dataclass(frozen=True)
class ImageRequest:
    name: str
    """The image's name in reports and output files."""

    prompt: str


FLUX2_REQUESTS = [
    ImageRequest(
        "fox",
        "A red fox sitting in fresh snow in a winter forest, looking directly"
        " at the camera, soft morning light filtering through the trees",
    ),
    ImageRequest(
        "lighthouse",
        "A lighthouse on a rocky cliff at sunset with an orange and purple"
        " sky, calm ocean in the background, photorealistic",
    ),
]


def load_flux2_references() -> dict[str, npt.NDArray[np.float32]]:
    """Returns each request's reference image, one set for both GPUs.

    The images are read from :data:`FLUX2_GOLDEN_TAR` as float arrays in
    [0, 1].
    """
    golden = NumpyDecoder().decode(
        (data_dir(FLUX2_GOLDEN_TAR) / FLUX2_GOLDEN_JSON).read_text()
    )
    by_prompt = {row["prompt"]: np.asarray(row["images"]) for row in golden}
    references = {}
    for request in FLUX2_REQUESTS:
        images = by_prompt[request.prompt]
        references[request.name] = (
            images[0] if images.ndim == 4 else images
        ).astype(np.float32)
    return references


class ImageLimits(TypedDict):
    ssim: float
    """The smallest SSIM of any image."""
    lpips: float
    """The largest LPIPS of any image."""


# Worst values, on the fox and identical across repeated runs: SSIM 0.9972
# on both GPUs, LPIPS 0.0015 on B200 and 0.0016 on MI355. LPIPS 3e-3 leaves
# about twice that and still catches 45 denoising steps instead of 50, which
# measured 0.0041.
IMAGE_THRESHOLDS: Mapping[str, ImageLimits] = {
    "b200": {"ssim": 0.98, "lpips": 3e-3},
    "mi355": {"ssim": 0.98, "lpips": 3e-3},
}
"""FLUX.2-dev's thresholds per GPU."""


def check_images(values: ImageLimits) -> None:
    """Fails if the worst image is past FLUX.2-dev's thresholds."""
    limits = IMAGE_THRESHOLDS[gpu_kind()]
    print(f"flux2_dev: {values}")
    assert values["ssim"] >= limits["ssim"], (
        f"flux2_dev: SSIM {values['ssim']:.4f} is under the"
        f" {limits['ssim']} threshold."
    )
    assert values["lpips"] <= limits["lpips"], (
        f"flux2_dev: LPIPS {values['lpips']:.4f} is over the"
        f" {limits['lpips']} threshold."
    )


def save_output(name: str, data: bytes) -> None:
    """Saves a file as a test output."""
    out = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if out:
        Path(out, name).write_bytes(data)


class Report:
    """Collects timings and metrics, printed and saved as a test output."""

    def __init__(self, run: str) -> None:
        self.run = run
        self.data: dict[str, object] = {"run": run, "gpu": gpu_kind()}

    def time(self, key: str, start: float) -> None:
        self.data[key] = round(time.monotonic() - start, 2)

    def save(self) -> None:
        text = json.dumps(self.data, indent=2, default=str)
        print(f"GENERATE_API_REPORT {text}")
        out = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if out:
            Path(out, f"{self.run}.json").write_text(text)


class Limits(TypedDict):
    kl: float
    """The largest KL of any step."""
    kl_confident: float
    """The largest KL of a confident step."""
    cos: float
    """The largest cosine distance of any step."""


THRESHOLDS: Mapping[str, Mapping[str, Limits]] = {
    # Measured over two runs, which matched exactly, single and batched:
    # B200 KL 8.9e-3, confident KL 8.3e-4, cosine 8.4e-3.
    # MI355 KL 1.0e-2, confident KL 1.1e-3, cosine 8.4e-3.
    "gemma4_text": {
        "b200": {"kl": 3e-2, "kl_confident": 3e-3, "cos": 3e-2},
        "mi355": {"kl": 3e-2, "kl_confident": 3e-3, "cos": 3e-2},
    },
    # The worst step is the dogs conversation's second step, where the
    # reference splits its top two tokens about 0.63/0.23 and MAX about
    # 0.79/0.13. Measured over two runs, which matched exactly:
    # B200 KL 7.1e-2, confident KL 2.2e-2, cosine 2.2e-2.
    # MI355 KL 9.7e-2, confident KL 1.6e-2, cosine 3.0e-2.
    "gemma4_multimodal": {
        "b200": {"kl": 0.2, "kl_confident": 6e-2, "cos": 6e-2},
        "mi355": {"kl": 0.25, "kl_confident": 6e-2, "cos": 8e-2},
    },
}


def thresholds(run: str) -> Limits:
    return THRESHOLDS[run][gpu_kind()]


@dataclasses.dataclass
class RowComparison:
    """How one MAX row compares with its reference row."""

    first_divergence: int | None
    text: str
    reference_text: str
    metrics: list[StepMetrics]

    def summary(self) -> dict[str, object]:
        return {
            "first_divergence": self.first_divergence,
            "text": self.text,
            "reference_text": self.reference_text,
            "kl_per_step": [round(m.kl, 6) for m in self.metrics],
            "cos_per_step": [round(m.cos, 6) for m in self.metrics],
            "reference_top1_per_step": [
                round(m.reference_top1, 4) for m in self.metrics
            ],
        }


def compare_steps(
    reference_steps: Sequence[Step],
    logits: Sequence[npt.NDArray[np.float32]],
) -> list[StepMetrics]:
    return [
        step_metrics(ref, row)
        for ref, row in zip(reference_steps, logits, strict=True)
    ]


def worst(rows: Sequence[RowComparison]) -> Limits:
    """Returns the worst values over every compared step."""
    metrics = [m for row in rows for m in row.metrics]
    confident = [
        m.kl for m in metrics if m.reference_top1 >= CONFIDENT_PROBABILITY
    ]
    return {
        "kl": max(m.kl for m in metrics),
        "kl_confident": max(confident, default=0.0),
        "cos": max(m.cos for m in metrics),
    }


def check_worst(run: str, values: Limits) -> None:
    """Fails if any worst value is over the run's threshold."""
    limits = thresholds(run)
    print(f"{run}: {values}")
    for name, value, limit in (
        ("kl", values["kl"], limits["kl"]),
        ("kl_confident", values["kl_confident"], limits["kl_confident"]),
        ("cos", values["cos"], limits["cos"]),
    ):
        assert value <= limit, (
            f"{run}: {name} {value:.3e} is over the {limit:.1e} threshold."
        )


def dump_logits(name: str, logits: Sequence[npt.NDArray[np.float32]]) -> None:
    """Saves each row's logits as a test output.

    Only when ``GENERATE_API_DUMP_LOGITS=1``.
    """
    out = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if out and os.environ.get("GENERATE_API_DUMP_LOGITS") == "1":
        np.savez_compressed(Path(out, name), *logits)
