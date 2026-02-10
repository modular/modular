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

from __future__ import annotations

import contextlib
import dataclasses
import enum
import functools
import json
import os
import sys
import time
import traceback
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import click
from generate_llm_logits import Flake, generate_llm_logits
from max.pipelines.lib.device_specs import (
    device_specs_from_normalized_device_handle,
    normalize_device_specs_input,
)
from test_common.evaluate import ModelOutput
from test_common.numpy_encoder import NumpyDecoder
from test_common.process_isolation import run_in_isolated_process
from test_common.storage import load_from_tar
from verify import DiscrepancyReport, verify
from verify import ModelModality as Modality

# This is far from a universal standard, but this is the closest to a standard
# that I could find: BSD-derived programs sometimes use exit codes from
# "sysexits.h", which defines this exit code as "temp failure; user is invited
# to retry".  generate_llm_logits will emit this if it detects a failure is
# likely caused by a network flake and could be resolved by a retry.
EX_TEMPFAIL = 75


def validate_hf_token() -> None:
    """
    It's a regular occurrence that people are asked to run logit verification
    locally, and not everyone has an HF_TOKEN set. Let's help them out
    """
    if os.getenv("HF_TOKEN") is None:
        raise ValueError(
            "Environment variable `HF_TOKEN` must be set. "
            "See https://www.notion.so/modularai/HuggingFace-Access-Token-29d1044d37bb809fbe70e37428faf9da"
        )


class DeviceKind(enum.Enum):
    CPU = "cpu"
    GPU = "gpu"


class VerificationStatus(str, enum.Enum):
    OK = "ok"
    INVALID = "invalid"
    ERROR = "error"
    FLAKE = "flake"
    INFRA = "infra"

    @property
    def emoji(self) -> str:
        return _VERDICT_EMOJI[self]


_VERDICT_EMOJI = {
    VerificationStatus.OK: "✅",
    VerificationStatus.INVALID: "🟡",
    VerificationStatus.ERROR: "❌",
    VerificationStatus.FLAKE: "❄️",
    VerificationStatus.INFRA: "🧯",
}


@dataclass
class VerificationVerdict:
    status: VerificationStatus
    discrepancy_report: DiscrepancyReport | None = None
    kl_div_threshold: float | None = None

    @property
    def emoji(self) -> str:
        return self.status.emoji


def resolve_rlocation(rloc: str) -> Path:
    from python.runfiles import runfiles

    r = runfiles.Create()
    assert r
    resolved = r.Rlocation(rloc)
    if resolved is None:
        raise FileNotFoundError(f"Rlocation {rloc!r} could not be resolved")
    return Path(resolved)


def verdict_sorting_key(
    model_name_and_verdict: tuple[str, VerificationVerdict],
) -> tuple[int, str]:
    """Sort key for model names, ordered by dtype, then alphabetically."""
    model_name, _ = model_name_and_verdict

    # Determine dtype priority
    sort_order_by_dtype = ["float32", "bfloat16", "float8", "q4_k", "gptq"]
    name_lower = model_name.lower()

    dtype_priority = len(sort_order_by_dtype)  # Default for unknown dtypes
    for i, dtype in enumerate(sort_order_by_dtype):
        if name_lower.endswith(dtype):
            dtype_priority = i
            break

    return (dtype_priority, name_lower)


def save_verdicts_to_json(
    verdicts: dict[str, VerificationVerdict], filepath: Path
) -> None:
    """Save verdicts to JSON file."""
    verdicts_dict = {k: dataclasses.asdict(v) for k, v in verdicts.items()}
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(verdicts_dict, f, indent=2)


def load_verdicts_from_json(filepath: Path) -> dict[str, VerificationVerdict]:
    """Load verdicts from JSON file."""
    try:
        with open(filepath) as f:
            data = json.load(f)
        return {
            k: VerificationVerdict(
                status=VerificationStatus(v["status"]),
                discrepancy_report=DiscrepancyReport(**v["discrepancy_report"])
                if v.get("discrepancy_report")
                else None,
            )
            for k, v in data.items()
        }
    except Exception as e:
        print(f"Error loading verdicts from JSON: {e}", file=sys.stderr)
        return {}


def display_name(name: str) -> str:
    """Remove the org name from the model name for display purposes.

    Args:
        name: Full model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")

    Returns:
        Display name with org prefix removed (e.g., "Llama-3.1-8B-Instruct")
    """
    return name.split("/", 1)[-1]


def compute_diff(
    current_verdict: VerificationVerdict,
    previous_verdict: VerificationVerdict,
) -> str:
    """Compute the difference between current and previous metric values.

    Args:
        current_verdict: Current verification verdict
        previous_verdict: Previous verification verdict
        metric_type: Either "kl_div" or "mae"

    Returns:
        Formatted diff string (+X.XXe-XX (1.3x) if worse results,
          -X.XXe-XX (1.4x) if better results,
          ---- if no change)
    """
    # Early return if discrepancy reports missing
    if (
        current_verdict.discrepancy_report is None
        or previous_verdict.discrepancy_report is None
    ):
        return "N/A"

    prev_val = previous_verdict.discrepancy_report.default_metric
    curr_val = current_verdict.discrepancy_report.default_metric

    diff = float(f"{curr_val:.2e}") - float(f"{prev_val:.2e}")

    if diff == 0:
        return "---"

    ratio_indicator = ""
    if prev_val != 0:
        abs_ratio = abs(curr_val / prev_val)
        if abs_ratio < 1:
            abs_ratio = 1 / abs_ratio

        if abs_ratio >= 100:
            ratio_indicator = " (>99x)"
        elif abs_ratio >= 10:
            ratio_indicator = f" ({int(abs_ratio):>3}x)"
        else:
            ratio_indicator = f" ({abs_ratio:3.1f}x)"

    return f"{diff:+.2e}{ratio_indicator}"


def dump_results(
    verdicts: Mapping[str, VerificationVerdict],
    *,
    to: TextIO = sys.stdout,
    previous_verdicts: Mapping[str, VerificationVerdict] | None = None,
) -> None:
    # Even if verdicts is empty, we want to make sure to call write.  When we
    # call this from 'main', click passes us a LazyFile, and if we don't write
    # anything, we won't create the output file, which breaks downstream
    # workflows.

    # Warning: The logits verification pipeline parses these results
    # using grep/awk.
    # Please verify that this doesn't break before changing the output format

    any_logit, any_embedding, any_failed = False, False, False
    for verdict in verdicts.values():
        if verdict.discrepancy_report is None:
            any_failed = True
        elif verdict.discrepancy_report.model_modality == Modality.LOGIT:
            any_logit = True
        elif verdict.discrepancy_report.model_modality == Modality.EMBEDDING:
            any_embedding = True

    if node := os.environ.get("NODE_NAME"):
        to.write(f"\n\nRan on node: {node}")

    if any_failed:
        to.write("\n\n## Failed/Crashed Models\n")
        to.write("| Status | Model |\n")
        to.write("| :---:  | :---  |\n")

        for name, verdict in sorted(verdicts.items(), key=verdict_sorting_key):
            if verdict.discrepancy_report is not None:
                continue
            to.write(f"| {verdict.emoji} | {display_name(name)} | N/A |\n")

    if any_logit:
        to.write("\n\n## LLMs\n")
        to.write(
            "**KL Div (max)** = max KL Div over all prompts. This is the threshold used for pass/fail checks.\n"
            "**KL Div (avg)** = average over all prompts (lower is better)\n"
            "**Diff** = change of the average KL Div from previous run\n"
            "  • Negative = accuracy improved\n"
            "  • Positive = accuracy worsened\n"
            "  • N/A = no previous verdict\n"
            "  • --- = no change\n"
        )
        to.write(
            "| Status | Model | KL Div (max) | KL Div (avg) | Diff (avg) |\n"
        )
        to.write(
            "| :----: | :---- | :----------: | :----------: | :--------: |\n"
        )

        for name, verdict in sorted(verdicts.items(), key=verdict_sorting_key):
            if verdict.discrepancy_report is None:
                continue
            if verdict.discrepancy_report.model_modality != Modality.LOGIT:
                continue
            kl_max = f"{verdict.discrepancy_report.max_kl_div:.2e}"
            threshold_max = f"{verdict.kl_div_threshold:.2e}"
            kl_avg = f"{verdict.discrepancy_report.avg_kl_div:.2e}"
            if (
                verdict.discrepancy_report.max_kl_div is None
                or verdict.kl_div_threshold is None
            ):
                kl_max_str = f"{kl_max} (? {threshold_max})"
            elif (
                verdict.discrepancy_report.max_kl_div > verdict.kl_div_threshold
            ):
                kl_max_str = f"{kl_max} (>{threshold_max})"
            else:
                kl_max_str = f"{kl_max} (<={threshold_max})"

            diff_str = "N/A"
            if previous_verdicts and name in previous_verdicts:
                diff_str = compute_diff(verdict, previous_verdicts[name])

            to.write(
                f"| {verdict.emoji} | {display_name(name)} | {kl_max_str} | {kl_avg} | {diff_str} |\n"
            )

    if any_embedding:
        to.write("\n\n## Embedding Models\n")
        to.write("| Status | Model | MAE | Diff |\n")
        to.write("| :----: | :---  |:---:| :---:|\n")

        for name, verdict in sorted(verdicts.items(), key=verdict_sorting_key):
            if verdict.discrepancy_report is None:
                continue
            if verdict.discrepancy_report.model_modality != Modality.EMBEDDING:
                continue
            mae = f"{verdict.discrepancy_report.avg_mae:.2e}"

            diff_str = "N/A"
            if previous_verdicts and name in previous_verdicts:
                diff_str = compute_diff(verdict, previous_verdicts[name])

            to.write(
                f"| {verdict.emoji} | {display_name(name)} | {mae} | {diff_str} |\n"
            )


@dataclass
class TagFilter:
    """User-provided filters on a tag list."""

    must_have: Sequence[str] = field(default_factory=list)
    must_not_have: Sequence[str] = field(default_factory=list)

    def satisfied_by(self, tags: Sequence[str]) -> bool:
        """Determines if this filter is satisfied by a tag list."""
        if not all(required_tag in tags for required_tag in self.must_have):
            return False
        return not any(
            forbidden_tag in tags for forbidden_tag in self.must_not_have
        )


class TagFilterParamType(click.ParamType):
    name = "tag filter"

    def convert(
        self,
        value: str | TagFilter,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> TagFilter:
        # Unsure why click sometimes tries to re-convert an already-converted
        # value, but it does.
        if isinstance(value, TagFilter):
            return value
        assert isinstance(value, str), f"Value of unexpected type {type(value)}"
        if not value:
            return TagFilter()
        parts = value.split(",")
        required = []
        forbidden = []
        for part in parts:
            if part.startswith("+"):
                required.append(part[1:])
            elif part.startswith("-"):
                forbidden.append(part[1:])
            else:
                raise ValueError(
                    f"Tag filter part {part!r} does not start with '+' or '-'"
                )
        return TagFilter(must_have=required, must_not_have=forbidden)


@dataclass
class PregeneratedTorchGoldens:
    tar_file: str
    """S3 path to the tar file containing the bundled golden json files."""
    json_file: str
    """Name of the json file containing the golden logits."""


class InfraError(Exception):
    """Raised when an error with the runner environment has been encountered."""


@contextlib.contextmanager
def detect_infra_errors() -> Generator[None, None, None]:
    try:
        yield
    except ValueError as exc:
        exc_str = str(exc)
        if (
            'failed to create device: No supported "gpu" device available.'
            in exc_str
            and "CUDA call failed: CUDA_ERROR_UNKNOWN" in exc_str
        ):
            raise InfraError(
                "GPU device seems to have fallen off from runner"
            ) from exc
        raise


def generate_llm_logits_with_optional_retry(
    *,
    framework: str,
    device: str,
    pipeline: str,
    encoding: str,
    output_path: Path,
    reference: list[ModelOutput] | None = None,
    retry_on_flake: bool = True,
    timeout: int | None = None,
) -> None:
    """Generate logits with optional retry capability.

    If retry_on_flake is True, will retry once after 60 seconds on Flake exception.
    """
    device_specs = device_specs_from_normalized_device_handle(
        normalize_device_specs_input(device)
    )

    def attempt() -> None:
        run_in_isolated_process(
            functools.partial(
                generate_llm_logits,
                framework_name=framework,
                device_specs=device_specs,
                pipeline_name=pipeline,
                encoding_name=encoding,
                output_path=output_path,
                print_output=False,
                reference=reference,
            ),
            timeout=timeout if timeout is not None else 1200,
        )

    try:
        attempt()
    except Flake:
        if not retry_on_flake:
            raise
        print(
            "Generating LLM logits flaked.... waiting a minute and "
            "trying again.",
            file=sys.stderr,
        )
        time.sleep(60)
        print("OK, trying again.", file=sys.stderr)
        try:
            attempt()
        except Flake:
            print(
                "Flake remains after second attempt.  Giving up this time.",
                file=sys.stderr,
            )
            raise


def run_llm_verification(
    *,
    device_type: DeviceKind,
    devices: str,
    find_tolerances: bool,
    print_suggested_tolerances: bool,
    pipeline: str,
    encoding: str,
    pregenerated_torch_goldens: PregeneratedTorchGoldens | None = None,
    absolute_tolerance: float | None = None,
    relative_tolerance: float | None = None,
    cos_dist_threshold: float | None = None,
    kl_div_threshold: float | None = None,
    timeout: int | None = None,
) -> VerificationVerdict:
    """Run a Llama3 verification with the given model and weights encoding.

    extra_verify_flags are passed to
    max/tests/integration/architectures/llama3/verify.py -- check that script
    for details on acceptable flags.
    """

    fssafe_pipeline = pipeline.replace("/", "_")

    # Run the torch baseline or load it from golden.
    if pregenerated_torch_goldens is not None:
        # This workflow runs on an A10. The Torch reference runs out of memory
        # on an A10, so it was run manually on an A100 and the result goldens
        # uploaded. Use these pre-generated goldens in this case.
        tar_file = load_from_tar(pregenerated_torch_goldens.tar_file)
        torch_golden_path = Path(tar_file, pregenerated_torch_goldens.json_file)
    else:
        torch_golden_path = Path(
            f"/tmp/goldens_torch_{device_type.value}_{fssafe_pipeline}_{encoding}.json"
        )
        generate_llm_logits_with_optional_retry(
            framework="torch",
            device=devices,
            pipeline=pipeline,
            encoding=encoding,
            output_path=torch_golden_path,
            timeout=timeout,
        )

    torch_results: list[ModelOutput] = NumpyDecoder().decode(
        torch_golden_path.read_text()
    )

    # When find_tolerances is enabled, we set all tolerances to a lower bound and enable print_suggested_tolerances.
    # This ensures we find the suggested lower bound tolerances for a model.
    if find_tolerances:
        print_suggested_tolerances = True
        kl_div_threshold = 1e-10
        cos_dist_threshold = 1e-10
        absolute_tolerance = 1e-4
        relative_tolerance = 1e-4

    max_golden_path = Path(
        f"/tmp/goldens_max_{device_type.value}_{fssafe_pipeline}_{encoding}.json"
    )
    generate_llm_logits_with_optional_retry(
        framework="max",
        device=devices,
        pipeline=pipeline,
        encoding=encoding,
        output_path=max_golden_path,
        reference=torch_results,
        timeout=timeout,
    )

    eval_metrics = []
    if absolute_tolerance is not None and relative_tolerance is not None:
        eval_metrics.append("tol")
    if cos_dist_threshold is not None:
        eval_metrics.append("cos")
    if kl_div_threshold is not None:
        eval_metrics.append("kl")

    if not eval_metrics:
        raise ValueError(
            "Please provide absolute, relative, cos, or kldiv error thresholds."
            " Otherwise no metrics will be computed."
        )

    try:
        result = verify(
            pipeline_outputs=max_golden_path,
            torch_outputs=torch_golden_path,
            eval_metric=eval_metrics,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            cos_dist_threshold=cos_dist_threshold,
            kl_div_threshold=kl_div_threshold,
            print_suggested_tolerances=print_suggested_tolerances,
        )
        status = (
            VerificationStatus.OK
            if result.passed
            else VerificationStatus.INVALID
        )
        return VerificationVerdict(
            status=status,
            discrepancy_report=result.discrepancy_report,
            kl_div_threshold=kl_div_threshold,
        )
    except Exception:
        traceback.print_exc()
        return VerificationVerdict(status=VerificationStatus.ERROR)


@dataclass
class PipelineDef:
    """Definition of the requirements and method of running a pipeline.

    'compatible_with' lists all device types this pipeline is compatible with.
    'run' should run and verify the pipeline results, returning a
    VerificationVerdict with the result of the verification, or alternatively
    raising an exception (same as returning VerificationVerdict.ERROR).
    """

    compatible_with: Sequence[DeviceKind]
    run: Callable[[DeviceKind, str, bool, bool], VerificationVerdict]
    tags: Sequence[str] = field(default_factory=list)

    def run_protected(
        self,
        device_type: DeviceKind,
        devices: str,
        find_tolerances: bool,
        print_suggested_tolerances: bool,
    ) -> VerificationVerdict:
        try:
            with detect_infra_errors():
                return self.run(
                    device_type,
                    devices,
                    find_tolerances,
                    print_suggested_tolerances,
                )
        except Flake:
            return VerificationVerdict(status=VerificationStatus.FLAKE)
        except InfraError:
            traceback.print_exc()
            return VerificationVerdict(status=VerificationStatus.INFRA)
        except Exception:
            traceback.print_exc()
            return VerificationVerdict(status=VerificationStatus.ERROR)


# Helper function to create pipeline runner
def _make_pipeline_runner(
    *,
    pipeline: str,
    encoding: str,
    pregenerated_torch_goldens: PregeneratedTorchGoldens | None = None,
    absolute_tolerance: float | None = None,
    relative_tolerance: float | None = None,
    cos_dist_threshold: float | None = None,
    kl_div_threshold: float | None = None,
    timeout: int | None = None,
) -> Callable[[DeviceKind, str, bool, bool], VerificationVerdict]:
    """
    Build and return a small closure that executes `run_llm_verification`
    for a single model configuration.

    Args:
        pipeline: Name of the model / pipeline to verify.
        encoding: Weight / activation dtype (e.g. "float32", "bfloat16").
        pregenerated_torch_goldens: Paths to the pregenerated torch golden
            logits. If provided, the torch golden values are read from the file.
            Otherwise, Torch outputs are generated on the fly.
        absolute_tolerance: Per-token element-wise absolute tolerance (atol).
        relative_tolerance: Per-token element-wise relative tolerance (rtol).
        cos_dist_threshold: Per-token cosine-distance threshold
            (not element-wise).
        kl_div_threshold: Per-token KL-divergence threshold
            (not element-wise).
        timeout: Timeout in seconds for the verification.

    Returns:
        A callable that runs the verification and yields a `VerificationVerdict`.
    """
    return (
        lambda device_type,
        devices,
        find_tolerances,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            find_tolerances=find_tolerances,
            print_suggested_tolerances=print_suggested_tolerances,
            pipeline=pipeline,
            encoding=encoding,
            pregenerated_torch_goldens=pregenerated_torch_goldens,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            cos_dist_threshold=cos_dist_threshold,
            kl_div_threshold=kl_div_threshold,
            timeout=timeout,
        )
    )


PIPELINES = {
    # ========== Robust Pipelines ==========
    # The models here are considered robust. They are tested with all metrics.
    # Other models avoid absolute and relative tolerance because they are quite
    # noisy for inaccurate models.
    # Generally speaking, these models should have absolute and relative
    # tolerances below ~5e-2.
    "meta-llama/Meta-Llama-3-8B-Instruct-float32": PipelineDef(
        compatible_with=[DeviceKind.CPU, DeviceKind.GPU],
        tags=["big"],
        run=_make_pipeline_runner(
            pipeline="meta-llama/Meta-Llama-3-8B-Instruct",
            encoding="float32",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_llama_golden/6/03d7f428e3fdd43f6436ff19c5c5f7245e7cb71deacd17e8b0d0bd8f35701daa/torch_llama_golden.tar.gz",
                json_file="torch_llama3-8b_float32_golden.json",
            ),
            absolute_tolerance=2.9e-2,
            relative_tolerance=9.4e-2,
            cos_dist_threshold=2.6e-6,
            kl_div_threshold=8.6e-07,
        ),
    ),
    "meta-llama/Llama-3.1-8B-Instruct-float32": PipelineDef(
        compatible_with=[DeviceKind.CPU, DeviceKind.GPU],
        tags=["big"],
        run=_make_pipeline_runner(
            pipeline="meta-llama/Llama-3.1-8B-Instruct",
            encoding="float32",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_llama_golden/6/03d7f428e3fdd43f6436ff19c5c5f7245e7cb71deacd17e8b0d0bd8f35701daa/torch_llama_golden.tar.gz",
                json_file="torch_llama3_1_float32_golden.json",
            ),
            absolute_tolerance=2.4e-2,
            relative_tolerance=7.8e-3,
            cos_dist_threshold=3.3e-6,
            kl_div_threshold=1.0e-10,
        ),
    ),
    "sentence-transformers/all-mpnet-base-v2-float32": PipelineDef(
        compatible_with=[DeviceKind.CPU, DeviceKind.GPU],
        run=_make_pipeline_runner(
            pipeline="sentence-transformers/all-mpnet-base-v2",
            encoding="float32",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_mpnet_golden/1/d93f10114938b5102f529f436170e2eb33a3d2c76889acf3406b54603cc1be97/torch_mpnet_golden.tar.gz",
                json_file="torch_mpnet_float32_golden.json",
            ),
            # On CPU, mpnet passes with all values set to `1e-4`
            # GPU specifically requires these higher tolerances (30x worse).
            absolute_tolerance=2.3e-3,
            relative_tolerance=2.7e-2,
            cos_dist_threshold=2.1e-5,
            kl_div_threshold=1.0e-10,
        ),
    ),
    "unsloth/gpt-oss-20b-BF16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-multi", "no-b200-multi"],
        run=_make_pipeline_runner(
            pipeline="unsloth/gpt-oss-20b-BF16",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_gpt-oss_golden/3/808b22644ad4c499e44408f2e80a14367f8c7cc16a16c7df60c0b2227a1812c3/torch_gpt-oss_golden.tar.gz",
                json_file="torch_gpt-oss_bfloat16_golden.json",
            ),
            cos_dist_threshold=5.0e-03,
            kl_div_threshold=8.0e-02,
        ),
    ),
    "allenai/OLMo-1B-hf-float32": PipelineDef(
        compatible_with=[DeviceKind.CPU, DeviceKind.GPU],
        run=_make_pipeline_runner(
            pipeline="allenai/OLMo-1B-hf",
            encoding="float32",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_olmo-1b-hf_golden/1/cdb42a9758e27374be0b410b8d38a2f9a66fc3662346702a5930730da2a50302/torch_olmo-1b-hf_golden.tar.gz",
                json_file="torch_olmo-1b-hf_float32_golden.json",
            ),
            # On CPU, olmo passes with atol set to `5e-4`
            # GPU specifically requires these higher tolerances (160x worse).
            absolute_tolerance=3.7e-2,
            relative_tolerance=4.2e-2,
            cos_dist_threshold=8.2e-6,
            kl_div_threshold=6.6e-5,
        ),
    ),
    # ========== Brittle Pipelines ==========
    # The models here are considered brittle. They have never reached high
    # accuracy. They tend to be more sensitive to noise as code changes.
    # These models are only tested with aggregate metrics of cosine distance
    # and kl divergence.
    # Likely as cosine distance and kl divergence drop below ~1e-5, they should
    # be migrated to being a robust pipelines.
    "bartowski/Meta-Llama-3-8B-Instruct-GGUF-q4_k": PipelineDef(
        compatible_with=[DeviceKind.CPU],
        run=_make_pipeline_runner(
            pipeline="meta-llama/Meta-Llama-3-8B-Instruct",
            encoding="q4_k",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_llama3-8b_q4_k_golden/1/5078639f4386c9d7e11eff0b09de2917b41494aae1168efe9e3db2bb4c8be3ef/torch_llama3-8b_q4_k_golden.tar.gz",
                json_file="torch_llama3-8b_q4_k_golden.json",
            ),
            # TODO(AIPIPE-135): Something is wildly wrong about our Q4_K
            # pipeline.  We only pass with these sky-high tolerances --
            # something is very wrong but at least we will be able to detect
            # further regressions with this.
            cos_dist_threshold=0.39,
            kl_div_threshold=6.5,
        ),
    ),
    "meta-llama/Meta-Llama-3-8B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        run=_make_pipeline_runner(
            pipeline="meta-llama/Meta-Llama-3-8B-Instruct",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_llama3-8b_bfloat16_golden/1/1ba2f2214bfffa3e8e5f166d5d1d5573201efa4dcb5c668f4b84fb9466861e56/torch_llama3-8b_bfloat16_golden.tar.gz",
                json_file="torch_llama3-8b_bfloat16_golden.json",
            ),
            cos_dist_threshold=3.7e-2,
            kl_div_threshold=1.3e-1,
        ),
    ),
    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF-q4_k": PipelineDef(
        compatible_with=[DeviceKind.CPU],
        run=_make_pipeline_runner(
            pipeline="meta-llama/Llama-3.1-8B-Instruct",
            encoding="q4_k",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_llama3.1-8b_q4_k_golden/1/13c75e9e445ecf05c1fbc0f2f9e73361062c2efbc62c01575278cfa4d0a64b2e/torch_llama3.1-8b_q4_k_golden.tar.gz",
                json_file="torch_llama3.1-8b_q4_k_golden.json",
            ),
            # TODO(AIPIPE-135): Something is wildly wrong about our Q4_K
            # pipeline.  We only pass with these sky-high tolerances --
            # something is very wrong but at least we will be able to detect
            # further regressions with this.
            cos_dist_threshold=0.62,
            kl_div_threshold=6.8,
        ),
    ),
    "meta-llama/Llama-3.1-8B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        run=_make_pipeline_runner(
            pipeline="meta-llama/Llama-3.1-8B-Instruct",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_llama_golden/6/03d7f428e3fdd43f6436ff19c5c5f7245e7cb71deacd17e8b0d0bd8f35701daa/torch_llama_golden.tar.gz",
                json_file="torch_llama3_1_bfloat16_golden.json",
            ),
            cos_dist_threshold=2.0e-2,
            kl_div_threshold=4.0e-2,
        ),
    ),
    "meta-llama/Llama-3.1-8B-Instruct-data-parallel-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=[
            "nvidia-multi",
            "no-h100",
        ],  # TODO(MODEL-779): Accuracy issues on H100.
        run=_make_pipeline_runner(
            pipeline="meta-llama/Llama-3.1-8B-Instruct-data-parallel",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_llama_golden/6/03d7f428e3fdd43f6436ff19c5c5f7245e7cb71deacd17e8b0d0bd8f35701daa/torch_llama_golden.tar.gz",
                json_file="torch_llama3_1_bfloat16_golden.json",
            ),
            cos_dist_threshold=3.0e-4,
            kl_div_threshold=7.4e-3,
            timeout=1200,
        ),
    ),
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-float8-static": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["float8-support"],
        run=_make_pipeline_runner(
            pipeline="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-float8-static",
            encoding="float8_e4m3fn",
            # This model does not run with torch and transformers.
            # It only runs with vllm.
            # For now compare to the bfloat16 goldens cause we have them.
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_llama_golden/6/03d7f428e3fdd43f6436ff19c5c5f7245e7cb71deacd17e8b0d0bd8f35701daa/torch_llama_golden.tar.gz",
                json_file="torch_llama3_1_bfloat16_golden.json",
            ),
            cos_dist_threshold=9.7e-03,
            kl_div_threshold=8.6e-2,
        ),
    ),
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic-float8-dynamic": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["float8-support"],
        run=_make_pipeline_runner(
            pipeline="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
            encoding="float8_e4m3fn",
            # This model does not run with torch and transformers.
            # It only runs with vllm.
            # For now compare to the bfloat16 goldens cause we have them.
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_llama_golden/6/03d7f428e3fdd43f6436ff19c5c5f7245e7cb71deacd17e8b0d0bd8f35701daa/torch_llama_golden.tar.gz",
                json_file="torch_llama3_1_bfloat16_golden.json",
            ),
            cos_dist_threshold=1.4e-02,
            kl_div_threshold=4.1e-02,
        ),
    ),
    "nvidia/Llama-3.1-8B-Instruct-NVFP4": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-only", "no-h100"],
        run=_make_pipeline_runner(
            pipeline="nvidia/Llama-3.1-8B-Instruct-NVFP4",
            encoding="float4_e2m1fnx2",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/vllm_llama_3_1_8B_nvfp4/20260205_180241_nvidia-Llama-3.1-8B-Instruct-NVFP4_vllm.json.tar.gz",
                json_file="tmp/20260205_180241_nvidia-Llama-3.1-8B-Instruct-NVFP4_vllm.json",
            ),
            cos_dist_threshold=5.8e-04,
            kl_div_threshold=2.0e-01,
        ),
    ),
    "nvidia/Llama-3.1-405B-Instruct-NVFP4": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-only", "no-h100", "2xb200"],
        run=_make_pipeline_runner(
            pipeline="nvidia/Llama-3.1-405B-Instruct-NVFP4",
            encoding="float4_e2m1fnx2",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/vllm_llama_3_1_405B_nvfp4/20260210_031603_nvidia-Llama-3.1-405B-Instruct-NVFP4_vllm.json.tar.gz",
                json_file="tmp/20260210_031603_nvidia-Llama-3.1-405B-Instruct-NVFP4_vllm.json",
            ),
            cos_dist_threshold=3.4e-04,
            kl_div_threshold=8.3e-02,
        ),
    ),
    "meta-llama/Llama-3.2-1B-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-only"],
        run=_make_pipeline_runner(
            pipeline="meta-llama/Llama-3.2-1B",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_llama3.2-1b_bfloat16_golden/1/f77876a9612aba6f0df1ab1bd9f3819656f962a46e2b3133d11a4290c936de3a/torch_llama3.2-1b_bfloat16_golden.tar.gz",
                json_file="torch_llama3.2-1b_bfloat16_golden.json",
            ),
            cos_dist_threshold=2.5e-03,
            kl_div_threshold=1.0e-02,
        ),
    ),
    "meta-llama/Llama-3.3-70B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-multi"],
        run=_make_pipeline_runner(
            pipeline="meta-llama/Llama-3.3-70B-Instruct",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_llama3.3-70b-instruct_bfloat16_golden/1/e4a07947f7f23c0b1bc5d297bbe1575b637abb53270e99b284936bf6d2dc703f/torch_llama3.3-70b-instruct_bfloat16_golden.tar.gz",
                json_file="torch_llama3.3-70b-instruct_bfloat16_golden.json",
            ),
            # TODO(AITLIB-194): Reduce thresholds after fixing correctness.
            cos_dist_threshold=9.0e-04,
            kl_div_threshold=5.0e-03,
        ),
    ),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-multi"],
        run=_make_pipeline_runner(
            pipeline="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_llama4_golden/2/fbb8ae9654ca68a7066e05944eda991b5365821adabbe9bf210f5cbfaad6512f/torch_llama4_golden.tar.gz",
                json_file="torch_llama4_scout_bfloat16_golden.json",
            ),
            cos_dist_threshold=0.7,
            kl_div_threshold=8.0,
            timeout=900,
        ),
    ),
    "mistralai/Mistral-Nemo-Instruct-2407-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big"],
        run=_make_pipeline_runner(
            pipeline="mistralai/Mistral-Nemo-Instruct-2407",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_mistral_golden/1/6f4718625a01e6e8b9f002a0bfdad8098cfe78ce50b9cd4175f27b1f020b405a/torch_mistral_golden.tar.gz",
                json_file="torch_nemo-instruct-2407_bfloat16_golden.json",
            ),
            # TODO(AIPIPE-230): These tolerances are very high due to an accuracy regression.
            cos_dist_threshold=2.1e-2,
            kl_div_threshold=3.0e-2,
        ),
    ),
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big"],
        run=_make_pipeline_runner(
            pipeline="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            encoding="bfloat16",
            cos_dist_threshold=3.0e-03,
            kl_div_threshold=5.2e-3,
        ),
    ),
    "OpenGVLab/InternVL3-1B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        # TODO(KERN-1861): MI300x: Memory access fault by GPU node-2.
        tags=["nvidia-only"],
        run=_make_pipeline_runner(
            pipeline="OpenGVLab/InternVL3-1B-Instruct",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_internvl3-1b_golden/1/c0cc901fe70ef9f90801a921aae56e6c8a3ce11e7173cdeed27c46f147f29463/torch_internvl3-1b_golden.tar.gz",
                json_file="torch_internvl3-1b_bfloat16_golden.json",
            ),
            # TODO(MODELS-565): Fix InternVL correctness.
            cos_dist_threshold=4.0e-03,
            kl_div_threshold=1.5e-02,
        ),
    ),
    "OpenGVLab/InternVL3-8B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-multi"],
        run=_make_pipeline_runner(
            pipeline="OpenGVLab/InternVL3-8B-Instruct",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_internvl3-8b_golden/1/18c65a6f996008b7bb44ccb5369e862df5868a1d13b15165fe80bd8f9edc7415/torch_internvl3-8b_golden.tar.gz",
                json_file="torch_internvl3-8b_bfloat16_golden.json",
            ),
            # TODO(MODELS-565): Fix InternVL correctness.
            cos_dist_threshold=3.5e-1,
            kl_div_threshold=7.0e-01,
        ),
    ),
    "OpenGVLab/InternVL3-14B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-multi"],
        run=_make_pipeline_runner(
            pipeline="OpenGVLab/InternVL3-14B-Instruct",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_internvl3-14b_golden/1/b0b6be883a40a1f44d85abbf077edec5c768fee8302f0595147ce7197da8560f/torch_internvl3-14b_golden.tar.gz",
                json_file="torch_internvl3-14b_bfloat16_golden.json",
            ),
            absolute_tolerance=1.0e-04,
            relative_tolerance=2.0e00,
            cos_dist_threshold=1.0e-03,
            kl_div_threshold=5.0e-02,
        ),
    ),
    "OpenGVLab/InternVL3-38B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-multi"],
        run=_make_pipeline_runner(
            pipeline="OpenGVLab/InternVL3-38B-Instruct",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_internvl3-38b_golden/1/68ce8e1ca017a4b42b1a9f270345dd64e869c57c582d5ade12eb654a42893b21/torch_internvl3-38b_golden.tar.gz",
                json_file="torch_internvl3-38b_bfloat16_golden.json",
            ),
            cos_dist_threshold=5.5e-03,
            kl_div_threshold=4.8e-02,
            timeout=900,
        ),
    ),
    "OpenGVLab/InternVL3_5-8B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-multi"],
        run=_make_pipeline_runner(
            pipeline="OpenGVLab/InternVL3_5-8B-Instruct",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_internvl3_5_8b_golden/1/c1e88d08b863bc4e5ad250bf91bbe9e6a371f1244f96dd37befb8c1a086afcea/torch_internvl3_5_8b_golden.tar.gz",
                json_file="torch_InternVL3_5_8B_bfloat16_golden.json",
            ),
            cos_dist_threshold=1.2e-2,
            kl_div_threshold=1.6e-02,
            timeout=1200,
        ),
    ),
    "mistral-community/pixtral-12b-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big"],
        run=_make_pipeline_runner(
            pipeline="mistral-community/pixtral-12b",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_pixtral_golden/1/e2ec8c3693bf758df21d5673a35847df88307fb6568a851be531c53e6b18f710/torch_pixtral_golden.tar.gz",
                json_file="torch_pixtral_bfloat16_golden.json",
            ),
            cos_dist_threshold=7.2e-3,
            kl_div_threshold=2.0e-2,
        ),
    ),
    "Qwen/Qwen2.5-7B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-only"],  # TODO: Has much worse accuracy on AMD GPUs.
        run=_make_pipeline_runner(
            pipeline="Qwen/Qwen2.5-7B-Instruct",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_qwen2.5-7b-instruct_bfloat16_golden/1/a18b2a5b0e4a63ba828fa956ffeed04feaf5501721dae0421670c94e93f445a0/torch_qwen2.5-7b-instruct_bfloat16_golden.tar.gz",
                json_file="torch_qwen2.5_7b_instruct_bfloat16_golden.json",
            ),
            cos_dist_threshold=5.0e-2,
            kl_div_threshold=3.5e-1,
        ),
    ),
    "Qwen/Qwen2.5VL-3B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=[],  # TODO(MODELS-803) Errors on 4x GPU
        run=_make_pipeline_runner(
            pipeline="Qwen/Qwen2.5-VL-3B-Instruct",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_qwen2.5-vl-3b-instruct_bfloat16_golden/1/40a5c716c43ae003113b7a38eff4121619d4ca58bdf4d8b6f6b362529279ba5d/torch_qwen2.5-vl-3b-instruct_bfloat16_golden.tar.gz",
                json_file="torch_qwen2.5_vl_3b_instruct_bfloat16_golden.json",
            ),
            cos_dist_threshold=1.9e00,
            kl_div_threshold=1.5e01,
        ),
    ),
    "Qwen/Qwen2.5VL-7B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-multi"],
        run=_make_pipeline_runner(
            pipeline="Qwen/Qwen2.5-VL-7B-Instruct",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_qwen2.5-vl-7b-instruct_bfloat16_golden/1/6c5aaac177ce9a8428b972667383621d2bb853a1ec3c44fdd63ca00661fff4e0/torch_qwen2.5-vl-7b-instruct_bfloat16_golden.tar.gz",
                json_file="torch_qwen2.5_vl_7b_instruct_bfloat16_golden.json",
            ),
            cos_dist_threshold=3.5e-1,
            kl_div_threshold=4.0e-1,
        ),
    ),
    "Qwen/Qwen2.5VL-32B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-multi"],
        run=_make_pipeline_runner(
            pipeline="Qwen/Qwen2.5-VL-32B-Instruct",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_qwen2.5-vl-32b-instruct_bfloat16_golden/1/62db96f972e210adb18a6042441b49feb5a163bdb6f5592e8788dbb9f887dbe8/torch_qwen2.5-vl-32b-instruct_bfloat16_golden.tar.gz",
                json_file="torch_qwen2.5_vl_32b_instruct_bfloat16_golden.json",
            ),
            cos_dist_threshold=7.0e-2,
            kl_div_threshold=2.6e-1,
            timeout=900,
        ),
    ),
    "Qwen/Qwen3-VL-30B-A3B-Instruct": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-multi"],
        run=_make_pipeline_runner(
            pipeline="Qwen/Qwen3-VL-30B-A3B-Instruct",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_qwen3-vl-30b-a3b-instruct_bfloat16_golden/1/43b1500269067c2e25bed53fefc98c3973e2132fabdfb6923acbedc531636bc0/torch_qwen3-vl-30b-a3b-instruct_bfloat16_golden.tar.gz",
                json_file="torch_qwen3_vl_30b_a3b_instruct_bfloat16_golden.json",
            ),
            cos_dist_threshold=1.7e00,
            kl_div_threshold=2.1e01,
        ),
    ),
    "Qwen/Qwen3-VL-4B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=[],
        run=_make_pipeline_runner(
            pipeline="Qwen/Qwen3-VL-4B-Instruct",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_qwen3-vl-4b-instruct_bfloat16_golden/1/7c5d43eeab21e1478b79fc849a537766809b1c3aabb7f253309dc47895d1de39/torch_qwen3-vl-4b-instruct_bfloat16_golden.tar.gz",
                json_file="torch_qwen3_vl_4b_instruct_bfloat16_golden.json",
            ),
            cos_dist_threshold=1.7e00,
            kl_div_threshold=4.4e-01,
        ),
    ),
    "Qwen/Qwen3-VL-4B-Instruct-FP8": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=[],
        run=_make_pipeline_runner(
            pipeline="Qwen/Qwen3-VL-4B-Instruct-FP8",
            encoding="float8_e4m3fn",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_qwen3-vl-4b-instruct_fp8_golden/1/f1060a92f10622b249b900f03c64a81fe44111141b704b17125063e9da350dd3/torch_qwen3-vl-4b-instruct_fp8_golden.tar.gz",
                json_file="torch_qwen3_vl_4b_instruct_fp8_golden.json",
            ),
            cos_dist_threshold=1.7e00,
            kl_div_threshold=3.6e-01,
        ),
    ),
    "Qwen/Qwen3-8B-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big", "nvidia-only"],  # TODO: Attention is broken on AMD.
        run=_make_pipeline_runner(
            pipeline="Qwen/Qwen3-8B",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_qwen3_8b_golden/1/a952c1f1cae7dd647abbb01db7232b73d2cd084a561a7da5d3027e95acbeaf71/torch_qwen3_8b_golden.tar.gz",
                json_file="torch_qwen3_8b_golden.json",
            ),
            cos_dist_threshold=1.1e-3,
            kl_div_threshold=7.1e-3,
        ),
    ),
    "Qwen/Qwen3-30B-A3B-Instruct-2507-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big", "nvidia-only", "no-h100"],
        run=_make_pipeline_runner(
            pipeline="Qwen/Qwen3-30B-A3B-Instruct-2507",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_qwen3-30b-a3b-instruct-2507_bfloat16_golden/1/4427d3205be77073d7f0274d8fca0774974fe8677934a5b8dc60b3705aac9aaf/torch_qwen3-30b-a3b-instruct-2507_bfloat16_golden.tar.gz",
                json_file="torch_qwen3_30b_a3b_instruct_2507_bfloat16_golden.json",
            ),
            cos_dist_threshold=7.0e-02,
            kl_div_threshold=8.0e-01,
        ),
    ),
    "Qwen/Qwen3-Embedding-0.6B-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        run=_make_pipeline_runner(
            pipeline="Qwen/Qwen3-Embedding-0.6B",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_qwen3_embedding_0.6b_bfloat16_golden/1/98f263166b17cbc486f6709903c35ab1126f938efc0ea07289abd0419d6b9b48/torch_qwen3_embedding_0.6b_bfloat16_golden.tar.gz",
                json_file="torch_qwen3_embedding_0.6b_bfloat16_golden.json",
            ),
            relative_tolerance=1.0e-04,
            absolute_tolerance=4.2e-01,
            cos_dist_threshold=2.4e-1,
            kl_div_threshold=2.6e-04,
        ),
    ),
    # Qwen2.VL-FP8
    "allenai/olmOCR-2-7B-1025-FP8": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-only"],  # TODO(KERN-2196)
        run=_make_pipeline_runner(
            pipeline="allenai/olmOCR-2-7B-1025-FP8",
            encoding="float8_e4m3fn",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_olmocr_2_7b_fp8_golden/1/7177335f99779ffdf6dd516a1c6082d75ce3d26b9a7dbe06ec88ebbd729a090e/torch_olmocr_2_7b_fp8_golden.tar.gz",
                json_file="olmocr_2_7b_fp8_torch_goldens.json",
            ),
            cos_dist_threshold=2.4e-01,
            kl_div_threshold=8.8e-01,
        ),
    ),
    "allenai/OLMo-2-1124-7B-float32": PipelineDef(
        compatible_with=[DeviceKind.CPU, DeviceKind.GPU],
        tags=["big"],
        run=_make_pipeline_runner(
            pipeline="allenai/OLMo-2-1124-7B",
            encoding="float32",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_olmo2_1124_7b_float32_golden/1/65d0d4860591ae05c7ae3cf062bc0bb5429733db08b49635f3deeeb7369eb122/torch_olmo2_1124_7b_float32_golden.tar.gz",
                json_file="olmo2_1124_7b_float32_torch_goldens.json",
            ),
            cos_dist_threshold=2.1e-5,
            kl_div_threshold=4.6e-7,
        ),
    ),
    "HuggingFaceM4/Idefics3-8B-Llama3": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big", "nvidia-only"],
        run=_make_pipeline_runner(
            pipeline="HuggingFaceM4/Idefics3-8B-Llama3",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_idefics3_8b_llama3_golden/1/c50d33d355c5594d57336cd9b2b46fbb27b3f8a06f58d72e1078cc5c3fb6ffde/torch_idefics3_8b_llama3_golden.tar.gz",
                json_file="idefics3_8b_llama3_torch_goldens.json",
            ),
            # TODO: Accuracy is much worse on AMD.
            # so we might have an AMD kernel bug here
            # TODO(MODELS-730): With the update to transformers=4.55, the
            # kl_div_threshold went from 8.7e-02 to 6.6e-01.
            # This is likely due to changes in the reference implementation.
            cos_dist_threshold=5.0e-02,
            kl_div_threshold=6.8e-01,
        ),
    ),
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-float32": PipelineDef(
        compatible_with=[DeviceKind.CPU, DeviceKind.GPU],
        tags=["big"],
        run=_make_pipeline_runner(
            pipeline="LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
            encoding="float32",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_exaone_3_5_2_4b_instruct_float32_golden/1/ae16f10ebd7514f5d60f25755babbcff9636a4fa1c68d1470f614474fa907427/torch_exaone_3_5_2_4b_instruct_float32_golden.tar.gz",
                json_file="exaone_3_5_2_4b_instruct_float32_torch_goldens.json",
            ),
            # TODO: Accuracy is much better on AMD.
            # so we might have an nvidia kernel bug here
            cos_dist_threshold=2.5e-2,
            kl_div_threshold=1.3e-2,
        ),
    ),
    "microsoft/Phi-3.5-mini-instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        run=_make_pipeline_runner(
            pipeline="microsoft/Phi-3.5-mini-instruct",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_phi_3_5_mini_instruct_bfloat16_golden/1/23beeadd54f6b299463a0bb5c8d2bdbe010ba27dabb686f942d33e56ec4d0870/torch_phi_3_5_mini_instruct_bfloat16_golden.tar.gz",
                json_file="phi_3_5_mini_instruct_bfloat16_torch_goldens.json",
            ),
            # TODO(MODELS-458): This model seems broken based on the thresholds
            cos_dist_threshold=1.6e-2,
            kl_div_threshold=4.0e-1,
        ),
    ),
    "microsoft/phi-4-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big"],
        run=_make_pipeline_runner(
            pipeline="microsoft/phi-4",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_phi_4_bfloat16_golden/0/8634be2cd675ea0dbb8a2ce84692572dd003505fa86edd91f0d8b0659d09b6f0/torch_phi_4_bfloat16_golden.tar.gz",
                json_file="phi_4_bfloat16_torch_goldens.json",
            ),
            absolute_tolerance=1.0e-04,
            relative_tolerance=2.0e00,
            cos_dist_threshold=1.5e-3,
            kl_div_threshold=1.3e-02,
        ),
    ),
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4-gptq": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-only"],
        run=_make_pipeline_runner(
            pipeline="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
            encoding="gptq",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_llama-gptq_golden/0/7e5b7b4d1764033be69e85e0badc9dca82c94c8d2def1216d317b149a621daef/torch_llama-gptq_golden.tar.gz",
                json_file="torch_llama-gptq_golden.json",
            ),
            cos_dist_threshold=1e-3,
            kl_div_threshold=2.7e-3,
        ),
    ),
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4-gptq-no-perm-idx": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-only"],
        run=_make_pipeline_runner(
            pipeline="kaitchup/DeepSeek-R1-Distill-Llama-8B-AutoRound-GPTQ-4bit",
            encoding="gptq",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_llama-gptq_golden/0/7e5b7b4d1764033be69e85e0badc9dca82c94c8d2def1216d317b149a621daef/torch_llama-gptq_golden.tar.gz",
                json_file="torch_llama-gptq-no-perm-idx_golden.json",
            ),
            absolute_tolerance=1.0e-04,
            relative_tolerance=2.0e00,
            cos_dist_threshold=6.4e-04,
            kl_div_threshold=5.5e-03,
        ),
    ),
    # TODO(AITLIB-372): investigate why accuracy tanked when switching to explicit weight dtype casting.
    # TODO(SERVOPT-571): Re-enable after fixing.
    # "deepseek-ai/DeepSeek-V2-Lite-Chat-bfloat16": PipelineDef(
    #     compatible_with=[DeviceKind.GPU],
    #     tags=["big", "nvidia-only"],
    #     run=_make_pipeline_runner(
    #         pipeline="deepseek-ai/DeepSeek-V2-Lite-Chat",
    #         encoding="bfloat16",
    #         # TODO(MODELS-516): Investigate need for high tolerances here.
    #         # TODO(GENAI-216): Investigate non-deterministic output.
    #         cos_dist_threshold=4.1e-03,
    #         kl_div_threshold=2.6e-01,
    #     ),
    # ),
    # TODO(MODELS-812): Investigate deepseek timeout
    "kathywu95/deepseek-v3-small-random-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-only", "no-b200"],  # Times out on B200
        run=_make_pipeline_runner(
            pipeline="kathywu95/deepseek-v3-small-random",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_deepseek-v3-small-random-bfloat16_golden/1/a014fa9dfbadae61869bb51da13e6e5c9fa6294a03b18b49c9217da6024a13eb/torch_deepseek-v3-small-random-bfloat16_golden.tar.gz",
                json_file="deepseek-v3-small-random_torch_goldens.json",
            ),
            cos_dist_threshold=2.9e-02,
            kl_div_threshold=8.0e-2,  # TODO(MODELS-998)
        ),
    ),
    "kathywu95/deepseek-v3-small-random-fp8": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-only", "no-h100"],  # B200 only
        run=_make_pipeline_runner(
            pipeline="kathywu95/deepseek-v3-small-random-fp8",
            encoding="float8_e4m3fn",
            # Goldens generated using VLLM.
            # Script: https://gist.github.com/k-w-w/420b2d64283e83c1121f89d35027a1d6
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_deepseek-v3-small-random-fp8_golden/1/29b77d635f3e2cb5f3b8df155174ce5a77f5d6d1a074a4283fddcaf090906cc8/torch_deepseek-v3-small-random-fp8_golden.tar.gz",
                json_file="torch_deepseek-v3-small-random_bfloat16_golden.json",
            ),
            cos_dist_threshold=2.7e-03,
            kl_div_threshold=1.1e-2,
        ),
    ),
    "deepseek-ai/DeepSeek-R1": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-multi", "8xb200"],  # Requires 8 B200s to run
        run=_make_pipeline_runner(
            pipeline="deepseek-ai/DeepSeek-R1",
            encoding="float8_e4m3fn",
            # Goldens generated using VLLM.
            # Script: https://gist.github.com/k-w-w/1dc387dc41f11789e464d4a9267a8d20
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/vllm_deepseek-r1_golden/1/f4b3ce07362060a857724d8721aa008880b2f1da3a9f90aec667672c92f7e5e9/vllm_deepseek-r1_golden.tar.gz",
                json_file="vllm_deepseek-r1_float8_golden.json",
            ),
            cos_dist_threshold=8.8e-03,
            kl_div_threshold=1.6e-1,
            timeout=1200,
        ),
    ),
    "nvidia/DeepSeek-R1-0528-NVFP4-v2": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-multi", "8xb200"],  # Requires 8 B200s to run
        run=_make_pipeline_runner(
            pipeline="nvidia/DeepSeek-R1-0528-NVFP4-v2",
            encoding="float4_e2m1fnx2",
            # Goldens generated using vLLM with NVFP4 support
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/vllm_deepseek-r1-nvfp4_golden/9/9b19a48a9bba02fe76bda80402950c1ae13c5e0f93444b08c2c6499f4b3247e7/vllm_deepseek-r1-nvfp4_golden.tar.gz",
                json_file="vllm_deepseek-r1-nvfp4_float4_golden.json",
            ),
            # Tolerances from running --find-tolerances against vLLM goldens
            cos_dist_threshold=2.7e-02,
            kl_div_threshold=2.1e-01,
            timeout=1200,
        ),
    ),
    "google/gemma-3-1b-it-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        run=_make_pipeline_runner(
            pipeline="google/gemma-3-1b-it",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_gemma3-1b_golden/1/31d4f0ff8f50b9ab0f877d8765114f6bc4ae73677d2cd2d6ce658866fabf15d4/torch_gemma3-1b_golden.tar.gz",
                json_file="torch_gemma3-1b_bfloat16_golden.json",
            ),
            cos_dist_threshold=1.3e-3,
            kl_div_threshold=6.0e-02,
        ),
    ),
    "google/gemma-3-12b-it-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big"],
        run=_make_pipeline_runner(
            pipeline="google/gemma-3-12b-it",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_gemma3-multimodal_golden/1/06d0fa8ed540ae7141a42c432af1661c85d31f8584d017345992df7a52c21ccb/torch_gemma3-multimodal_golden.tar.gz",
                json_file="torch_gemma3-multimodal_bfloat16_golden.json",
            ),
            absolute_tolerance=1.0e-04,
            relative_tolerance=2.0,
            cos_dist_threshold=3.0e-02,
            kl_div_threshold=0.35,
        ),
    ),
    "google/gemma-3-27b-it-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-multi"],
        run=_make_pipeline_runner(
            pipeline="google/gemma-3-27b-it",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_gemma3-27b_golden/0/d4747c90804cbfb6ee4ee06ec15c042dd436558354cfae819e0203d1c3610b38/torch_gemma3-27b_golden.tar.gz",
                json_file="torch_gemma3-27b_bfloat16_golden.json",
            ),
            cos_dist_threshold=1.9e-02,
            kl_div_threshold=6.9e-01,
        ),
    ),
    "RedHatAI/gemma-3-27b-it-FP8-dynamic": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big", "float8-support"],
        run=_make_pipeline_runner(
            pipeline="RedHatAI/gemma-3-27b-it-FP8-dynamic",
            encoding="float8_e4m3fn",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/vllm_gemma3-27b_golden/1/1a619d49187cdce335f4492acab40fd950922748e6631c0478572344ff295efc/vllm_gemma3-27b_golden.tar.gz",
                json_file="vllm_gemma3-27b_float8-dynamic_golden.json",
            ),
            cos_dist_threshold=3.6e-2,
            kl_div_threshold=1.4e0,
        ),
    ),
    # Multi-GPU variant
    "RedHatAI/gemma-3-27b-it-FP8-dynamic-multi": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big", "float8-support", "nvidia-multi"],
        run=_make_pipeline_runner(
            pipeline="RedHatAI/gemma-3-27b-it-FP8-dynamic",
            encoding="float8_e4m3fn",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/vllm_gemma3-27b_golden/1/1a619d49187cdce335f4492acab40fd950922748e6631c0478572344ff295efc/vllm_gemma3-27b_golden.tar.gz",
                json_file="vllm_gemma3-27b_float8-dynamic_golden.json",
            ),
            cos_dist_threshold=2.3e-2,
            kl_div_threshold=7.0e-1,
        ),
    ),
    "HKUSTAudio/Llasa-8B-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big", "tts"],  # TTS tag to identify text-to-speech models
        run=_make_pipeline_runner(
            pipeline="HKUSTAudio/Llasa-8B",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_llasa-8b_bfloat16_golden/1/301c70b6c564f667ce279151964878354fe58d070f4f31869ecd6e2139172fc1/torch_llasa-8b_bfloat16_golden.tar.gz",
                json_file="HKUSTAudio_Llasa-8B_torch_goldens.json",
            ),
            cos_dist_threshold=1.5e-02,
            kl_div_threshold=7.7e-01,
        ),
    ),
    "HuggingFaceTB/SmolLM2-360M-Instruct-LoRA-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        # TODO(E2EOPT-698)
        # TODO(MODELS-885): Thresholds are 'inf', and/or non-determinism
        tags=["nvidia-only"],  # Small model (<8B params)
        run=_make_pipeline_runner(
            pipeline="HuggingFaceTB/SmolLM2-360M-Instruct",
            encoding="bfloat16",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/torch_smollm2_360m_instruct_lora_bfloat16_golden/1/1149947546122f5eef59073a8b06a0fe525051eb6003d2891510bdff8e49be6e/torch_smollm2_360m_instruct_lora_bfloat16_golden.tar.gz",
                json_file="HuggingFaceTB_SmolLM2-360M-Instruct_torch_goldens.json",
            ),
            cos_dist_threshold=1e3,
            kl_div_threshold=1e3,
        ),
    ),
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic-BF16-LoRA": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-only", "no-h100", "float8-support"],
        run=_make_pipeline_runner(
            pipeline="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic-BF16-LoRA",
            encoding="float8_e4m3fn",
            pregenerated_torch_goldens=PregeneratedTorchGoldens(
                tar_file="s3://modular-bazel-artifacts-public/artifacts/vllm_llama_3_1_8B_fp8_bf16_lora/1/6db6cad8339db70f2975e9a610d79a8a57ba9b8c43a949d8008b95a0faf22f28/vllm_llama_3_1_8B_fp8_bf16_lora.tar.gz",
                json_file="vllm_llama3_1_8B_float8_dyanmic_bf16_lora_golden.json",
            ),
            cos_dist_threshold=1.41e-01,
            kl_div_threshold=7.1e-01,
        ),
    ),
}


@click.command()
@click.option(
    "--report",
    type=click.File("w"),
    help="Output the coverage report to the specified file",
)
@click.option(
    "--store-verdicts-json",
    type=click.Path(path_type=Path),
    help="Store verdicts in JSON format to the specified file",
)
@click.option(
    "--load-verdicts-json",
    type=click.Path(path_type=Path),
    help="Load previous verdicts from JSON file to compare changes",
)
@click.option("--devices", "devices_str", help="Devices to run pipeline on")
@click.option("--pipeline", help="Run only a specified pipeline")
@click.option(
    "--tags",
    "tag_filter",
    type=TagFilterParamType(),
    help="Tags to filter to (+) or exclude (-), comma-separated",
    default=TagFilter(),
)
@click.option(
    "--find-tolerances",
    is_flag=True,
    default=False,
    help=(
        "Set all tolerances to a lower bound and enables `--print-suggested-tolerances`."
        " This leads to automatically searching for the suggested tolerances."
    ),
)
@click.option(
    "--print-suggested-tolerances",
    is_flag=True,
    default=False,
    help=(
        "On failure, prints a set of potential tolerances based on the pareto"
        " frontier of passing absolute and relative tolerance combinations."
    ),
)
def main(
    report: TextIO | None,
    store_verdicts_json: Path | None,
    load_verdicts_json: Path | None,
    devices_str: str | None,
    pipeline: str | None,
    tag_filter: TagFilter,
    find_tolerances: bool,
    print_suggested_tolerances: bool,
) -> None:
    """Run logit-level comparisons of a Modular pipeline against a reference."""

    # Let generate_llm_logits.py validate the `--devices` CLI arg and just pass
    # it through as a string (but use it here to figure out cpu vs. gpu).
    device_type = (
        DeviceKind.CPU
        if isinstance(devices_str, str) and "cpu" in devices_str
        else DeviceKind.GPU
    )
    devices_str = "cpu" if devices_str is None else devices_str

    verdicts: dict[str, VerificationVerdict] = {}
    if pipeline is None:
        for pipeline_name, pipeline_def in PIPELINES.items():
            if device_type not in pipeline_def.compatible_with:
                continue
            if not tag_filter.satisfied_by(pipeline_def.tags):
                continue
            start_time = time.time()
            print(f"\n===== Running {pipeline_name} =====", flush=True)
            verdicts[pipeline_name] = pipeline_def.run_protected(
                device_type,
                devices_str,
                find_tolerances,
                print_suggested_tolerances,
            )
            duration = f"{time.time() - start_time:.0f}s"
            print(
                f"\n===== Finished {pipeline_name} ({duration}) =====",
                flush=True,
            )
    else:
        # TODO: Temporarily allow to not specify the org name when running a
        # pipeline by name. This is because the bisection script does not
        # currently have access to the org name. Fix this by making the
        # bisection use the existing json status report.
        for pipeline_name, pipeline_def in PIPELINES.items():  # noqa: B007
            if display_name(pipeline_name) == pipeline:
                pipeline = pipeline_name
                break
        if pipeline not in PIPELINES:
            raise click.ClickException(f"Unknown pipeline {pipeline!r}")
        pipeline_def = PIPELINES[pipeline]
        if device_type not in pipeline_def.compatible_with:
            raise click.ClickException(
                f"Pipeline {pipeline!r} not compatible with {device_type!r}"
            )
        if not tag_filter.satisfied_by(pipeline_def.tags):
            raise click.ClickException(
                f"Pipeline {pipeline!r} doesn't match tag filter {tag_filter}"
            )
        verdicts[pipeline] = pipeline_def.run_protected(
            device_type,
            devices_str,
            find_tolerances,
            print_suggested_tolerances,
        )

    # Load previous verdicts if provided
    previous_verdicts = None
    if load_verdicts_json:
        previous_verdicts = load_verdicts_from_json(load_verdicts_json)

    if report:
        dump_results(verdicts, to=report, previous_verdicts=previous_verdicts)

    if store_verdicts_json:
        save_verdicts_to_json(verdicts, store_verdicts_json)

    print()
    print("-" * 40)
    print()
    print("# pipelines run:", len(verdicts))
    for status in list(VerificationStatus):
        print(
            f"# pipelines {status.name}:",
            sum(v.status == status for v in verdicts.values()),
        )
    print()
    dump_results(verdicts, previous_verdicts=previous_verdicts)

    if any(v.status != VerificationStatus.OK for v in verdicts.values()):
        if all(
            v.status in (VerificationStatus.OK, VerificationStatus.FLAKE)
            for v in verdicts.values()
        ):
            # If every failure was a flake, propagate the EX_TEMPFAIL status code onward.
            sys.exit(EX_TEMPFAIL)
        sys.exit(1)


if __name__ == "__main__":
    validate_hf_token()

    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    main()
