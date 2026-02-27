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
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import click
from generate_llm_logits import Flake, generate_llm_logits
from max.pipelines.lib.device_specs import (
    device_specs_from_normalized_device_handle,
    normalize_device_specs_input,
)
from max.tests.integration.accuracy.logit_verification.logit_verification_config import (
    LOGIT_VERIFICATION_CONFIG,
    DeviceKind,
    PipelineConfig,
    SupportedEncoding,
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

# Encodings that cannot generate torch reference goldens locally.
# These require pregenerated goldens from S3 (e.g. generated via vLLM).
# When --no-aws is set, pipelines with these encodings are skipped.
TORCH_INCOMPATIBLE_ENCODINGS = frozenset({"float8_e4m3fn", "float4_e2m1fnx2"})


def validate_hf_token() -> None:
    """
    It's a regular occurrence that people are asked to run logit verification
    locally, and not everyone has an HF_TOKEN set. Let's help them out
    """
    if os.getenv("HF_HUB_OFFLINE", "").lower() in ("1", "t", "true"):
        return
    if os.getenv("HF_TOKEN") is None:
        raise ValueError(
            "Environment variable `HF_TOKEN` must be set. "
            "See https://www.notion.so/modularai/HuggingFace-Access-Token-29d1044d37bb809fbe70e37428faf9da"
        )


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

    any_logit, any_embedding, any_image, any_failed = False, False, False, False
    for verdict in verdicts.values():
        if verdict.discrepancy_report is None:
            any_failed = True
        elif verdict.discrepancy_report.model_modality == Modality.LOGIT:
            any_logit = True
        elif verdict.discrepancy_report.model_modality == Modality.EMBEDDING:
            any_embedding = True
        elif verdict.discrepancy_report.model_modality == Modality.IMAGE:
            any_image = True

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

    if any_image:
        to.write("\n\n## Image Models\n")
        to.write(
            "**SSIM (avg)** = average SSIM over all prompts (higher is better)\n"
            "**LPIPS (avg)** = average LPIPS over all prompts (lower is better)\n"
            "**Diff (LPIPS avg)** = change of average LPIPS from previous run\n"
        )
        to.write(
            "| Status | Model | SSIM (avg) | LPIPS (avg) | MAE | Diff (LPIPS avg) |\n"
        )
        to.write(
            "| :----: | :---  | :--------: | :---------: |:---:| :--------------: |\n"
        )

        for name, verdict in sorted(verdicts.items(), key=verdict_sorting_key):
            if verdict.discrepancy_report is None:
                continue
            if verdict.discrepancy_report.model_modality != Modality.IMAGE:
                continue

            ssim_avg = verdict.discrepancy_report.avg_ssim
            lpips_avg = verdict.discrepancy_report.avg_lpips
            ssim_str = "N/A" if ssim_avg is None else f"{ssim_avg:.2e}"
            lpips_str = "N/A" if lpips_avg is None else f"{lpips_avg:.2e}"
            mae = f"{verdict.discrepancy_report.avg_mae:.2e}"

            diff_str = "N/A"
            if previous_verdicts and name in previous_verdicts:
                diff_str = compute_diff(verdict, previous_verdicts[name])

            to.write(
                f"| {verdict.emoji} | {display_name(name)} | {ssim_str} | {lpips_str} | {mae} | {diff_str} |\n"
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
    encoding: SupportedEncoding,
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


def _run_llm_verification(
    config: PipelineConfig,
    *,
    device_type: DeviceKind,
    devices: str,
    find_tolerances: bool,
    print_suggested_tolerances: bool,
) -> VerificationVerdict:
    """Run verification with the given model and weights encoding."""

    encoding = config.encoding
    fssafe_pipeline = config.pipeline.replace("/", "_")

    # Run the torch baseline or load it from golden.
    if config.pregenerated_torch_goldens is not None:
        # This workflow runs on an A10. The Torch reference runs out of memory
        # on an A10, so it was run manually on an A100 and the result goldens
        # uploaded. Use these pre-generated goldens in this case.
        tar_file = load_from_tar(config.pregenerated_torch_goldens.tar_file)
        torch_golden_path = Path(
            tar_file, config.pregenerated_torch_goldens.json_file
        )
    else:
        torch_golden_path = Path(
            f"/tmp/goldens_torch_{device_type.value}_{fssafe_pipeline}_{encoding}.json"
        )
        generate_llm_logits_with_optional_retry(
            framework="torch",
            device=devices,
            pipeline=config.pipeline,
            encoding=encoding,
            output_path=torch_golden_path,
            timeout=config.timeout,
        )

    torch_results: list[ModelOutput] = NumpyDecoder().decode(
        torch_golden_path.read_text()
    )

    absolute_tolerance = config.absolute_tolerance
    relative_tolerance = config.relative_tolerance
    cos_dist_threshold = config.cos_dist_threshold
    kl_div_threshold = config.kl_div_threshold

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
        pipeline=config.pipeline,
        encoding=encoding,
        output_path=max_golden_path,
        reference=torch_results,
        timeout=config.timeout,
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


def run_llm_verification(
    config: PipelineConfig,
    *,
    device_type: DeviceKind,
    devices: str,
    find_tolerances: bool,
    print_suggested_tolerances: bool,
) -> VerificationVerdict:
    """Run LLM verification with error handling for flakes and infra issues."""
    try:
        with detect_infra_errors():
            return _run_llm_verification(
                config,
                device_type=device_type,
                devices=devices,
                find_tolerances=find_tolerances,
                print_suggested_tolerances=print_suggested_tolerances,
            )
    except Flake:
        return VerificationVerdict(status=VerificationStatus.FLAKE)
    except InfraError:
        traceback.print_exc()
        return VerificationVerdict(status=VerificationStatus.INFRA)
    except Exception:
        traceback.print_exc()
        return VerificationVerdict(status=VerificationStatus.ERROR)


def _run_pixel_generation_verification(
    config: PipelineConfig,
    *,
    device_type: DeviceKind,
    devices: str,
    find_tolerances: bool,
    print_suggested_tolerances: bool,
) -> VerificationVerdict:
    """Run pixel generation verification with the given model and weights encoding.

    This verification uses multiple metrics for comprehensive image quality assessment:
    - MAE/RMSE for pixel-level accuracy
    - SSIM for structural similarity
    - LPIPS for learned perceptual similarity
    """

    encoding = config.encoding
    fssafe_pipeline = config.pipeline.replace("/", "_")

    # Run the torch baseline or load it from golden
    if config.pregenerated_torch_goldens is not None:
        tar_file = load_from_tar(config.pregenerated_torch_goldens.tar_file)
        torch_golden_path = Path(
            tar_file, config.pregenerated_torch_goldens.json_file
        )
    else:
        torch_golden_path = Path(
            f"/tmp/goldens_torch_{device_type.value}_{fssafe_pipeline}_{encoding}.json"
        )
        generate_llm_logits_with_optional_retry(
            framework="torch",
            device=devices,
            pipeline=config.pipeline,
            encoding=encoding,
            output_path=torch_golden_path,
            timeout=config.timeout,
        )

    torch_results: list[ModelOutput] = NumpyDecoder().decode(
        torch_golden_path.read_text()
    )

    absolute_tolerance = config.absolute_tolerance
    relative_tolerance = config.relative_tolerance
    ssim_threshold = config.ssim_threshold
    lpips_threshold = config.lpips_threshold

    # When find_tolerances is enabled, we set all tolerances to a lower bound and enable print_suggested_tolerances
    if find_tolerances:
        print_suggested_tolerances = True
        ssim_threshold = 0.999  # Very high SSIM threshold to find minimum
        absolute_tolerance = 1e-4
        relative_tolerance = 1e-4

    max_golden_path = Path(
        f"/tmp/goldens_max_{device_type.value}_{fssafe_pipeline}_{encoding}.json"
    )
    generate_llm_logits_with_optional_retry(
        framework="max",
        device=devices,
        pipeline=config.pipeline,
        encoding=encoding,
        output_path=max_golden_path,
        reference=torch_results,
        timeout=config.timeout,
    )

    eval_metrics = []
    if absolute_tolerance is not None and relative_tolerance is not None:
        eval_metrics.append("tol")
    if ssim_threshold is not None:
        eval_metrics.append("ssim")
    if lpips_threshold is not None:
        eval_metrics.append("lpips")
    if not eval_metrics:
        raise ValueError(
            "Please provide absolute, relative, SSIM or LPIPS thresholds."
            " Otherwise no metrics will be computed."
        )

    try:
        result = verify(
            pipeline_outputs=max_golden_path,
            torch_outputs=torch_golden_path,
            eval_metric=eval_metrics,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            ssim_threshold=ssim_threshold,
            lpips_threshold=lpips_threshold,
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
            kl_div_threshold=None,  # Not applicable for images
        )
    except Exception:
        traceback.print_exc()
        return VerificationVerdict(status=VerificationStatus.ERROR)


def run_pixel_generation_verification(
    config: PipelineConfig,
    *,
    device_type: DeviceKind,
    devices: str,
    find_tolerances: bool,
    print_suggested_tolerances: bool,
) -> VerificationVerdict:
    """Run pixel generation verification with error handling for flakes and infra issues."""
    try:
        with detect_infra_errors():
            return _run_pixel_generation_verification(
                config,
                device_type=device_type,
                devices=devices,
                find_tolerances=find_tolerances,
                print_suggested_tolerances=print_suggested_tolerances,
            )
    except Flake:
        return VerificationVerdict(status=VerificationStatus.FLAKE)
    except InfraError:
        traceback.print_exc()
        return VerificationVerdict(status=VerificationStatus.INFRA)
    except Exception:
        traceback.print_exc()
        return VerificationVerdict(status=VerificationStatus.ERROR)


def _is_pixel_generation(config: PipelineConfig) -> bool:
    """Determines if a pipeline config is for pixel/image generation."""
    return (
        config.ssim_threshold is not None or config.lpips_threshold is not None
    )


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
@click.option(
    "--pipeline",
    "pipelines",
    type=click.Choice(list(LOGIT_VERIFICATION_CONFIG.pipelines.keys())),
    multiple=True,
    default=list(LOGIT_VERIFICATION_CONFIG.pipelines.keys()),
    show_default=False,
    help="Pipelines to run (repeatable). Defaults to all.",
)
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
@click.option(
    "--filter",
    "name_filter",
    type=str,
    default=None,
    help="Only run pipelines whose name matches the filter. Comma-separated for multiple filters (OR logic).",
)
@click.option(
    "--no-aws",
    is_flag=True,
    default=False,
    help=(
        "Run without AWS access. Ignores pregenerated torch goldens and"
        " generates reference outputs locally using torch instead."
        " Works for bf16/f32 models but will fail for FP8 models that"
        " require vLLM-generated goldens."
    ),
)
def main(
    report: TextIO | None,
    store_verdicts_json: Path | None,
    load_verdicts_json: Path | None,
    devices_str: str | None,
    pipelines: tuple[str, ...],
    tag_filter: TagFilter,
    find_tolerances: bool,
    print_suggested_tolerances: bool,
    name_filter: str | None,
    no_aws: bool,
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
    for pipeline_name in pipelines:
        pipeline_config = LOGIT_VERIFICATION_CONFIG.pipelines[pipeline_name]
        if device_type not in pipeline_config.compatible_with:
            continue
        if not tag_filter.satisfied_by(pipeline_config.tags):
            continue
        if name_filter and not any(
            f.strip().casefold() in pipeline_name.casefold()
            for f in name_filter.split(",")
            if f.strip()
        ):
            continue
        if no_aws and pipeline_config.encoding in TORCH_INCOMPATIBLE_ENCODINGS:
            raise click.ClickException(
                f"Pipeline {pipeline_name!r} uses encoding"
                f" {pipeline_config.encoding!r}, which cannot generate torch"
                " goldens locally. Remove --no-aws or choose a different"
                " pipeline."
            )
        if no_aws:
            pipeline_config = pipeline_config.model_copy(
                update={"pregenerated_torch_goldens": None}
            )
        start_time = time.time()
        print(f"\n===== Running {pipeline_name} =====", flush=True)
        if _is_pixel_generation(pipeline_config):
            verdicts[pipeline_name] = run_pixel_generation_verification(
                pipeline_config,
                device_type=device_type,
                devices=devices_str,
                find_tolerances=find_tolerances,
                print_suggested_tolerances=print_suggested_tolerances,
            )
        else:
            verdicts[pipeline_name] = run_llm_verification(
                pipeline_config,
                device_type=device_type,
                devices=devices_str,
                find_tolerances=find_tolerances,
                print_suggested_tolerances=print_suggested_tolerances,
            )
        duration = f"{time.time() - start_time:.0f}s"
        print(
            f"\n===== Finished {pipeline_name} ({duration}) =====",
            flush=True,
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
