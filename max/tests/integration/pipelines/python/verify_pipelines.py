# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import dataclasses
import enum
import functools
import json
import os
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, TextIO, Union

import click
from generate_llm_logits import Flake, generate_llm_logits
from max.entrypoints.cli import DevicesOptionType
from test_common.evaluate import ModelOutput
from test_common.numpy_encoder import NumpyDecoder
from test_common.process_isolation import run_in_isolated_process
from verify import DiscrepancyReport, verify
from verify import ModelModality as Modality

# This is far from a universal standard, but this is the closest to a standard
# that I could find: BSD-derived programs sometimes use exit codes from
# "sysexits.h", which defines this exit code as "temp failure; user is invited
# to retry".  generate_llm_logits will emit this if it detects a failure is
# likely caused by a network flake and could be resolved by a retry.
EX_TEMPFAIL = 75


class DeviceKind(enum.Enum):
    CPU = "cpu"
    GPU = "gpu"


class VerificationStatus(str, enum.Enum):
    OK = "ok"
    INVALID = "invalid"
    ERROR = "error"
    FLAKE = "flake"

    @property
    def emoji(self) -> str:
        return _VERDICT_EMOJI[self]


_VERDICT_EMOJI = {
    VerificationStatus.OK: "✅",
    VerificationStatus.INVALID: "🟡",
    VerificationStatus.ERROR: "❌",
    VerificationStatus.FLAKE: "❄️",
}


@dataclass
class VerificationVerdict:
    status: VerificationStatus
    discrepancy_report: Optional[DiscrepancyReport] = None

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
            "**KL Div** = average over all prompts (lower is better)\n"
            "Note: This is NOT the max threshold used for pass/fail checks\n\n"
            "**Diff** = change from previous run\n"
            "  • Negative = accuracy improved\n"
            "  • Positive = accuracy worsened\n"
            "  • N/A = no previous verdict\n"
            "  • --- = no change\n"
        )
        to.write("| Status | Model | KL Div | Diff |\n")
        to.write("| :----: | :---  | :---:  | :---:|\n")

        for name, verdict in sorted(verdicts.items(), key=verdict_sorting_key):
            if verdict.discrepancy_report is None:
                continue
            if verdict.discrepancy_report.model_modality != Modality.LOGIT:
                continue
            kl = f"{verdict.discrepancy_report.avg_kl_div:.2e}"

            diff_str = "N/A"
            if previous_verdicts and name in previous_verdicts:
                diff_str = compute_diff(verdict, previous_verdicts[name])

            to.write(
                f"| {verdict.emoji} | {display_name(name)} | {kl} | {diff_str} |\n"
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
        value: Union[str, TagFilter],
        param: Optional[click.Parameter],
        ctx: Optional[click.Context],
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


def generate_llm_logits_with_optional_retry(
    *,
    framework: str,
    device: str,
    pipeline: str,
    encoding: str,
    output_path: Path,
    reference: list[ModelOutput] | None = None,
    retry_on_flake: bool = True,
) -> None:
    """Generate logits with optional retry capability.

    If retry_on_flake is True, will retry once after 60 seconds on Flake exception.
    """
    parsed_device = DevicesOptionType.parse_from_str(device)
    device_specs = DevicesOptionType.device_specs(parsed_device)

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
            timeout=600,
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
    pregenerated_torch_goldens_rlocation: Optional[str] = None,
    absolute_tolerance: Optional[float] = None,
    relative_tolerance: Optional[float] = None,
    cos_dist_threshold: Optional[float] = None,
    kl_div_threshold: Optional[float] = None,
) -> VerificationVerdict:
    """Run a Llama3 verification with the given model and weights encoding.

    extra_verify_flags are passed to
    SDK/integration-test/pipelines/python/llama3/verify.py -- check that script
    for details on acceptable flags.
    """

    # Run the torch baseline or load it from golden.
    if pregenerated_torch_goldens_rlocation is not None:
        # This workflow runs on an A10. The Torch reference runs out of memory
        # on an A10, so it was run manually on an A100 and the result goldens
        # uploaded. Use these pre-generated goldens in this case.
        torch_golden_path = resolve_rlocation(
            pregenerated_torch_goldens_rlocation
        )
    else:
        torch_golden_path = Path(
            f"/tmp/goldens_torch_{device_type.value}_{pipeline}_{encoding}.json"
        )
        generate_llm_logits_with_optional_retry(
            framework="torch",
            device=devices,
            pipeline=pipeline,
            encoding=encoding,
            output_path=torch_golden_path,
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
        f"/tmp/goldens_max_{device_type.value}_{pipeline}_{encoding}.json"
    )
    generate_llm_logits_with_optional_retry(
        framework="max",
        device=devices,
        pipeline=pipeline,
        encoding=encoding,
        output_path=max_golden_path,
        reference=torch_results,
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
            return self.run(
                device_type,
                devices,
                find_tolerances,
                print_suggested_tolerances,
            )
        except Flake:
            return VerificationVerdict(status=VerificationStatus.FLAKE)
        except Exception:
            traceback.print_exc()
            return VerificationVerdict(status=VerificationStatus.ERROR)


# Helper function to create pipeline runner
def _make_pipeline_runner(
    *,
    pipeline: str,
    encoding: str,
    pregenerated_torch_goldens_rlocation: Optional[str] = None,
    absolute_tolerance: Optional[float] = None,
    relative_tolerance: Optional[float] = None,
    cos_dist_threshold: Optional[float] = None,
    kl_div_threshold: Optional[float] = None,
) -> Callable[[DeviceKind, str, bool, bool], VerificationVerdict]:
    """
    Build and return a small closure that executes `run_llm_verification`
    for a single model configuration.

    Args:
        pipeline: Name of the model / pipeline to verify.
        encoding: Weight / activation dtype (e.g. "float32", "bfloat16").
        pregenerated_torch_goldens_rlocation: Runfiles-relative path to a
            JSON file with cached Torch reference outputs.  If provided,
            it is resolved via `resolve_rlocation`; otherwise Torch
            outputs are generated on the fly.
        absolute_tolerance: Per-token element-wise absolute tolerance (atol).
        relative_tolerance: Per-token element-wise relative tolerance (rtol).
        cos_dist_threshold: Per-token cosine-distance threshold
            (not element-wise).
        kl_div_threshold: Per-token KL-divergence threshold
            (not element-wise).

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
            pregenerated_torch_goldens_rlocation=pregenerated_torch_goldens_rlocation,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            cos_dist_threshold=cos_dist_threshold,
            kl_div_threshold=kl_div_threshold,
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
            pipeline="llama3-8b",
            encoding="float32",
            pregenerated_torch_goldens_rlocation="torch_llama_golden/torch_llama3-8b_float32_golden.json",
            absolute_tolerance=2.9e-2,
            relative_tolerance=9.4e-2,
            cos_dist_threshold=2.1e-6,
            kl_div_threshold=3.0e-7,
        ),
    ),
    "meta-llama/Llama-3.1-8B-Instruct-float32": PipelineDef(
        compatible_with=[DeviceKind.CPU, DeviceKind.GPU],
        tags=["big"],
        run=_make_pipeline_runner(
            pipeline="llama3.1-8b",
            encoding="float32",
            absolute_tolerance=2.1e-02,
            relative_tolerance=7.2e-3,
            cos_dist_threshold=1.7e-6,
            kl_div_threshold=1.0e-10,
        ),
    ),
    "sentence-transformers/all-mpnet-base-v2-float32": PipelineDef(
        compatible_with=[DeviceKind.CPU, DeviceKind.GPU],
        run=_make_pipeline_runner(
            pipeline="mpnet",
            encoding="float32",
            pregenerated_torch_goldens_rlocation="torch_mpnet_golden/torch_mpnet_float32_golden.json",
            # On CPU, mpnet passes with all values set to `1e-4`
            # GPU specifically requires these higher tolerances (30x worse).
            absolute_tolerance=2.3e-3,
            relative_tolerance=2.5e-2,
            cos_dist_threshold=2e-5,
            kl_div_threshold=1.0e-10,
        ),
    ),
    "allenai/OLMo-1B-hf-float32": PipelineDef(
        compatible_with=[DeviceKind.CPU, DeviceKind.GPU],
        run=_make_pipeline_runner(
            pipeline="olmo",
            encoding="float32",
            # On CPU, olmo passes with atol set to `5e-4`
            # GPU specifically requires these higher tolerances (160x worse).
            absolute_tolerance=3.7e-2,
            relative_tolerance=4.2e-2,
            cos_dist_threshold=8.2e-6,
            kl_div_threshold=6.0e-5,
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
            pipeline="llama3-8b",
            encoding="q4_k",
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
            pipeline="llama3-8b",
            encoding="bfloat16",
            cos_dist_threshold=3.7e-2,
            kl_div_threshold=1.3e-1,
        ),
    ),
    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF-q4_k": PipelineDef(
        compatible_with=[DeviceKind.CPU],
        run=_make_pipeline_runner(
            pipeline="llama3.1-8b",
            encoding="q4_k",
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
            pipeline="llama3.1-8b",
            encoding="bfloat16",
            cos_dist_threshold=2.6e-4,
            kl_div_threshold=4.8e-3,
        ),
    ),
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-float8-static": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        # This does not require multigpu, but does require h100.
        tags=["h100-multi"],
        run=_make_pipeline_runner(
            pipeline="llama3.1-8b-float8-static",
            encoding="float8_e4m3fn",
            # This model does not run with torch and transformers.
            # It only runs with vllm.
            # For now compare to the bfloat16 goldens cause we have them.
            pregenerated_torch_goldens_rlocation=(
                "torch_llama_golden/torch_llama3_1_bfloat16_golden.json"
            ),
            cos_dist_threshold=8.5e-3,
            kl_div_threshold=8.6e-2,
        ),
    ),
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic-float8-dynamic": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        # This does not require multigpu, but does require h100.
        tags=["h100-multi"],
        run=_make_pipeline_runner(
            pipeline="llama3.1-8b-float8-dynamic",
            encoding="float8_e4m3fn",
            # This model does not run with torch and transformers.
            # It only runs with vllm.
            # For now compare to the bfloat16 goldens cause we have them.
            pregenerated_torch_goldens_rlocation=(
                "torch_llama_golden/torch_llama3_1_bfloat16_golden.json"
            ),
            cos_dist_threshold=5.6e-3,
            kl_div_threshold=3.9e-2,
        ),
    ),
    "meta-llama/Llama-3.2-1B-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        # Needs h100 for specific kernels.
        tags=["h100-multi"],
        run=_make_pipeline_runner(
            pipeline="llama3.2-1b",
            encoding="bfloat16",
            cos_dist_threshold=9.5e-04,
            kl_div_threshold=2.5e-03,
        ),
    ),
    "meta-llama/Llama-3.3-70B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["h100-multi"],
        run=_make_pipeline_runner(
            pipeline="llama3.3-70b",
            encoding="bfloat16",
            # TODO(AITLIB-194): Reduce thresholds after fixing correctness.
            cos_dist_threshold=3.8e-4,
            kl_div_threshold=1.9e-3,
        ),
    ),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["h100-multi"],
        run=_make_pipeline_runner(
            pipeline="llama4-scout",
            encoding="bfloat16",
            pregenerated_torch_goldens_rlocation=(
                "torch_llama4_golden/torch_llama4_scout_bfloat16_golden.json"
            ),
            cos_dist_threshold=5.0e-3,
            kl_div_threshold=4.0e-1,
        ),
    ),
    "mistralai/Mistral-Nemo-Instruct-2407-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big"],
        run=_make_pipeline_runner(
            pipeline="mistral",
            encoding="bfloat16",
            pregenerated_torch_goldens_rlocation="torch_mistral_golden/torch_nemo-instruct-2407_bfloat16_golden.json",
            # TODO(AIPIPE-230): These tolerances are very high due to an accuracy regression.
            cos_dist_threshold=1.3e-2,
            kl_div_threshold=2.7e-2,
        ),
    ),
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big", "h100-multi"],
        run=_make_pipeline_runner(
            pipeline="mistral3",
            encoding="bfloat16",
            cos_dist_threshold=6.3e-4,
            kl_div_threshold=2.6e-3,
        ),
    ),
    "meta-llama/Llama-3.2-11B-Vision-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big"],
        run=_make_pipeline_runner(
            pipeline="llama3-vision",
            encoding="bfloat16",
            pregenerated_torch_goldens_rlocation="torch_llama3-vision_golden/torch_llama3_2_bfloat16_golden.json",
            # Note: llama-vision is not yet using llama3 rope.
            cos_dist_threshold=1.1e-3,
            kl_div_threshold=5.4e-3,
        ),
    ),
    "OpenGVLab/InternVL3-1B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        # TODO(KERN-1861): MI300x: Memory access fault by GPU node-2.
        tags=["nvidia-only"],
        run=_make_pipeline_runner(
            pipeline="internvl3-1b-instruct",
            encoding="bfloat16",
            # TODO(MODELS-565): Fix InternVL correctness.
            cos_dist_threshold=2.3e-03,
            kl_div_threshold=1.5e-02,
        ),
    ),
    "OpenGVLab/InternVL3-8B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["h100-multi"],
        run=_make_pipeline_runner(
            pipeline="internvl3-8b-instruct",
            encoding="bfloat16",
            # TODO(MODELS-565): Fix InternVL correctness.
            cos_dist_threshold=1.1e-1,
            kl_div_threshold=3.3e-01,
        ),
    ),
    "OpenGVLab/InternVL3-14B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["h100-multi"],
        run=_make_pipeline_runner(
            pipeline="internvl3-14b-instruct",
            encoding="bfloat16",
            cos_dist_threshold=4.4e-03,
            kl_div_threshold=9.8e-03,
        ),
    ),
    "OpenGVLab/InternVL3-38B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["h100-multi"],
        run=_make_pipeline_runner(
            pipeline="internvl3-38b-instruct",
            encoding="bfloat16",
            cos_dist_threshold=1.9e-03,
            kl_div_threshold=7.5e-03,
        ),
    ),
    "mistral-community/pixtral-12b-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big"],
        run=_make_pipeline_runner(
            pipeline="pixtral",
            encoding="bfloat16",
            pregenerated_torch_goldens_rlocation="torch_pixtral_golden/torch_pixtral_bfloat16_golden.json",
            cos_dist_threshold=1.5e-3,
            kl_div_threshold=4.5e-3,
        ),
    ),
    "Qwen/Qwen2.5-7B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["nvidia-only"],  # TODO: Has much worse accuracy on AMD GPUs.
        run=_make_pipeline_runner(
            pipeline="qwen",
            encoding="bfloat16",
            cos_dist_threshold=2.7e-3,
            kl_div_threshold=1.3e-1,
        ),
    ),
    "Qwen/Qwen3-8B-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big", "nvidia-only"],  # TODO: Attention is broken on AMD.
        run=_make_pipeline_runner(
            pipeline="qwen3",
            encoding="bfloat16",
            cos_dist_threshold=1.1e-3,
            kl_div_threshold=3.7e-3,
        ),
    ),
    "allenai/OLMo-2-1124-7B-float32": PipelineDef(
        compatible_with=[DeviceKind.CPU, DeviceKind.GPU],
        tags=["big"],
        run=_make_pipeline_runner(
            pipeline="olmo2-7b",
            encoding="float32",
            cos_dist_threshold=2.1e-5,
            kl_div_threshold=4.6e-7,
        ),
    ),
    "HuggingFaceM4/Idefics3-8B-Llama3": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big", "nvidia-only"],
        run=_make_pipeline_runner(
            pipeline="idefics3-8b-llama3",
            encoding="bfloat16",
            # TODO: Accuracy is much worse on AMD.
            # so we might have an AMD kernel bug here
            cos_dist_threshold=8.8e-03,
            kl_div_threshold=8.7e-02,
        ),
    ),
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-float32": PipelineDef(
        compatible_with=[DeviceKind.CPU, DeviceKind.GPU],
        tags=["big"],
        run=_make_pipeline_runner(
            pipeline="exaone",
            encoding="float32",
            # TODO: Accuracy is much better on AMD.
            # so we might have an nvidia kernel bug here
            cos_dist_threshold=2.5e-2,
            kl_div_threshold=1.3e-2,
        ),
    ),
    "microsoft/Phi-3.5-mini-instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        run=_make_pipeline_runner(
            pipeline="phi-3.5-mini",
            encoding="bfloat16",
            # TODO(MODELS-458): This model seems broken based on the thresholds
            cos_dist_threshold=1.5e-2,
            kl_div_threshold=4.0e-1,
        ),
    ),
    "microsoft/phi-4-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big"],
        run=_make_pipeline_runner(
            pipeline="phi-4",
            encoding="bfloat16",
            cos_dist_threshold=9.8e-5,
            kl_div_threshold=6.9e-3,
        ),
    ),
    # TODO(MODELS-693): bfloat16->float16 casting issue.
    # "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4-gptq": PipelineDef(
    #     compatible_with=[DeviceKind.GPU],
    #     tags=["nvidia-only"],
    #     run=_make_pipeline_runner(
    #         pipeline="llama-gptq",
    #         encoding="gptq",
    #         pregenerated_torch_goldens_rlocation="torch_llama-gptq_golden/torch_llama-gptq_golden.json",
    #         cos_dist_threshold=3.3e-4,
    #         kl_div_threshold=2.7e-3,
    #     ),
    # ),
    # "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4-gptq-no-perm-idx": PipelineDef(
    #     compatible_with=[DeviceKind.GPU],
    #     tags=["nvidia-only"],
    #     run=_make_pipeline_runner(
    #         pipeline="llama-gptq-no-perm-idx",
    #         encoding="gptq",
    #         pregenerated_torch_goldens_rlocation="torch_llama-gptq_golden/torch_llama-gptq-no-perm-idx_golden.json",
    #         cos_dist_threshold=3.6e-4,
    #         kl_div_threshold=1.4e-3,
    #     ),
    # ),
    # TODO(AITLIB-372): investigate why accuracy tanked when switching to explicit weight dtype casting.
    "deepseek-ai/DeepSeek-V2-Lite-Chat-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big", "nvidia-only"],
        run=_make_pipeline_runner(
            pipeline="deepseek-v2-lite",
            encoding="bfloat16",
            # TODO(MODELS-516): Investigate need for high tolerances here.
            # TODO(GENAI-216): Investigate non-deterministic output.
            cos_dist_threshold=4.1e-03,
            kl_div_threshold=2.6e-01,
        ),
    ),
    "google/gemma-3-1b-it-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        run=_make_pipeline_runner(
            pipeline="gemma3-1b",
            encoding="bfloat16",
            pregenerated_torch_goldens_rlocation="torch_gemma3-1b_golden/torch_gemma3-1b_bfloat16_golden.json",
            cos_dist_threshold=1.3e-3,
            kl_div_threshold=9.4e-03,
        ),
    ),
    "google/gemma-3-12b-it-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big"],
        run=_make_pipeline_runner(
            pipeline="gemma3-multimodal",
            encoding="bfloat16",
            pregenerated_torch_goldens_rlocation="torch_gemma3-multimodal_golden/torch_gemma3-multimodal_bfloat16_golden.json",
            absolute_tolerance=1.0e-04,
            relative_tolerance=2.0,
            cos_dist_threshold=2.2e-02,
            kl_div_threshold=0.19,
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
    report: Optional[TextIO],
    store_verdicts_json: Optional[Path],
    load_verdicts_json: Optional[Path],
    devices_str: Optional[str],
    pipeline: Optional[str],
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
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    main()
