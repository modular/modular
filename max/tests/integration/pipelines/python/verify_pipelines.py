# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import enum
import functools
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
from llama3.verify import DiscrepancyReport, verify
from max.entrypoints.cli import DevicesOptionType
from test_common.process_isolation import run_in_isolated_process

# This is far from a universal standard, but this is the closest to a standard
# that I could find: BSD-derived programs sometimes use exit codes from
# "sysexits.h", which defines this exit code as "temp failure; user is invited
# to retry".  generate_llm_logits will emit this if it detects a failure is
# likely caused by a network flake and could be resolved by a retry.
EX_TEMPFAIL = 75


class DeviceKind(enum.Enum):
    CPU = "cpu"
    GPU = "gpu"


class VerificationStatus(enum.Enum):
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


def dump_results(
    verdicts: Mapping[str, VerificationVerdict], *, to: TextIO = sys.stdout
) -> None:
    # Even if verdicts is empty, we want to make sure to call write.  When we
    # call this from 'main', click passes us a LazyFile, and if we don't write
    # anything, we won't create the output file, which breaks downstream
    # workflows.

    # Warning: The logits verification pipeline parses these results
    # using grep/awk.
    # Please verify that this doesn't break before changing the output format
    to.write("| Status | Pipeline | Modality | KL Div | MAE |\n")
    to.write("|:------:|:---------|:--------:|:------:|:----:|\n")

    for pipeline, verdict in sorted(
        verdicts.items(), key=lambda x: x[0].lower()
    ):
        kl_div = "N/A"
        mae = "N/A"
        modality = "N/A"

        if verdict.discrepancy_report is not None:
            modality = verdict.discrepancy_report.model_modality
            mae = f"{verdict.discrepancy_report.avg_mae:.2e}"

            # Handle KL Div which may be None for non logits models
            if verdict.discrepancy_report.avg_kl_div is not None:
                kl_div = f"{verdict.discrepancy_report.avg_kl_div:.2e}"

        to.write(
            f"| {verdict.emoji} | {pipeline} | {modality} | {kl_div} | {mae} |\n"
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
        if any(forbidden_tag in tags for forbidden_tag in self.must_not_have):
            return False
        return True


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


def generate_llm_logits_nonretrying(
    *,
    framework: str,
    device: str,
    pipeline: str,
    encoding: str,
    output_path: Path,
) -> None:
    """Run :generate_llm_logits to generate logits for a model.

    Do not retry on flake.
    """
    parsed_device = DevicesOptionType.parse_from_str(device)
    device_specs = DevicesOptionType.device_specs(parsed_device)

    run_in_isolated_process(
        functools.partial(
            generate_llm_logits,
            framework_name=framework,
            device_specs=device_specs,
            pipeline_name=pipeline,
            encoding_name=encoding,
            output_path=output_path,
            print_output=False,
        ),
        timeout=600,
    )


def generate_llm_logits_with_retry(
    *,
    framework: str,
    device: str,
    pipeline: str,
    encoding: str,
    output_path: Path,
) -> None:
    """Generate logits with retry capability.

    This function calls generate_llm_logits_nonretrying and implements
    a simple retry mechanism that will attempt the operation again after
    a 60-second delay if a Flake exception occurs.
    """

    def attempt() -> None:
        generate_llm_logits_nonretrying(
            framework=framework,
            device=device,
            pipeline=pipeline,
            encoding=encoding,
            output_path=output_path,
        )

    try:
        attempt()
    except Flake:
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
    max_golden_path = Path(
        f"/tmp/goldens_max_{device_type.value}_{pipeline}_{encoding}.json"
    )
    generate_llm_logits_with_retry(
        framework="max",
        device=devices,
        pipeline=pipeline,
        encoding=encoding,
        output_path=max_golden_path,
    )

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
        generate_llm_logits_with_retry(
            framework="torch",
            device=devices,
            pipeline=pipeline,
            encoding=encoding,
            output_path=torch_golden_path,
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


# TODO(akirchhoff): Make this kw_only when we drop support for Python 3.9.
@dataclass
class PipelineDef:
    """Definition of the requirements and method of running a pipeline.

    'compatible_with' lists all device types this pipeline is compatible with.
    'run' should run and verify the pipeline results, returning a
    VerificationVerdict with the result of the verification, or alternatively
    raising an exception (same as returning VerificationVerdict.ERROR).
    """

    compatible_with: Sequence[DeviceKind]
    run: Callable[[DeviceKind, str, bool], VerificationVerdict]
    tags: Sequence[str] = field(default_factory=list)

    def run_protected(
        self,
        device_type: DeviceKind,
        devices: str,
        print_suggested_tolerances: bool,
    ) -> VerificationVerdict:
        try:
            return self.run(device_type, devices, print_suggested_tolerances)
        except Flake:
            return VerificationVerdict(status=VerificationStatus.FLAKE)
        except Exception:
            traceback.print_exc()
            return VerificationVerdict(status=VerificationStatus.ERROR)


PIPELINES = {
    # TODO(MODELS-454): Investigate sign flips.
    # We have sign flips that are seen across many models.
    # I definitely suspect a bug. The only place I don't see this is llama3_1-float32 on cpu.
    # Many of these values are far enough away from zero that they shouldn't be cause by minor rounding errors.
    # That said, they might be minor rounding errors that get magnified later in the model.
    # Layer by layer golden tests would be exceptionally useful.
    # For now, using atol alone to check correctness for these models.
    # Setting rtol high hides other issues.
    "llama3_1-q4_k": PipelineDef(
        compatible_with=[DeviceKind.CPU],
        run=lambda device_type,
        devices,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            print_suggested_tolerances=print_suggested_tolerances,
            pipeline="llama3_1",
            encoding="q4_k",
            # TODO(AIPIPE-135): Something is wildly wrong about our Q4_K
            # pipeline.  We only pass with these sky-high tolerances --
            # something is very wrong but at least we will be able to detect
            # further regressions with this.
            # Example sign flip:
            # `(26, array([12445])) |  -3.12881e+00 │  3.12882e+00`
            absolute_tolerance=30,
            relative_tolerance=1e-4,
            cos_dist_threshold=2,
            kl_div_threshold=25,
        ),
    ),
    "llama3_1-float32": PipelineDef(
        compatible_with=[DeviceKind.CPU, DeviceKind.GPU],
        tags=["big"],
        run=lambda device_type,
        devices,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            print_suggested_tolerances=print_suggested_tolerances,
            pipeline="llama3_1",
            encoding="float32",
            # TODO(AIPIPE-134): GPU has significantly worse tolerances than cpu.
            # cpu passes with `2e-04` atol. Gpu requires 100x worse tolerances.
            # Example sign flip (gpu only):
            # `(2, array([1150])) │  2.25306e-03 │ -2.24400e-03`
            absolute_tolerance=2e-02,
            relative_tolerance=8e-3,
            cos_dist_threshold=2e-6,
            kl_div_threshold=1e-5,
        ),
    ),
    "llama3_1-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        run=lambda device_type,
        devices,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            print_suggested_tolerances=print_suggested_tolerances,
            pipeline="llama3_1",
            encoding="bfloat16",
            pregenerated_torch_goldens_rlocation=(
                "torch_llama_golden/torch_llama3_1_bfloat16_golden.json"
            ),
            # TODO(AIPIPE-134): The absolute and relative differences here seem
            # too high.
            # Example sign flip:
            # `(13, array([57398])) │  6.65283e-03 │ -6.65283e-03`
            absolute_tolerance=0.2,
            relative_tolerance=0.17,
            cos_dist_threshold=3.2e-4,
            kl_div_threshold=2e-3,
        ),
    ),
    "Llama-3.3-70B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["h100-multi"],
        run=lambda device_type,
        devices,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            print_suggested_tolerances=print_suggested_tolerances,
            pipeline="llama3.3-70b",
            encoding="bfloat16",
            # TODO(AITLIB-194): Reduce thresholds after fixing correctness.
            # Example sign flip:
            # `(31, array([109396])) │  2.55127e-02 │ -2.55127e-02`
            absolute_tolerance=1.2,
            relative_tolerance=0.4,
            cos_dist_threshold=8e-4,
            kl_div_threshold=3.7e-3,
        ),
    ),
    "Llama4-17B-Scout-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["h100-multi"],
        run=lambda device_type,
        devices,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            print_suggested_tolerances=print_suggested_tolerances,
            pipeline="llama4-scout",
            encoding="bfloat16",
            absolute_tolerance=55,
            relative_tolerance=1.2,
            cos_dist_threshold=1.6,
            kl_div_threshold=float("inf"),
        ),
    ),
    "replit-code-v1_5-3b-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        run=lambda device_type,
        devices,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            print_suggested_tolerances=print_suggested_tolerances,
            pipeline="replit",
            encoding="bfloat16",
            pregenerated_torch_goldens_rlocation="torch_replit_golden/torch_replit-code-v1_5-3b_bfloat16_golden.json",
            # TODO(AIPIPE-166): Replit on GPU currently has very large
            # deviation between MAX and Torch, almost certainly a bug
            # somewhere, so these thresholds are extremely high.  Once the
            # deviation has been fixed, these thresholds should be adjusted
            # down to be more reasonable.
            # Example sign flip:
            # `(17, array([25123])) │ -7.65354e-01 │ 7.65625e-01`
            absolute_tolerance=63,
            relative_tolerance=1.0,
            cos_dist_threshold=0.9,
            kl_div_threshold=float("inf"),
        ),
    ),
    "mistral-nemo-instruct-2407-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big"],
        run=lambda device_type,
        devices,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            print_suggested_tolerances=print_suggested_tolerances,
            pipeline="mistral",
            encoding="bfloat16",
            pregenerated_torch_goldens_rlocation="torch_mistral_golden/torch_nemo-instruct-2407_bfloat16_golden.json",
            # TODO(AIPIPE-230): These tolerances are very high due to an accuracy regression.
            # Example sign flip:
            # `(18, array([113226])) │  5.44386e-02 │ -5.44434e-02`
            absolute_tolerance=2.1,
            relative_tolerance=1.3,
            cos_dist_threshold=2e-2,
            kl_div_threshold=3e-2,
        ),
    ),
    "llama3-vision-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big"],
        run=lambda device_type,
        devices,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            print_suggested_tolerances=print_suggested_tolerances,
            pipeline="llama3-vision",
            encoding="bfloat16",
            pregenerated_torch_goldens_rlocation="torch_llama3-vision_golden/torch_llama3_2_bfloat16_golden.json",
            # Note: llama-vision is not yet using llama3 rope.
            # TODO(bduke): Absolute tolerance here is larger than expected.
            # TODO(bduke): Relative tolerance is high due to sign flips for
            # small values near zero.
            # We should account for this since otherwise relative elementwise
            # tolerance isn't useful.
            # Example sign flip:
            # `(3, array([24040])) │  3.97949e-02 │ -3.97949e-02`
            absolute_tolerance=0.58,
            relative_tolerance=0.3,
            cos_dist_threshold=1e-3,
            kl_div_threshold=3.2e-3,
        ),
    ),
    "pixtral-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big"],
        run=lambda device_type,
        devices,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            print_suggested_tolerances=print_suggested_tolerances,
            pipeline="pixtral",
            encoding="bfloat16",
            pregenerated_torch_goldens_rlocation="torch_pixtral_golden/torch_pixtral_bfloat16_golden.json",
            # Example sign flip:
            # `(5, array([46295])) │ 6.83594e-02 │ -6.83594e-02`
            absolute_tolerance=0.9,
            relative_tolerance=0.2,
            cos_dist_threshold=2e-3,
            kl_div_threshold=6e-3,
        ),
    ),
    "mpnet-float32": PipelineDef(
        compatible_with=[DeviceKind.CPU, DeviceKind.GPU],
        run=lambda device_type,
        devices,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            print_suggested_tolerances=print_suggested_tolerances,
            pipeline="mpnet",
            encoding="float32",
            pregenerated_torch_goldens_rlocation="torch_mpnet_golden/torch_mpnet_float32_golden.json",
            # On CPU, mpnet passes with all values set to `1e-4`
            # GPU specifically requires these higher tolerances (30x worse).
            absolute_tolerance=2.3e-3,
            relative_tolerance=3e-2,
            cos_dist_threshold=2e-5,
            kl_div_threshold=3e-7,
        ),
    ),
    "mpnet-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        run=lambda device_type,
        devices,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            print_suggested_tolerances=print_suggested_tolerances,
            pipeline="mpnet",
            encoding="bfloat16",
            pregenerated_torch_goldens_rlocation="torch_mpnet_golden/torch_mpnet_bfloat16_golden.json",
            absolute_tolerance=9e-3,
            relative_tolerance=6e-2,
            cos_dist_threshold=2e-4,
            kl_div_threshold=3e-6,
        ),
    ),
    "Qwen2.5-7B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        run=lambda device_type,
        devices,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            print_suggested_tolerances=print_suggested_tolerances,
            pipeline="qwen",
            encoding="bfloat16",
            # Exmaple sign flip:
            # `(3, array([90716])) │ -2.88086e-02 │ 2.88086e-02`
            absolute_tolerance=0.9,
            relative_tolerance=6e-2,
            cos_dist_threshold=2.2e-3,
            kl_div_threshold=1.2e-1,
        ),
    ),
    "EXAONE-3.5-2.4B-Instruct-float32": PipelineDef(
        compatible_with=[DeviceKind.CPU, DeviceKind.GPU],
        tags=["big"],
        run=lambda device_type,
        devices,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            print_suggested_tolerances=print_suggested_tolerances,
            pipeline="exaone",
            encoding="float32",
            # TODO: Investigate why this is inf here.
            # Response text looks semantically close.
            # Example sign flip (seen on cpu and gpu for this model):
            # `(31, array([65924]) │ -8.71094e-01 │ 8.71138e-01`
            absolute_tolerance=3.0,
            relative_tolerance=0.3,
            cos_dist_threshold=3e-2,
            kl_div_threshold=float("inf"),
        ),
    ),
    "OLMo-1B-float32": PipelineDef(
        compatible_with=[DeviceKind.CPU, DeviceKind.GPU],
        run=lambda device_type,
        devices,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            print_suggested_tolerances=print_suggested_tolerances,
            pipeline="olmo",
            encoding="float32",
            # On CPU, olmo passes with atol set to `5e-4`
            # GPU specifically requires these higher tolerances (160x worse).
            # Example sign flip (gpu only):
            # `(36, array([48880])) │  5.52034e-03 │ -5.52034e-03`
            absolute_tolerance=3.5e-2,
            relative_tolerance=5e-2,
            cos_dist_threshold=9e-6,
            kl_div_threshold=9e-5,
        ),
    ),
    "Phi-3.5-mini-instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        run=lambda device_type,
        devices,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            print_suggested_tolerances=print_suggested_tolerances,
            pipeline="phi-3.5-mini",
            encoding="bfloat16",
            # TODO(MODELS-458): This model seems broken based on the thresholds
            absolute_tolerance=6.8,
            relative_tolerance=0.7,
            cos_dist_threshold=0.2,
            kl_div_threshold=1.1,
        ),
    ),
    "Phi-4-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big"],
        run=lambda device_type,
        devices,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            print_suggested_tolerances=print_suggested_tolerances,
            pipeline="phi-4",
            encoding="bfloat16",
            absolute_tolerance=14,
            relative_tolerance=1.4,
            cos_dist_threshold=0.7,
            kl_div_threshold=20.0,
        ),
    ),
    "llama-gptq": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        run=lambda device_type,
        devices,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            print_suggested_tolerances=print_suggested_tolerances,
            pregenerated_torch_goldens_rlocation="torch_llama-gptq_golden/torch_llama-gptq_golden.json",
            pipeline="llama-gptq",
            encoding="gptq",
            absolute_tolerance=9.0,
            relative_tolerance=0.7,
            cos_dist_threshold=0.2,
            kl_div_threshold=5,
        ),
    ),
    "llama-gptq-no-perm-idx": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        run=lambda device_type,
        devices,
        print_suggested_tolerances: run_llm_verification(
            device_type=device_type,
            devices=devices,
            print_suggested_tolerances=print_suggested_tolerances,
            pipeline="llama-gptq-no-perm-idx",
            pregenerated_torch_goldens_rlocation="torch_llama-gptq_golden/torch_llama-gptq-no-perm-idx_golden.json",
            encoding="gptq",
            absolute_tolerance=20,
            relative_tolerance=1.1,
            cos_dist_threshold=0.6,
            kl_div_threshold=25,
        ),
    ),
}


@click.command()
@click.option(
    "--report",
    type=click.File("w"),
    help="Output the coverage report to the specified file",
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
    devices_str: Optional[str],
    pipeline: Optional[str],
    tag_filter: TagFilter,
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
            print(f"\n===== Running {pipeline_name} =====", flush=True)
            verdicts[pipeline_name] = pipeline_def.run_protected(
                device_type, devices_str, print_suggested_tolerances
            )
            print(f"===== Finished {pipeline_name} =====", flush=True)
    else:
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
            device_type, devices_str, print_suggested_tolerances
        )

    if report:
        dump_results(verdicts, to=report)

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
    dump_results(verdicts)

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
