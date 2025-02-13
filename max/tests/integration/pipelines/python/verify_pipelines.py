# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import enum
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, TextIO, Union

import click


class DeviceKind(enum.Enum):
    CPU = "cpu"
    GPU = "gpu"


class VerificationVerdict(enum.Enum):
    OK = "ok"
    INVALID = "invalid"
    ERROR = "error"

    @property
    def emoji(self) -> str:
        return _VERDICT_EMOJI[self]


_VERDICT_EMOJI = {
    VerificationVerdict.OK: "✅",
    VerificationVerdict.INVALID: "🟡",
    VerificationVerdict.ERROR: "❌",
}


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
    to.write("")
    for pipeline, verdict in verdicts.items():
        print(f"  {verdict.emoji} {pipeline}", file=to)


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


def generate_llm_logits(
    *,
    framework: str,
    device: str,
    pipeline: str,
    version: str,
    encoding: str,
    output_path: Path,
) -> None:
    """Run :generate_llm_logits to generate logits for a model."""
    subprocess.run(
        [
            os.environ["GENERATE_LLM_LOGITS_BIN"],
            f"--framework={framework}",
            f"--device={device}",
            f"--pipeline={pipeline}",
            f"--version={version}",
            f"--encoding={encoding}",
            f"--output={output_path}",
        ],
        check=True,
    )
    pass


def run_llm_verification(
    *,
    device_type: DeviceKind,
    devices: str,
    pipeline: str,
    version: str,
    encoding: str,
    pregenerated_torch_goldens_rlocation: Optional[str] = None,
    kl_div_threshold: Optional[float] = None,
    cos_dist_threshold: Optional[float] = None,
    absolute_tolerance: Optional[float] = None,
    relative_tolerance: Optional[float] = None,
) -> VerificationVerdict:
    """Run a Llama3 verification with the given model and weights encoding.

    See SDK/integration-test/pipelines/python/llama3/evaluate_llama.py for
    definitions of acceptable values for model and encoding.

    extra_verify_flags are passed to
    SDK/integration-test/pipelines/python/llama3/verify.py -- check that script
    for details on acceptable flags.
    """
    max_golden_path = Path(
        f"/tmp/goldens_max_{device_type.value}_{pipeline}_{version}_{encoding}.json"
    )
    generate_llm_logits(
        framework="max",
        device=devices,
        pipeline=pipeline,
        version=version,
        encoding=encoding,
        output_path=max_golden_path,
    )
    if pregenerated_torch_goldens_rlocation is not None:
        # This workflow runs on an A10.  The Torch reference runs out of memory
        # on an A10, so it was run manually on an A100 and the result goldens
        # uploaded.  Use these pre-generated goldens in this case.
        torch_golden_path = resolve_rlocation(
            pregenerated_torch_goldens_rlocation
        )
    else:
        torch_golden_path = Path(
            f"/tmp/goldens_torch_{device_type.value}_{pipeline}_{version}_{encoding}.json"
        )
        generate_llm_logits(
            framework="torch",
            device=devices,
            pipeline=pipeline,
            version=version,
            encoding=encoding,
            output_path=torch_golden_path,
        )

    eval_metrics = []
    threshold_flags = []
    if absolute_tolerance is not None and relative_tolerance is not None:
        eval_metrics.append("tol")
        threshold_flags.append(f"--absolute-tolerance={absolute_tolerance}")
        threshold_flags.append(f"--relative-tolerance={relative_tolerance}")
    if cos_dist_threshold is not None:
        eval_metrics.append("cos")
        threshold_flags.append(f"--cos-dist-threshold={cos_dist_threshold}")
    if kl_div_threshold is not None:
        eval_metrics.append("kl")
        threshold_flags.append(f"--kl-div-threshold={kl_div_threshold}")
    if not eval_metrics:
        raise ValueError(
            "Please provide absolute, relative, cos, or kldiv error thresholds."
            " Otherwise no metrics will be computed."
        )
    try:
        subprocess.run(
            [
                os.environ["LLAMA3_VERIFY_BIN"],
                str(max_golden_path),
                str(torch_golden_path),
                "--eval-metric=" + ",".join(eval_metrics),
            ]
            + threshold_flags,
            check=True,
        )
    except subprocess.CalledProcessError:
        return VerificationVerdict.INVALID
    return VerificationVerdict.OK


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
    run: Callable[[DeviceKind, str], VerificationVerdict]
    tags: Sequence[str] = field(default_factory=list)

    def run_protected(
        self, device_type: DeviceKind, devices: str
    ) -> VerificationVerdict:
        try:
            return self.run(device_type, devices)
        except Exception:
            traceback.print_exc()
            return VerificationVerdict.ERROR


PIPELINES = {
    "llama3_1-q4_k": PipelineDef(
        compatible_with=[DeviceKind.CPU],
        run=lambda device_type, devices: run_llm_verification(
            device_type=device_type,
            devices=devices,
            pipeline="llama",
            version="llama3_1",
            encoding="q4_k",
            # TODO(AIPIPE-135): Something is wildly wrong about our Q4_K
            # pipeline.  We only pass with these sky-high tolerances --
            # something is very wrong but at least we will be able to detect
            # further regressions with this.
            kl_div_threshold=30.0,
            cos_dist_threshold=2.0,
            absolute_tolerance=25.0,
            relative_tolerance=2.1,
        ),
    ),
    "llama3_1-float32": PipelineDef(
        compatible_with=[DeviceKind.CPU],
        run=lambda device_type, devices: run_llm_verification(
            device_type=device_type,
            devices=devices,
            pipeline="llama",
            version="llama3_1",
            encoding="float32",
            kl_div_threshold=0.005,
            cos_dist_threshold=0.002,
            # TODO(AIPIPE-134): The absolute and relative differences here seem
            # too high.
            absolute_tolerance=0.8,
            relative_tolerance=2.1,
        ),
    ),
    "llama3_1-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        run=lambda device_type, devices: run_llm_verification(
            device_type=device_type,
            devices=devices,
            pipeline="llama",
            version="llama3_1",
            encoding="bfloat16",
            pregenerated_torch_goldens_rlocation=(
                "torch_llama_golden/torch_llama3_1_bfloat16_golden.json"
            ),
            kl_div_threshold=0.006,
            cos_dist_threshold=0.002,
            # TODO(AIPIPE-134): The absolute and relative differences here seem
            # too high.
            absolute_tolerance=0.8,
            relative_tolerance=2.1,
        ),
    ),
    "Llama-3.3-70B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["h100-multi"],
        run=lambda device_type, devices: run_llm_verification(
            device_type=device_type,
            devices=devices,
            pipeline="llama3.3-70b",
            version="Llama-3.3-70B-Instruct",
            encoding="bfloat16",
            # TODO(AITLIB-194): Reduce thresholds after fixing correctness.
            kl_div_threshold=0.5,
            cos_dist_threshold=0.002,
            absolute_tolerance=0.8,
            relative_tolerance=2.1,
        ),
    ),
    "replit-code-v1_5-3b-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        run=lambda device_type, devices: run_llm_verification(
            device_type=device_type,
            devices=devices,
            pipeline="replit",
            version="replit-code-v1_5-3b",
            encoding="bfloat16",
            pregenerated_torch_goldens_rlocation="torch_replit_golden/torch_replit-code-v1_5-3b_bfloat16_golden.json",
            # TODO(AIPIPE-166): Replit on GPU currently has very large
            # deviation between MAX and Torch, almost certainly a bug
            # somewhere, so these thresholds are extremely high.  Once the
            # deviation has been fixed, these thresholds should be adjusted
            # down to be more reasonable.
            kl_div_threshold=float("inf"),
            cos_dist_threshold=1.5,
            absolute_tolerance=100,
            relative_tolerance=2.5,
        ),
    ),
    "mistral-nemo-instruct-2407-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big"],
        run=lambda device_type, devices: run_llm_verification(
            device_type=device_type,
            devices=devices,
            pipeline="mistral",
            version="nemo-instruct-2407",
            encoding="bfloat16",
            pregenerated_torch_goldens_rlocation="torch_mistral_golden/torch_nemo-instruct-2407_bfloat16_golden.json",
            # TODO(AIPIPE-230): These tolerances are very high due to an accuracy regression.
            kl_div_threshold=0.03,
            cos_dist_threshold=0.02,
            absolute_tolerance=1.5,
            relative_tolerance=2.0,
        ),
    ),
    "llama3-vision-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big"],
        run=lambda device_type, devices: run_llm_verification(
            device_type=device_type,
            devices=devices,
            pipeline="llama3-vision",
            version="llama3_2",
            encoding="bfloat16",
            pregenerated_torch_goldens_rlocation="torch_llama3-vision_golden/torch_llama3_2_bfloat16_golden.json",
            kl_div_threshold=8e-3,
            cos_dist_threshold=2e-3,
            # TODO(bduke): Absolute tolerance here is larger than expected.
            absolute_tolerance=1.0,
            # TODO(bduke): Relative tolerance is high due to sign flips for
            # small values near zero.
            # We should account for this since otherwise relative elementwise
            # tolerance isn't useful.
            relative_tolerance=2.5,
        ),
    ),
    "pixtral-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        tags=["big"],
        run=lambda device_type, devices: run_llm_verification(
            device_type=device_type,
            devices=devices,
            pipeline="pixtral",
            version="pixtral12b",
            encoding="bfloat16",
            pregenerated_torch_goldens_rlocation="torch_pixtral_golden/torch_pixtral_bfloat16_golden.json",
            kl_div_threshold=0.05,
            cos_dist_threshold=0.005,
            absolute_tolerance=1.0,
            relative_tolerance=2.0,
        ),
    ),
    "mpnet-float32": PipelineDef(
        compatible_with=[DeviceKind.CPU],
        run=lambda device_type, devices: run_llm_verification(
            device_type=device_type,
            devices=devices,
            pipeline="mpnet",
            version="general",
            encoding="float32",
            pregenerated_torch_goldens_rlocation="torch_mpnet_golden/torch_mpnet_float32_golden.json",
            cos_dist_threshold=1e-5,
            absolute_tolerance=1e-4,
            relative_tolerance=0.5,
        ),
    ),
    "mpnet-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        run=lambda device_type, devices: run_llm_verification(
            device_type=device_type,
            devices=devices,
            pipeline="mpnet",
            version="general",
            encoding="bfloat16",
            pregenerated_torch_goldens_rlocation="torch_mpnet_golden/torch_mpnet_bfloat16_golden.json",
            cos_dist_threshold=2e-4,
            # Relative/abs tolerances are a lot higher for bfloat16, but since
            # the cosine distance is reasonable we just test one metric.
        ),
    ),
    "Qwen2.5-7B-Instruct-bfloat16": PipelineDef(
        compatible_with=[DeviceKind.GPU],
        run=lambda device_type, devices: run_llm_verification(
            device_type=device_type,
            devices=devices,
            pipeline="qwen",
            version="2.5-7B-Instruct",
            encoding="bfloat16",
            kl_div_threshold=0.2,
            cos_dist_threshold=0.01,
            absolute_tolerance=1.4,
            relative_tolerance=2.1,
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
def main(
    report: Optional[TextIO],
    devices_str: Optional[str],
    pipeline: Optional[str],
    tag_filter: TagFilter,
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
            print(f"Running {pipeline_name}...", flush=True)
            verdicts[pipeline_name] = pipeline_def.run_protected(
                device_type, devices_str
            )
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
            device_type, devices_str
        )

    if report:
        dump_results(verdicts, to=report)

    print()
    print("-" * 40)
    print()
    print("# pipelines run:", len(verdicts))
    for verdict in list(VerificationVerdict):
        print(
            f"# pipelines {verdict.name}:",
            sum(v == verdict for v in verdicts.values()),
        )
    print()
    dump_results(verdicts)

    if any(v != VerificationVerdict.OK for v in verdicts.values()):
        sys.exit(1)


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    main()
