#!/usr/bin/env python3
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

# Run: bazel test max/kernels/benchmarks/autotune:autotune_tests

import os
import string
import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.core import Command
from click.testing import CliRunner
from kbench import cli as kbench_cli
from kbench_model import (
    KbenchCache,
    Param,
    ProcessOutput,
    Scheduler,
    SpecInstance,
    SupportedLangs,
)
from kplot import _resolve_ytext_unit
from kplot import cli as kplot_cli
from kprofile import cli as kprofile_cli
from pytest_mock import MockerFixture
from rich.console import Console
from terminal_viz import render_results
from utils import _get_gpu_count, check_valid_target_accelerator


def get_abs_path(path: str) -> Path:
    return Path(string.Template(str(path)).substitute(os.environ)).absolute()


kernel_benchmarks_root = os.environ["KERNEL_BENCHMARKS_ROOT"]


def test_check_valid_target_accelerator() -> None:
    # Valid accelerators should return True
    assert check_valid_target_accelerator("amdgpu:mi300x")

    # Invalid accelerators should return False
    assert not check_valid_target_accelerator("invalid_accelerator")
    assert not check_valid_target_accelerator("")


# TODO: refactor to match the expected results
def _invoke_cli(
    cli: Command,
    test_cases: list[str],
    exit_code: int = os.EX_OK,
) -> None:
    os_env = os.environ.copy()
    for _, test_cmd in enumerate(test_cases):
        try:
            result = CliRunner().invoke(cli, test_cmd, env=os_env)
            assert result.exit_code == exit_code, result.output
            print(result.output)
        except Exception as e:
            print(
                f"Exit code: {result.exit_code}, Exception: {result.exception}"
            )


def test_kbench() -> None:
    _invoke_cli(
        kbench_cli,
        test_cases=[
            "--skip-clock-check --help",
            f"{kernel_benchmarks_root}/autotune/test.yaml -fv --dryrun",
            f"{kernel_benchmarks_root}/autotune/test.yaml -fv -o {kernel_benchmarks_root}/autotune/tests/output",
        ],
    )

    print(
        "autotune/tests:",
        os.listdir(f"{kernel_benchmarks_root}/autotune/tests"),
        os.listdir("."),
    )

    path = [
        Path(f"{kernel_benchmarks_root}/autotune/tests/output.txt"),
        Path(f"{kernel_benchmarks_root}/autotune/tests/output.pkl"),
        Path(f"{kernel_benchmarks_root}/autotune/tests/output.csv"),
        Path(f"{kernel_benchmarks_root}/autotune/tests/baseline.csv"),
    ]

    for p in path:
        assert p.exists()

    df = pd.read_csv(f"{kernel_benchmarks_root}/autotune/tests/output.csv")
    baseline_df = pd.read_csv(
        f"{kernel_benchmarks_root}/autotune/tests/baseline.csv"
    )

    pd.testing.assert_series_equal(df["name"], baseline_df["name"])
    pd.testing.assert_series_equal(df["spec"], baseline_df["spec"])


# TODO: resolving mpirun deps in bazel.
# def test_kbench_mpirun() -> None:
#     _invoke_cli(
#         kbench_cli,
#         test_cases=[
#             f"{kernel_benchmarks_root}/autotune/test.yaml --mpirun-np 5 -fv -o {kernel_benchmarks_root}/autotune/tests/output_mpirun",
#         ],
#     )

#     print(
#         "autotune/tests:",
#         os.listdir(f"{kernel_benchmarks_root}/autotune/tests"),
#         os.listdir("."),
#     )

#     path = [
#         Path(f"{kernel_benchmarks_root}/autotune/tests/output_mpirun.txt"),
#         Path(f"{kernel_benchmarks_root}/autotune/tests/output_mpirun.pkl"),
#         Path(f"{kernel_benchmarks_root}/autotune/tests/output_mpirun.csv"),
#         Path(f"{kernel_benchmarks_root}/autotune/tests/baseline_mpirun.csv"),
#     ]

#     for p in path:
#         assert p.exists()

#     df = pd.read_csv(f"{kernel_benchmarks_root}/autotune/tests/output_mpirun.csv")
#     baseline_df = pd.read_csv(
#         f"{kernel_benchmarks_root}/autotune/tests/baseline_mpirun.csv"
#     )

#     pd.testing.assert_series_equal(df["name"], baseline_df["name"])
#     pd.testing.assert_series_equal(df["spec"], baseline_df["spec"])


def test_kbench_cache() -> None:
    _invoke_cli(
        kbench_cli,
        test_cases=[
            "-cc",
            "-cc -v",
        ],
    )


def test_kplot() -> None:
    _invoke_cli(
        kplot_cli,
        test_cases=[
            "-f --help",
            f"{kernel_benchmarks_root}/autotune/tests/output.csv -o {kernel_benchmarks_root}/autotune/tests/img_csv",
            f"{kernel_benchmarks_root}/autotune/tests/output.pkl -o {kernel_benchmarks_root}/autotune/tests/img_pkl",
            f"{kernel_benchmarks_root}/autotune/tests/output.pkl -o {kernel_benchmarks_root}/autotune/tests/img_pkl -x pdf",
        ],
    )

    path = [
        Path(f"{kernel_benchmarks_root}/autotune/tests/output.csv"),
        Path(f"{kernel_benchmarks_root}/autotune/tests/img_csv_0.png"),
        Path(f"{kernel_benchmarks_root}/autotune/tests/img_pkl_0.png"),
        Path(f"{kernel_benchmarks_root}/autotune/tests/img_pkl_0.pdf"),
    ]

    for p in path:
        assert p.exists()


def test_kplot_ytext_units() -> None:
    _invoke_cli(
        kplot_cli,
        test_cases=[
            f"{kernel_benchmarks_root}/autotune/tests/baseline.csv "
            f"-o {kernel_benchmarks_root}/autotune/tests/img_csv_us "
            "--ytext True --ytext-unit us --prec 2",
        ],
    )

    path = [
        Path(f"{kernel_benchmarks_root}/autotune/tests/img_csv_us_0.png"),
    ]

    for p in path:
        assert p.exists()


@pytest.mark.parametrize(
    ("y_title", "y_values", "requested_unit", "expected"),
    [
        # Explicit unit conversion: ms -> us
        ("met (ms)", np.array([[1.0, 2.0]]), "us", ("us", 1e3, "ms")),
        # Auto-selection based on magnitude: s with microsecond values -> us
        ("met (s)", np.array([[1e-6, 2e-6]]), "auto", ("us", 1e6, "s")),
    ],
    ids=["explicit_ms_to_us", "auto_select_us"],
)
def test_resolve_ytext_unit(
    y_title: str,
    y_values: np.ndarray,
    requested_unit: str,
    expected: tuple[str, float, str],
) -> None:
    result_title, scale, suffix, base_unit = _resolve_ytext_unit(
        y_title=y_title,
        y_values=y_values,
        requested_unit=requested_unit,
    )
    expected_suffix, expected_scale, expected_base = expected

    assert result_title.endswith(f"({expected_suffix})")
    assert suffix == expected_suffix
    assert scale == pytest.approx(expected_scale)
    assert base_unit == expected_base


def test_kprofile() -> None:
    _invoke_cli(
        kprofile_cli,
        test_cases=[
            "--help",
            f"{kernel_benchmarks_root}/autotune/tests/output.pkl",
            f"{kernel_benchmarks_root}/autotune/tests/output.pkl -c",
        ],
    )

    path = [
        Path("./correlation.png"),
    ]

    for p in path:
        assert p.exists()


# --- Scheduler tests ---


def test_scheduler_init_validates_num_gpu(tmp_path: Path) -> None:
    """num_gpu must be > 0 and <= num_cpu"""
    cache = KbenchCache()

    # num_gpu = 0 should fail
    with pytest.raises(ValueError, match="num_gpu"):
        Scheduler(
            num_cpu=4,
            num_gpu=0,
            obj_cache=cache,
            run_only=False,
            spec_list=[],
            output_dir=tmp_path,
            build_opts=[],
            dryrun=True,
        )

    # num_gpu > num_cpu should fail
    with pytest.raises(ValueError, match="num_gpu"):
        Scheduler(
            num_cpu=2,
            num_gpu=4,
            obj_cache=cache,
            run_only=False,
            spec_list=[],
            output_dir=tmp_path,
            build_opts=[],
            dryrun=True,
        )


def test_scheduler_get_chunksize(tmp_path: Path) -> None:
    """Test chunksize calculation"""
    cache = KbenchCache()
    scheduler = Scheduler(
        num_cpu=4,
        num_gpu=2,
        obj_cache=cache,
        run_only=False,
        spec_list=[],
        output_dir=tmp_path,
        build_opts=[],
        dryrun=True,
    )

    # With few elements, chunksize should be small
    assert scheduler.get_chunksize(2) == 1
    assert scheduler.get_chunksize(4) == 1
    # Chunksize is capped at CHUNK_SIZE (1)
    assert scheduler.get_chunksize(100) == 1


def test_scheduler_kbench_mkdir_creates_directory(tmp_path: Path) -> None:
    """Test that kbench_mkdir creates directories"""
    output_dir = tmp_path / "new_dir"
    assert not output_dir.exists()

    result = Scheduler.kbench_mkdir((output_dir, "output.csv", False))

    assert result == output_dir
    assert output_dir.exists()


def test_scheduler_kbench_mkdir_run_only_requires_existing(
    tmp_path: Path,
) -> None:
    """run_only=True should fail if directory doesn't exist"""
    output_dir = tmp_path / "nonexistent"

    with pytest.raises(ValueError, match="does not exist"):
        Scheduler.kbench_mkdir((output_dir, "output.csv", True))


def test_kbench_cache_basic_operations(tmp_path: Path) -> None:
    """Test KbenchCache store and query"""
    cache_path = tmp_path / "test_cache.pkl"
    cache = KbenchCache(path=cache_path)

    # Cache not active, query returns None
    assert cache.query("key1") is None

    cache.load()  # activates cache

    # Store and retrieve
    bin_path = tmp_path / "binary"
    bin_path.touch()  # create the file

    cache.store("key1", bin_path)
    assert cache.query("key1") == str(bin_path)

    # Non-existent key
    assert cache.query("nonexistent") is None


def test_process_output_log() -> None:
    """Test ProcessOutput.log doesn't crash"""
    # Just verify it doesn't raise
    output = ProcessOutput(stdout="test output", stderr="test error")
    output.log()

    empty_output = ProcessOutput()
    empty_output.log()


def test_param_define_regular() -> None:
    """Test Param.define() for regular compile-time parameters"""
    param = Param(name="BLOCK_SIZE", value=128)
    assert param.define(lang=SupportedLangs.MOJO) == ["-D", "BLOCK_SIZE=128"]

    param_str = Param(name="TYPE", value="float32")
    assert param_str.define(lang=SupportedLangs.MOJO) == ["-D", "TYPE=float32"]


def test_param_define_variable() -> None:
    """Test Param.define(lang=SupportedLangs.MOJO) for runtime variable parameters (prefixed with $)"""
    param = Param(name="$M", value=1024)
    result = param.define(lang=SupportedLangs.MOJO)
    # Variable params get converted to --<name>=<value> format
    assert result == ["--M=1024"]

    param = Param(name="$a$b", value="test")
    result = param.define(lang=SupportedLangs.MOJO)
    assert result == ["--a$b=test"]

    param = Param(name="$$a", value="test")
    result = param.define(lang=SupportedLangs.MOJO)
    assert result == ["--$a=test"]


def test_terminal_viz_render() -> None:
    """E2E test for terminal visualization output."""
    # Synthetic benchmark data
    df = pd.DataFrame(
        {
            "spec": [
                "bench_kernel/$shape=1x1x4096",
                "bench_kernel/$shape=1x512x4096",
                "bench_kernel/$shape=1x1024x4096",
            ],
            "met (s)": [0.004, 0.006, 0.010],  # 4ms, 6ms, 10ms
        }
    )

    # Test bars mode
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=80)
    render_results(df, mode="bars", console=console)
    result = output.getvalue()

    # Check that shape labels appear
    assert "1x1x4096" in result
    assert "1x512x4096" in result
    # Check that bar characters present
    assert "█" in result
    # Check that time values present
    assert "ms" in result

    # Test table mode
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=80)
    render_results(df, mode="table", console=console)
    result = output.getvalue()

    # Check that table headers
    assert "time/iter" in result
    assert "vs base" in result
    # Check that relative comparison shown
    assert "1.00x" in result


# --- GPU count detection tests ---


def _nvidia_smi_output(n: int) -> bytes:
    """Build fake nvidia-smi CSV output for n GPUs."""
    return "\n".join([f"NVIDIA H100 {i}" for i in range(n)]).encode()


def test_get_gpu_count_nvidia_basic(mocker: MockerFixture) -> None:
    """_get_gpu_count returns the hardware count when no env var is set."""
    mocker.patch("utils.get_nvidia_smi", return_value="/usr/bin/nvidia-smi")
    mocker.patch(
        "utils.subprocess.check_output",
        return_value=_nvidia_smi_output(4),
    )
    mocker.patch.dict(os.environ, {}, clear=False)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    assert _get_gpu_count("nvidia:sm_90") == 4


def test_get_gpu_count_nvidia_capped_by_env(mocker: MockerFixture) -> None:
    """When CUDA_VISIBLE_DEVICES lists more IDs than hardware, cap to hardware."""
    mocker.patch("utils.get_nvidia_smi", return_value="/usr/bin/nvidia-smi")
    mocker.patch(
        "utils.subprocess.check_output",
        return_value=_nvidia_smi_output(4),
    )
    mocker.patch.dict(
        os.environ,
        {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7,8,9"},
        clear=False,
    )
    assert _get_gpu_count("nvidia:sm_90") == 4


def test_get_gpu_count_nvidia_env_fewer_than_hw(mocker: MockerFixture) -> None:
    """When CUDA_VISIBLE_DEVICES lists fewer IDs than hardware, use env count."""
    mocker.patch("utils.get_nvidia_smi", return_value="/usr/bin/nvidia-smi")
    mocker.patch(
        "utils.subprocess.check_output",
        return_value=_nvidia_smi_output(8),
    )
    mocker.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}, clear=False)
    assert _get_gpu_count("nvidia:sm_90") == 2


def test_get_gpu_count_nvidia_no_smi(mocker: MockerFixture) -> None:
    """Returns None when nvidia-smi is not found."""
    mocker.patch("utils.get_nvidia_smi", return_value=None)
    assert _get_gpu_count("nvidia:sm_90") is None


def test_get_gpu_count_unknown_accelerator() -> None:
    """Returns None for unrecognised accelerator strings."""
    assert _get_gpu_count("metal:1") is None
    assert _get_gpu_count("") is None


def test_get_gpu_count_nvidia_smi_failure(mocker: MockerFixture) -> None:
    """Returns None when nvidia-smi fails."""
    mocker.patch("utils.get_nvidia_smi", return_value="/usr/bin/nvidia-smi")
    mocker.patch(
        "utils.subprocess.check_output",
        side_effect=subprocess.SubprocessError("boom"),
    )
    assert _get_gpu_count("nvidia:sm_90") is None


def _rocm_smi_csv_output(n: int) -> bytes:
    """Build fake rocm-smi --showproductname --csv output for n GPUs."""
    header = "device,Card series,Card model,Card vendor,Card SKU"
    rows = [
        f"card{i},0x{i:04x},AMD Instinct MI355X,Advanced Micro Devices Inc. [AMD/ATI],D7520{i}"
        for i in range(n)
    ]
    return "\n".join([header] + rows).encode()


def test_get_gpu_count_amd_basic(mocker: MockerFixture) -> None:
    """_get_gpu_count returns the hardware count for AMD GPUs."""
    mocker.patch("utils.shutil.which", return_value="/usr/bin/rocm-smi")
    mocker.patch(
        "utils.subprocess.check_output",
        return_value=_rocm_smi_csv_output(8),
    )
    mocker.patch.dict(os.environ, {}, clear=False)
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    assert _get_gpu_count("amdgpu:mi355x") == 8


def test_get_gpu_count_amd_capped_by_env(mocker: MockerFixture) -> None:
    """When ROCR_VISIBLE_DEVICES lists fewer IDs than hardware, use env count."""
    mocker.patch("utils.shutil.which", return_value="/usr/bin/rocm-smi")
    mocker.patch(
        "utils.subprocess.check_output",
        return_value=_rocm_smi_csv_output(8),
    )
    mocker.patch.dict(os.environ, {"ROCR_VISIBLE_DEVICES": "0,1"}, clear=False)
    assert _get_gpu_count("amdgpu:mi355x") == 2


def test_get_gpu_count_amd_no_smi(mocker: MockerFixture) -> None:
    """Returns None when rocm-smi is not found."""
    mocker.patch("utils.shutil.which", return_value=None)
    assert _get_gpu_count("amdgpu:mi355x") is None


def test_get_gpu_count_amd_smi_failure(mocker: MockerFixture) -> None:
    """Returns None when rocm-smi fails."""
    mocker.patch("utils.shutil.which", return_value="/usr/bin/rocm-smi")
    mocker.patch(
        "utils.subprocess.check_output",
        side_effect=subprocess.SubprocessError("boom"),
    )
    assert _get_gpu_count("amdgpu:mi355x") is None


def test_cli_rejects_num_gpu_exceeding_available(mocker: MockerFixture) -> None:
    """CLI should raise ValueError when --num-gpu exceeds detected GPUs."""
    mocker.patch("utils.get_nvidia_smi", return_value="/usr/bin/nvidia-smi")
    mocker.patch(
        "utils.subprocess.check_output",
        return_value=_nvidia_smi_output(4),
    )
    mocker.patch.dict(os.environ, {}, clear=False)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os_env = os.environ.copy()
    result = CliRunner().invoke(
        kbench_cli,
        f"{kernel_benchmarks_root}/autotune/test.yaml"
        " --num-gpu 8 --target-accelerator nvidia:sm_90 --skip-clock-check --dryrun",
        env=os_env,
    )
    assert result.exit_code != 0
    assert "exceeds" in str(result.exception) or "exceeds" in (
        result.output or ""
    )


def test_cli_accepts_num_gpu_within_available(mocker: MockerFixture) -> None:
    """CLI should not raise when --num-gpu <= detected GPUs."""
    mocker.patch("utils.get_nvidia_smi", return_value="/usr/bin/nvidia-smi")
    mocker.patch(
        "utils.subprocess.check_output",
        return_value=_nvidia_smi_output(4),
    )
    mocker.patch.dict(os.environ, {}, clear=False)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os_env = os.environ.copy()
    result = CliRunner().invoke(
        kbench_cli,
        f"{kernel_benchmarks_root}/autotune/test.yaml"
        " --num-gpu 4 --target-accelerator nvidia:sm_90 --skip-clock-check --dryrun",
        env=os_env,
    )
    assert result.exit_code == 0, result.output


def _make_spec_instance(
    compile_params: dict[str, str | int],
    runtime_params: dict[str, str | int],
) -> SpecInstance:
    """Helper to create a SpecInstance pointing at sample.mojo."""
    file = get_abs_path(f"{kernel_benchmarks_root}/autotune/sample.mojo")
    params = [Param(k, v) for k, v in compile_params.items()]
    params += [Param(f"${k}", v) for k, v in runtime_params.items()]
    return SpecInstance(
        name="test", file=file, executor=SupportedLangs.MOJO, params=params
    )


def test_build_shared_lib(tmp_path: Path) -> None:
    """build_shared_lib() compiles sample.mojo to a .so."""
    spec = _make_spec_instance({"dtype": "DType.float16"}, {"x": 0})
    result = spec.build_shared_lib(
        output_dir=tmp_path,
        build_opts=[],
    )
    assert result.return_code == os.EX_OK, result.stderr
    assert result.path is not None
    assert result.path.exists()
    assert result.path.suffix == ".so"
    # Verify wrapper file was generated
    wrapper = tmp_path / "sample_wrapper.mojo"
    assert wrapper.exists()
    content = wrapper.read_text()
    assert "from sample import main as _bench_main" in content
    assert "benchmark_entry" in content


def test_group_by_binary_hash() -> None:
    """Items with same compile-time params group together; different params form separate groups."""
    spec_a0 = _make_spec_instance({"dtype": "float16"}, {"x": 0})
    spec_a1 = _make_spec_instance({"dtype": "float16"}, {"x": 1})
    spec_b0 = _make_spec_instance({"dtype": "float32"}, {"x": 0})
    spec_b1 = _make_spec_instance({"dtype": "float32"}, {"x": 1})

    # Same compile-time params → same hash (without variables)
    assert spec_a0.hash(with_variables=False) == spec_a1.hash(
        with_variables=False
    )
    assert spec_b0.hash(with_variables=False) == spec_b1.hash(
        with_variables=False
    )
    # Different compile-time params → different hash
    assert spec_a0.hash(with_variables=False) != spec_b0.hash(
        with_variables=False
    )
