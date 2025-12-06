#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
import tempfile
from pathlib import Path

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
    SupportedLangs,
)
from kplot import cli as kplot_cli
from kprofile import cli as kprofile_cli
from utils import check_valid_target_accelerator


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


def test_scheduler_init_validates_num_gpu() -> None:
    """num_gpu must be > 0 and <= num_cpu"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = KbenchCache()

        # num_gpu = 0 should fail
        with pytest.raises(ValueError, match="num_gpu"):
            Scheduler(
                num_cpu=4,
                num_gpu=0,
                obj_cache=cache,
                run_only=False,
                spec_list=[],
                output_dir=Path(tmpdir),
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
                output_dir=Path(tmpdir),
                build_opts=[],
                dryrun=True,
            )


def test_scheduler_get_chunksize() -> None:
    """Test chunksize calculation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = KbenchCache()
        scheduler = Scheduler(
            num_cpu=4,
            num_gpu=2,
            obj_cache=cache,
            run_only=False,
            spec_list=[],
            output_dir=Path(tmpdir),
            build_opts=[],
            dryrun=True,
        )

        # With few elements, chunksize should be small
        assert scheduler.get_chunksize(2) == 1
        assert scheduler.get_chunksize(4) == 1
        # Chunksize is capped at CHUNK_SIZE (1)
        assert scheduler.get_chunksize(100) == 1


def test_scheduler_kbench_mkdir_creates_directory() -> None:
    """Test that kbench_mkdir creates directories"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "new_dir"
        assert not output_dir.exists()

        result = Scheduler.kbench_mkdir((output_dir, "output.csv", False))

        assert result == output_dir
        assert output_dir.exists()


def test_scheduler_kbench_mkdir_run_only_requires_existing() -> None:
    """run_only=True should fail if directory doesn't exist"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "nonexistent"

        with pytest.raises(ValueError, match="does not exist"):
            Scheduler.kbench_mkdir((output_dir, "output.csv", True))


def test_kbench_cache_basic_operations() -> None:
    """Test KbenchCache store and query"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.pkl"
        cache = KbenchCache(path=cache_path)

        # Cache not active, query returns None
        assert cache.query("key1") is None

        cache.load()  # activates cache

        # Store and retrieve
        bin_path = Path(tmpdir) / "binary"
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
