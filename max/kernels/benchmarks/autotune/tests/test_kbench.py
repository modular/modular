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

# Run: bazel test open-source/max/max/kernels/benchmarks/autotune:autotune_tests

import os
import string
from pathlib import Path

import pandas as pd
from click.core import Command
from click.testing import CliRunner
from kbench import cli as kbench_cli
from kplot import cli as kplot_cli
from kprofile import cli as kprofile_cli


def get_abs_path(path: str) -> Path:
    return Path(string.Template(str(path)).substitute(os.environ)).absolute()


kernel_benchmarks_root = get_abs_path("open-source/max/max/kernels/benchmarks/")
os.environ["KERNEL_BENCHMARKS_ROOT"] = str(kernel_benchmarks_root)


# TODO: refactor to match the expected results
def _invoke_cli(
    cli: Command,
    test_cases: list[str],
    exit_code: int = os.EX_OK,
):
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


def test_kbench():
    _invoke_cli(
        kbench_cli,
        test_cases=[
            "-f --help",
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


def test_kbench_cache():
    print("here")
    _invoke_cli(
        kbench_cli,
        test_cases=[
            "-cc",
            "-cc -v",
        ],
    )


def test_kplot():
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


def test_kprofile():
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
