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
"""Sweep-mode kbench: one process per params group.

A `sweep: true` spec does not expand the product of its `$` parameters into
one process per point. Each params group collapses to the single process
that runs it, with every `$` parameter as one comma-joined list, and the
benchmark expands the lists into its own configs. Compile-time parameters
pin a single value; the benchmark's own `comptime for` arms cover the rest.
"""

import os
import string
from pathlib import Path

import pandas as pd
import pytest
from kbench_model import Param, Scheduler, Spec, SpecInstance, SupportedLangs

kernel_benchmarks_root = os.environ["KERNEL_BENCHMARKS_ROOT"]


def _fixture_path() -> Path:
    return Path(
        string.Template(
            f"{kernel_benchmarks_root}/autotune/tests/minimal_fixture.mojo"
        ).substitute(os.environ)
    ).absolute()


def _sweep_spec(yaml_params: str) -> Spec:
    return Spec.loads(
        f"""
name: sweep_fixture
file: {_fixture_path()}
sweep: true

params:
{yaml_params}
"""
    )


def test_sweep_collapses_each_group_to_one_process() -> None:
    spec = _sweep_spec(
        """
- num_heads: 8
  page_size: 128
  $seq_len: [6, 8192]
  $cache_len: [32768, 0]
- num_heads: 8
  page_size: 64
  $seq_len: [1, 6]
  $cache_len: [2042]
"""
    )
    assert len(spec) == 2
    first, second = spec.mesh
    for instance in spec.mesh:
        assert instance.sweep
    assert first.params == [
        Param("num_heads", 8),
        Param("page_size", 128),
        Param("$seq_len", "6,8192"),
        Param("$cache_len", "32768,0"),
    ]
    assert first._get_vars == ["--seq_len=6,8192", "--cache_len=32768,0"]
    assert first._get_defines == ["-D", "num_heads=8", "-D", "page_size=128"]
    # The build hash carries the pins only, so each group builds once.
    assert first.hash(with_variables=False) == (
        "minimal_fixture_num_heads-8_page_size-128"
    )
    assert second.hash(with_variables=False) == (
        "minimal_fixture_num_heads-8_page_size-64"
    )
    # Groups that share pins share one build.
    same_pins = _sweep_spec(
        """
- num_heads: 8
  $cache_len: [2042]
- num_heads: 8
  $cache_len: [4096]
"""
    )
    assert len(same_pins) == 2
    assert len({i.hash(with_variables=False) for i in same_pins.mesh}) == 1


def test_sweep_rejects_multi_valued_comptime_param() -> None:
    with pytest.raises(ValueError, match="sweep parameter 'q_dtype'"):
        _sweep_spec(
            """
- num_heads: 8
  q_dtype: [float8_e4m3fn, bfloat16]
  $cache_len: [2042]
"""
        )


def test_sweep_rejects_empty_axis_and_missing_groups() -> None:
    with pytest.raises(ValueError, match="has no values"):
        _sweep_spec(
            """
- $cache_len: [2042]
  $unused:
"""
        )
    with pytest.raises(ValueError, match="at least one params group"):
        Spec.loads(
            f"""
name: sweep_fixture
file: {_fixture_path()}
sweep: true
"""
        )


def test_sweep_rejects_rules() -> None:
    with pytest.raises(ValueError, match="rules are not supported"):
        Spec.loads(
            f"""
name: sweep_fixture
file: {_fixture_path()}
sweep: true
rules:
- cache_len > 1000

params:
- $cache_len: [512, 2042]
"""
        )


def test_sweep_filter_that_empties_the_run_warns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = _sweep_spec(
        """
- $cache_len: [512, 2042]
"""
    )
    # A filter compares against the joined list, not each value, so this
    # eliminates the run rather than restricting it.
    spec.filter(["cache_len=512"])
    assert len(spec) == 0
    err = capsys.readouterr().err
    assert "eliminated all 1 instances" in err
    assert "restrict one with --param instead" in err


def test_sweep_param_restricts_axes_and_pins_arms() -> None:
    spec = _sweep_spec(
        """
- num_heads: 8
  $seq_len: [6, 8192]
  $cache_len: [512, 2042, 4096]
"""
    )
    spec.extend_params(["cache_len:[2042]"])
    assert len(spec) == 1
    instance = spec.mesh[0]
    assert instance.sweep
    assert Param("$cache_len", "2042") in instance.params

    # A single-value comptime param pins one arm, and a new `$` param (a
    # benchmark flag the YAML does not declare) rides along as a list of
    # one, which the benchmark parses as a single value.
    spec = _sweep_spec(
        """
- $cache_len: [512, 2042, 4096]
"""
    )
    spec.extend_params(["q_dtype:[bfloat16]", "$target_time:[0.2]"])
    instance = spec.mesh[0]
    assert Param("q_dtype", "bfloat16") in instance.params
    assert instance._get_defines == ["-D", "q_dtype=bfloat16"]
    assert Param("$target_time", "0.2") in instance.params
    assert instance._get_vars == [
        "--cache_len=512,2042,4096",
        "--target_time=0.2",
    ]


def test_sweep_build_adds_the_benchmark_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sweep_instance = SpecInstance(
        name="sweep_fixture",
        file=_fixture_path(),
        executor=SupportedLangs.MOJO,
        params=[Param("$cache_len", "2042")],
        sweep=True,
    )
    sweep_instance.build(output_dir=tmp_path, build_opts=[], dryrun=True)
    sweep_cmd = capsys.readouterr().out
    assert f"-I {_fixture_path().parent} " in sweep_cmd

    plain_instance = SpecInstance(
        name="fixture",
        file=_fixture_path(),
        executor=SupportedLangs.MOJO,
        params=[Param("num_heads", 8)],
    )
    plain_instance.build(output_dir=tmp_path, build_opts=[], dryrun=True)
    assert "-I" not in capsys.readouterr().out


def test_sweep_multi_row_csv_loads_one_row_per_benchmark(
    tmp_path: Path,
) -> None:
    instance = SpecInstance(
        name="sweep_fixture",
        file=_fixture_path(),
        executor=SupportedLangs.MOJO,
        params=[
            Param("num_heads", 8),
            Param("$cache_len", "512,2042"),
        ],
        sweep=True,
    )
    csv_path = tmp_path / "output.csv"
    csv_path.write_text(
        "name,met (ms),iters\n"
        '"bench : cache_len=512",0.5,100\n'
        '"bench : cache_len=2042",0.6,100\n'
        '"bench : cache_len=2042, spread",0.7,100\n'
    )
    frames = Scheduler.load_csv_to_pd(
        mesh_idx=0, current_spec=instance, files=[csv_path]
    )
    df = pd.concat(frames, ignore_index=True)
    assert len(df) == 3
    assert list(df["name"]) == [
        "bench : cache_len=512",
        "bench : cache_len=2042",
        "bench : cache_len=2042, spread",
    ]
    # Every row of one sweep process carries the process's spec and filter.
    assert df["spec"].nunique() == 1
    assert (df["kbench_filter"] == "num_heads=8 cache_len=512,2042").all()


def test_sweep_join_requires_matching_mode() -> None:
    spec = _sweep_spec(
        """
- $cache_len: [2042]
"""
    )
    plain = Spec.loads(
        f"""
name: sweep_fixture
file: {_fixture_path()}

params:
- $cache_len: [2042]
"""
    )
    with pytest.raises(AssertionError, match="sweep spec"):
        spec.join(plain)
