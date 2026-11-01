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
"""Tests GPU selection and replay filtering for collective fuzz targets."""

import json
from pathlib import Path

import fuzz
import pytest


@pytest.mark.parametrize(
    "value,expected", [("0", "0"), ("2,4", "2,4"), (" 0, 02 ", "0,2")]
)
def test_gpu_list(value: str, expected: str) -> None:
    assert fuzz._parse_args(["--gpu", value]).gpu == expected


@pytest.mark.parametrize("value", ["", "0,", "-1", "0,0", "0,00", "abc", "1.5"])
def test_invalid_gpu_list(value: str) -> None:
    with pytest.raises(SystemExit):
        fuzz._parse_args(["--gpu", value])


@pytest.mark.parametrize(
    "oracle", ["diff", "ref", "poison", "redzone", "memcheck"]
)
def test_oracle_preserves_device_list(oracle: str) -> None:
    _, env = fuzz._oracle_command_and_env(oracle, ["kernel"], "2,4")
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
    ):
        assert env[name] == "2,4"


def test_collective_requires_multiple_devices(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert fuzz.main(["--target", "allgather_rows_slice", "--gpu", "0"]) == 1
    assert "requires at least 2 devices" in capsys.readouterr().err


@pytest.mark.parametrize("explicit,expected", [(False, 0), (True, 1)])
def test_replay_with_insufficient_devices(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    explicit: bool,
    expected: int,
) -> None:
    target = "allgather_rows_slice"
    corpus = tmp_path / target
    corpus.mkdir()
    (corpus / "pass.json").write_text(json.dumps({"target": target}))
    assert (
        fuzz.replay_corpus(
            tmp_path,
            tmp_path,
            target if explicit else None,
            "0",
            30,
            do_build=False,
        )
        == expected
    )
    assert "requires at least 2 devices" in capsys.readouterr().out
