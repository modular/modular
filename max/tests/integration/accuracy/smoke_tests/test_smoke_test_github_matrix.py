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

import inspect
import json

from click.testing import CliRunner
from smoke_tests import smoke_test, smoke_test_github_matrix


def _is_valid_matrix_entry(line: str) -> bool:
    """Checks a line like `"org/model__variant": NIGHTLY | VARIANT | _1xB200,`."""
    code = line.split("#")[0].rstrip()
    if not (code.startswith('    "') and code.endswith(",") and '": ' in code):
        return False
    tags = code.split('": ', 1)[1].removesuffix(",").split(" | ")
    leading = [t for t in ("NIGHTLY", "VARIANT", "PRIVATE") if t in tags]
    return tags[: len(leading)] == leading


def test_variant_tag_iff_dunder_key() -> None:
    bad = [
        m
        for m, tags in smoke_test_github_matrix.MODELS.items()
        if ("__" in m) != ("VARIANT" in tags)
    ]
    assert not bad, (
        f"Keys with '__' must carry the VARIANT tag, and only they may: {bad}"
    )


def test_models_are_sorted() -> None:
    models = list(smoke_test_github_matrix.MODELS)
    expected = sorted(models, key=str.casefold)
    assert models == expected, (
        "MODELS must be sorted by key, ignoring case. First out of place: "
        + next(m for m, e in zip(models, expected, strict=True) if m != e)
    )


def test_matrix_entries_are_one_line_with_tags_in_order() -> None:
    source = inspect.getsource(smoke_test_github_matrix).splitlines()
    bad = []
    start = next(
        i for i, line in enumerate(source) if line.startswith("MODELS:")
    )
    for line in source[start + 1 :]:
        if line == "}":
            break
        if not _is_valid_matrix_entry(line):
            bad.append(line)
    assert not bad, (
        "Matrix entries must be exactly one line, with NIGHTLY, VARIANT and "
        "PRIVATE, if present, in that order before the HW tags:\n"
        + "\n".join(bad)
    )


def test_variants_have_recipes() -> None:
    missing = [
        k
        for k, tags in smoke_test_github_matrix.MODELS.items()
        if "VARIANT" in tags
        and "PRIVATE" not in tags
        and k not in smoke_test.MODEL_RECIPES
    ]
    assert not missing, (
        f"VARIANT models must have a corresponding entry in "
        f"smoke_test.MODEL_RECIPES: {missing}"
    )


def test_recipes_are_variants_in_matrix() -> None:
    missing = [
        k
        for k in smoke_test.MODEL_RECIPES
        if "__" in k and k not in smoke_test_github_matrix.MODELS
    ]
    assert not missing, (
        f"MODEL_RECIPES keys with '__' must have a VARIANT entry in "
        f"smoke_test_github_matrix.MODELS: {missing}"
    )


def test_nightly_models_list_hw() -> None:
    """NIGHTLY is a schedule tag, not HW: alone it runs the model nowhere."""
    bare = sorted(
        model
        for model, tags in smoke_test_github_matrix.MODELS.items()
        if not tags & set(smoke_test_github_matrix.RUNNERS)
        and "NIGHTLY" in tags
    )
    assert not bare, f"NIGHTLY models must also list the HW they run on: {bare}"


def test_nightly_8xb200_pinned() -> None:
    """Pins the nightly 8xB200 set; widen it only deliberately."""
    result = CliRunner().invoke(
        smoke_test_github_matrix.main,
        ["--framework", "max-ci", "--tier", "nightly", "--run-on-8xb200"],
    )
    assert result.exit_code == 0
    scheduled = {job["model"] for job in json.loads(result.output)["include"]}
    assert scheduled == {
        "MiniMaxAI/MiniMax-M3-MXFP8__mtp",
        "nvidia/GLM-5.2-NVFP4__mtp_tpep",
        "nvidia/Kimi-K2.7-Code-NVFP4",
        "RadixArk/GLM-5.3-NVFP4__mtp_tpep",
        "tencent/Hy4-preview-FP8",
        "tencent/Hy4-preview-FP8__mtp",
        "zai-org/GLM-5.3-Flash__tpep",
    }


def test_8xmi355_stays_opt_in() -> None:
    """8xMI355 nodes are scarce, so landing there must be a deliberate choice.

    Pinning the schedule means a model that lists 8xMI355 in its HW tags fails
    here instead of quietly consuming the runner. Widen the expectation only
    when a model is meant to run on it.
    """
    expected = {
        "max-ci": {
            "MiniMaxAI/MiniMax-M3-MXFP8",
            "modularai/MiniMax-M3-MXFP6",
            "zai-org/GLM-5.3__fp8_tpep_mi355",
        },
        "max": {
            "MiniMaxAI/MiniMax-M3-MXFP8",
            "modularai/MiniMax-M3-MXFP6",
            "zai-org/GLM-5.3__fp8_tpep_mi355",
        },
        "vllm": {
            "MiniMaxAI/MiniMax-M3-MXFP8",
            "modularai/MiniMax-M3-MXFP6",
        },
        "sglang": {
            "MiniMaxAI/MiniMax-M3-MXFP8",
            "modularai/MiniMax-M3-MXFP6",
        },
    }
    for framework, models in expected.items():
        result = CliRunner().invoke(
            smoke_test_github_matrix.main,
            ["--framework", framework, "--run-on-8xmi355"],
        )
        assert result.exit_code == 0
        scheduled = {
            job["model"] for job in json.loads(result.output)["include"]
        }
        assert scheduled == models, f"{framework} on 8xMI355"


def test_mach_skips_variants() -> None:
    """Mach serves the base repo, so a variant would run without its recipe."""
    result = CliRunner().invoke(
        smoke_test_github_matrix.main,
        ["--framework", "mach-ci", "--run-on-b200"],
    )
    assert result.exit_code == 0
    scheduled = {job["model"] for job in json.loads(result.output)["include"]}
    assert scheduled
    assert not {m for m in scheduled if "__" in m}
