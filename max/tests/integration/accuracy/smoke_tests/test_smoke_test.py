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

from hf_repo_lock import load_db
from pytest import MonkeyPatch
from smoke_tests import smoke_test
from smoke_tests.smoke_test import MODEL_RECIPES


def test_hf_repo_lock_tsv_reachable() -> None:
    assert len(load_db()) > 0, "hf-repo-lock.tsv not found or empty"


def _custom_recipe_keys() -> list[str]:
    return [key for key in MODEL_RECIPES if "__" in key]


def test_model_aliases_contain_exactly_one_double_underscore() -> None:
    for alias in _custom_recipe_keys():
        count = alias.count("__")
        assert count == 1, (
            f"Model alias {alias!r} must contain exactly one '__'"
            f" (found {count})"
        )


def test_all_alias_hf_model_paths_in_hf_repo_lock() -> None:
    """Every custom recipe key's hf_model_path prefix must be pinned."""
    lock = load_db()
    missing = [
        alias
        for alias in _custom_recipe_keys()
        if alias.rsplit("__", 1)[0] not in lock
    ]
    assert not missing, (
        f"custom recipe hf_model_path prefixes missing from hf-repo-lock.tsv: {missing}"
    )


def test_all_recipe_hf_model_paths_in_hf_repo_lock() -> None:
    lock = {repo.casefold() for repo in load_db()}
    missing = []
    for recipe_path in MODEL_RECIPES.values():
        recipe = smoke_test._load_recipe(recipe_path)
        model_path = recipe.model.model_path
        assert model_path is not None
        if model_path.casefold() not in lock:
            missing.append((recipe_path, model_path))

    assert not missing, (
        f"MODEL_RECIPES model paths missing from hf-repo-lock.tsv: {missing}"
    )


def test_all_model_recipes_load() -> None:
    for alias, recipe_path in MODEL_RECIPES.items():
        recipe = smoke_test._load_recipe(recipe_path)
        assert recipe.model.model_path is not None, alias


def test_model_aliases_lookup_is_case_insensitive() -> None:
    for key in MODEL_RECIPES:
        assert MODEL_RECIPES.get(key.lower()) is not None
        assert MODEL_RECIPES.get(key.upper()) is not None


def test_recipe_aliases_preserve_key_model_path_and_speculation() -> None:
    mtp_recipe = smoke_test._load_recipe(
        MODEL_RECIPES["nvidia/DeepSeek-V3.1-NVFP4__mtp"]
    )
    assert mtp_recipe.speculative is not None
    assert mtp_recipe.speculative.num_speculative_tokens == 3

    kimi_recipe = smoke_test._load_recipe(
        MODEL_RECIPES["austinpowers/Kimi-K2.5-NVFP4-DeepseekV3__eagle"]
    )
    assert (
        kimi_recipe.model.model_path
        == "austinpowers/Kimi-K2.5-NVFP4-DeepseekV3"
    )
    assert kimi_recipe.speculative is not None
    assert kimi_recipe.speculative.num_speculative_tokens == 3


def test_recipe_gpu_overrides_scale_matching_parallelism() -> None:
    recipe = smoke_test._load_recipe(
        "deepseekV3/recipes/nvfp4_fp8kv_8x_b200.yaml"
    )

    args = smoke_test._recipe_gpu_overrides(recipe, gpu_count=4)

    assert args == [
        "--devices",
        "gpu:0,1,2,3",
        "--data-parallel-degree",
        "4",
        "--ep-size",
        "4",
    ]


def test_recipe_gpu_overrides_preserve_intentional_fixed_parallelism() -> None:
    recipe = smoke_test._load_recipe(
        "deepseekV3/recipes/nvfp4_tpep_8x_b200.yaml"
    )

    args = smoke_test._recipe_gpu_overrides(recipe, gpu_count=4)

    assert args == ["--devices", "gpu:0,1,2,3", "--ep-size", "4"]


def test_recipe_gpu_overrides_preserve_single_gpu_recipes() -> None:
    recipe = smoke_test._load_recipe("llama3/recipes/llama31_8b_eagle.yaml")

    args = smoke_test._recipe_gpu_overrides(recipe, gpu_count=4)

    assert args == []


def test_vllm_minimax_keeps_flashinfer_workaround(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        smoke_test,
        "get_gpu_name_and_count",
        lambda: ("NVIDIA B200", 8),
    )
    monkeypatch.setattr(smoke_test, "_inside_bazel", lambda: False)
    monkeypatch.setattr(smoke_test, "_load_hf_repo_lock", lambda: {})
    monkeypatch.delenv("VLLM_USE_FLASHINFER_MOE_FP8", raising=False)

    cmd = smoke_test.get_server_cmd("vllm", "MiniMaxAI/MiniMax-M2.7")

    assert "--enable-expert-parallel" in cmd
    assert "--enable-chunked-prefill" in cmd
    assert "--gpu-memory-utilization" in cmd
    assert "0.8" in cmd
    assert "--data-parallel-size=8" in cmd
    assert "--attention-backend" in cmd
    assert "FLASH_ATTN" in cmd
    assert smoke_test.os.environ["VLLM_USE_FLASHINFER_MOE_FP8"] == "0"


def test_vllm_uses_tp_for_recipe_default_data_parallel_degree(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        smoke_test,
        "get_gpu_name_and_count",
        lambda: ("NVIDIA B200", 8),
    )
    monkeypatch.setattr(smoke_test, "_inside_bazel", lambda: False)
    monkeypatch.setattr(smoke_test, "_load_hf_repo_lock", lambda: {})

    cmd = smoke_test.get_server_cmd("vllm", "nvidia/DeepSeek-V3.1-NVFP4__tpep")

    assert "--enable-expert-parallel" in cmd
    assert "--tensor-parallel-size=8" in cmd
    assert "--data-parallel-size=8" not in cmd


def test_sglang_uses_tp_for_recipe_with_tensor_parallel_attention(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        smoke_test,
        "get_gpu_name_and_count",
        lambda: ("NVIDIA B200", 8),
    )
    monkeypatch.setattr(smoke_test, "_inside_bazel", lambda: False)
    monkeypatch.setattr(smoke_test, "_load_hf_repo_lock", lambda: {})

    cmd = smoke_test.get_server_cmd(
        "sglang", "nvidia/DeepSeek-V3.1-NVFP4__tpep"
    )

    assert "--tp-size=8" in cmd
    assert "--expert-parallel-size" in cmd
    assert "8" in cmd
    assert "--mem-fraction-static" in cmd
    assert "0.8" in cmd
    assert "--data-parallel-size=8" not in cmd
    assert "--enable-dp-attention" not in cmd


def test_sglang_uses_data_parallel_attention_for_recipe_dp(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        smoke_test,
        "get_gpu_name_and_count",
        lambda: ("NVIDIA B200", 8),
    )
    monkeypatch.setattr(smoke_test, "_inside_bazel", lambda: False)
    monkeypatch.setattr(smoke_test, "_load_hf_repo_lock", lambda: {})

    cmd = smoke_test.get_server_cmd(
        "sglang", "nvidia/DeepSeek-V3.1-NVFP4__fp8kv"
    )

    assert "--data-parallel-size=8" in cmd
    assert "--enable-dp-attention" in cmd
    assert "--expert-parallel-size" in cmd
    assert "8" in cmd
    assert "--mem-fraction-static" in cmd
    assert "0.8" in cmd
    assert "--tp-size=8" not in cmd


def test_sglang_uses_recipe_memory_cap(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        smoke_test,
        "get_gpu_name_and_count",
        lambda: ("NVIDIA B200", 8),
    )
    monkeypatch.setattr(smoke_test, "_inside_bazel", lambda: False)
    monkeypatch.setattr(smoke_test, "_load_hf_repo_lock", lambda: {})

    cmd = smoke_test.get_server_cmd(
        "sglang", "austinpowers/Kimi-K2.5-NVFP4-DeepseekV3__eagle"
    )

    assert "--mem-fraction-static" in cmd
    assert "0.75" in cmd
