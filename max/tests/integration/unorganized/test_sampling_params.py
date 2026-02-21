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

from max.interfaces import (
    SamplingParams,
    SamplingParamsGenerationConfigDefaults,
    SamplingParamsInput,
)


def test_sampling_params_from_input_and_generation_config_defaults_override() -> (
    None
):
    """Test that SamplingParamsGenerationConfigDefaults values override SamplingParams class defaults."""
    # Create defaults that override some SamplingParams class defaults
    generation_defaults = SamplingParamsGenerationConfigDefaults(
        temperature=0.5,
        top_k=50,
        max_new_tokens=100,
    )

    # Create SamplingParams with no user overrides
    sampling_params = SamplingParams.from_input_and_generation_config(
        SamplingParamsInput(),
        sampling_params_defaults=generation_defaults,
    )

    # Verify that generation config defaults override SamplingParams class defaults
    assert (
        sampling_params.temperature == 0.5
    )  # from generation_defaults, not 1.0 (class default)
    assert (
        sampling_params.top_k == 50
    )  # from generation_defaults, not -1 (class default)
    assert (
        sampling_params.max_new_tokens == 100
    )  # from generation_defaults, not None (class default)

    # Verify that fields not in generation_defaults retain their class defaults
    assert sampling_params.top_p == 1  # class default
    assert sampling_params.min_p == 0.0  # class default
    assert sampling_params.frequency_penalty == 0.0  # class default
    assert sampling_params.repetition_penalty == 1.0  # class default


def test_sampling_params_from_input_and_generation_config_user_override() -> (
    None
):
    """Test that user-provided values take highest priority over defaults."""
    # Create generation defaults
    generation_defaults = SamplingParamsGenerationConfigDefaults(
        temperature=0.5,
        top_k=50,
        top_p=0.9,
        max_new_tokens=100,
    )

    # Create SamplingParams with user overrides
    user_input = SamplingParamsInput(
        temperature=0.8,  # Override generation default (0.5)
        top_k=10,  # Override generation default (50)
        min_new_tokens=5,  # Not in generation defaults, overrides class default (0)
    )

    sampling_params = SamplingParams.from_input_and_generation_config(
        user_input,
        sampling_params_defaults=generation_defaults,
    )

    # Verify user values take highest priority
    assert (
        sampling_params.temperature == 0.8
    )  # user value, not generation default (0.5)
    assert (
        sampling_params.top_k == 10
    )  # user value, not generation default (50)
    assert (
        sampling_params.min_new_tokens == 5
    )  # user value, not class default (0)

    # Verify generation defaults are used when user doesn't override
    assert sampling_params.top_p == 0.9  # from generation_defaults
    assert sampling_params.max_new_tokens == 100  # from generation_defaults

    # Verify class defaults are used when neither user nor generation defaults provide values
    assert sampling_params.min_p == 0.0  # class default
    assert sampling_params.frequency_penalty == 0.0  # class default


def test_sampling_params_none_top_p_normalized_to_default() -> None:
    """Test that top_p=None is normalized to the default value (1.0).

    When a user sends top_p=null in an API request, it should be treated
    the same as not specifying top_p at all, which means using the default
    of 1.0.  This keeps the gumbel sampling fast path reachable.
    """
    # Direct construction with None should normalize to 1.0
    params = SamplingParams(top_p=None)  # type: ignore[arg-type]
    assert params.top_p == 1.0

    # Through from_input_and_generation_config with no generation defaults
    params = SamplingParams.from_input_and_generation_config(
        SamplingParamsInput(top_p=None),
        sampling_params_defaults=SamplingParamsGenerationConfigDefaults(),
    )
    assert params.top_p == 1.0


def test_sampling_params_none_top_k_normalized_to_default() -> None:
    """Test that top_k=None is normalized to the default value (-1).

    When a user sends top_k=null in an API request, it should be treated
    the same as not specifying top_k at all, which means using the default
    of -1 (sample all tokens).
    """
    # Direct construction with None should normalize to -1
    params = SamplingParams(top_k=None)  # type: ignore[arg-type]
    assert params.top_k == -1

    # Through from_input_and_generation_config with no generation defaults
    params = SamplingParams.from_input_and_generation_config(
        SamplingParamsInput(top_k=None),
        sampling_params_defaults=SamplingParamsGenerationConfigDefaults(),
    )
    assert params.top_k == -1


def test_sampling_params_from_input_not_shared_across_calls() -> None:
    """Test that calling from_input_and_generation_config twice with different
    user inputs produces independent results (cached dict not mutated)."""
    generation_defaults = SamplingParamsGenerationConfigDefaults(
        temperature=0.7,
        top_k=50,
    )

    # First request: user overrides temperature
    params1 = SamplingParams.from_input_and_generation_config(
        SamplingParamsInput(temperature=0.1),
        sampling_params_defaults=generation_defaults,
    )

    # Second request: user provides no overrides
    params2 = SamplingParams.from_input_and_generation_config(
        SamplingParamsInput(),
        sampling_params_defaults=generation_defaults,
    )

    assert params1.temperature == 0.1
    # This must be 0.7 (generation default), not 0.1 (leaked from first call)
    assert params2.temperature == 0.7
