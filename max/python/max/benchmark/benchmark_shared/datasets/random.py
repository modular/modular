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

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from PIL import Image
from scipy.optimize import minimize
from scipy.stats import gamma  # type: ignore[attr-defined]
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .local import LocalBenchmarkDataset
from .types import ChatSession, SampledRequest, build_chat_message, encode_image

logger = logging.getLogger(__name__)

# Maximum ratio of model's max context length to use for random sequences.
# Set to 95% to leave buffer room for other overheads, like re-tokenization and special tokens.
MAX_CONTEXT_USAGE_RATIO = 0.95


def _parse_percentile_spec(spec: str) -> dict[float, int]:
    """Parse a percentile specification string into a dictionary.

    Args:
        spec: A string like "p5:10,p25:30,p75:91,p95:190"

    Returns:
        A dictionary mapping percentile (float) to value (int),
        e.g., {0.05: 10, 0.25: 30, 0.75: 91, 0.95: 190}
    """
    result: dict[float, int] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(
                f"Invalid percentile format: '{pair}'. Expected 'pX:Y' format."
            )
        raw_key, raw_value = pair.split(":", 1)
        key = raw_key.strip().lower()
        if not key.startswith("p"):
            raise ValueError(
                f"Invalid percentile key: '{key}'. Must start with 'p'."
            )
        try:
            percentile = float(key[1:]) / 100.0
            value = int(raw_value.strip())
        except ValueError as e:
            raise ValueError(
                f"Invalid percentile specification: '{pair}'. {e}"
            ) from e
        result[percentile] = value
    return result


def _fit_gamma_parameters(
    percentiles: dict[float, int],
) -> tuple[float, float]:
    """Fit gamma distribution parameters (shape k, scale theta) from percentile specs.

    Uses least-(relative-error-or-log-space)-squares optimization to find the
    gamma distribution parameters that best match the given percentile targets.

    Args:
        percentiles: A dictionary mapping percentile (0-1) to target value.
            e.g., {0.05: 70, 0.25: 85, 0.50: 100, 0.75: 140, 0.95: 190}
            Must contain keys 0.05, 0.5, and 0.95.

    Returns:
        A tuple (k, theta) representing the fitted gamma distribution parameters.
    """
    # Threshold for switching objective function from relative error to log-space
    OBJECTIVE_SWITCH_THRESHOLD = (
        1000  # use log-space when median percentile is above this value
    )

    if not all(k in percentiles for k in (0.05, 0.5, 0.95)):
        raise ValueError("Percentiles must contain keys 0.05, 0.5, and 0.95.")
    # Gamma distribution is always right-skewed.
    # Validate that the upper tail (50->95) is longer than the lower tail (5->50).
    if (
        percentiles[0.95] - percentiles[0.5]
        <= percentiles[0.5] - percentiles[0.05]
    ):
        raise ValueError(
            "Target percentiles are not right-skewed, which is incompatible with"
            " a gamma distribution. "
            f"(p5: {percentiles[0.05]}, p50: {percentiles[0.5]}, p95: {percentiles[0.95]})"
        )
    # Validate that values are non-decreasing with increasing percentile.
    ordered_pct = sorted(percentiles.items())
    if any(
        ordered_pct[i][1] > ordered_pct[i + 1][1]
        for i in range(len(ordered_pct) - 1)
    ):
        raise ValueError(
            "Percentile values must increase as percentiles increase."
        )

    p = np.array(list(percentiles.keys()))
    q_target = np.array(list(percentiles.values()))

    def objective(params: np.typing.NDArray[np.floating]) -> float:
        k, theta = params
        if k <= 0 or theta <= 0:
            return 1e9
        q_model = gamma.ppf(p, a=k, scale=theta)
        if np.any(q_model == 0):
            return 1e9

        if percentiles[0.5] > OBJECTIVE_SWITCH_THRESHOLD:
            errors = np.log(q_model) - np.log(q_target)
        else:
            errors = (q_model - q_target) / q_target
        return float(np.sum(errors**2))

    # Initial guess from mean and variance estimates
    # Use p50 for mean guess
    mean_guess = percentiles[0.5]
    # Estimate variance from p5-p95 range (covers ~90% of distribution)
    # In a normal distribution, p5 and p95 lie at -/+ 1.645 standard deviations
    # from the mean, so the total width (p95 - p5) is about 3.29 * sigma.
    # We divide by 4.0 (a conservative approximation for skewed distributions)
    # to get an initial sigma estimate, then square it to obtain variance.
    var_guess = ((percentiles[0.95] - percentiles[0.05]) / 4.0) ** 2

    # Gamma distribution: mean = k*theta, var = k*theta^2
    # So: theta = var/mean and k = mean/theta = mean^2/var
    k0 = mean_guess**2 / var_guess if var_guess > 0 else 1.0
    theta0 = var_guess / mean_guess if mean_guess > 0 else 1.0

    res = minimize(
        objective,
        x0=np.array([k0, theta0]),
        bounds=[(1e-6, None), (1e-6, None)],
        options={"maxiter": 2000},
    )

    k_hat, theta_hat = res.x
    logger.debug(
        f"Fitted gamma parameters: k={k_hat:.4f}, theta={theta_hat:.4f}"
    )
    # Theoretical quantiles
    q_model = gamma.ppf(p, a=k_hat, scale=theta_hat)
    logger.debug(
        f"Theoretical percentiles {list(percentiles.keys())}: {q_model.tolist()}"
    )
    return float(k_hat), float(theta_hat)


def _sample_gamma_lengths(
    percentiles: dict[float, int],
    num_samples: int,
    min_len: int,
    random_state: np.random.Generator,
) -> list[int]:
    """Sample integer lengths from a gamma distribution fitted to percentile specs.

    Args:
        percentiles: A dictionary mapping percentile (0-1) to target value.
            e.g., {0.05: 10, 0.50: 50, 0.95: 190}
            Must at least contain keys 0.05, 0.5, and 0.95.
        num_samples: Number of samples to generate.
        min_len: Minimum allowed length.
        random_state: Random state for reproducibility.

    Returns:
        A list of sampled integer lengths.
    """
    percentile_keys = sorted(percentiles.keys())
    logger.debug(
        f"Target percentiles {percentile_keys}: {[percentiles[k] for k in percentile_keys]}"
    )
    k, theta = _fit_gamma_parameters(percentiles)

    # Sample integers from the fitted gamma distribution.
    samples = gamma.rvs(
        a=k, scale=theta, size=num_samples, random_state=random_state
    )
    samples_int = np.maximum(np.round(samples).astype(int), min_len)

    # Log empirical percentiles from sampled integer lengths
    empirical_percentiles = np.percentile(
        samples_int, [p * 100 for p in percentile_keys]
    )
    logger.info(
        f"Empirical percentiles {percentile_keys}: "
        f"{empirical_percentiles.tolist()}"
    )

    return samples_int.tolist()


class RandomBenchmarkDataset(LocalBenchmarkDataset):
    def fetch(self) -> None:
        """Fetch Random dataset.

        Random datasets are generated synthetically and don't require file fetching.
        """
        pass

    def gen_multiturn_random_requests(
        self,
        input_len: int,
        output_len: int,
        num_chat_sessions: int,
        num_turns: int,
        coefficient_of_variation: str,
        tokenizer: PreTrainedTokenizerBase,
        sys_prompt_ratio: float,
        max_num_unique_sys_prompt: int,
        distribution_type: str,
        random_state: np.random.Generator,
        min_input_len: int = 4,
        min_output_len: int = 1,
        first_turn_ratio: float = 1.0,
    ) -> Sequence[ChatSession]:
        first_turns = self.sample_requests(
            num_requests=num_chat_sessions,
            tokenizer=tokenizer,
            input_len=int(input_len * first_turn_ratio),
            output_len=output_len,
            coefficient_of_variation=coefficient_of_variation,
            sys_prompt_ratio=sys_prompt_ratio,
            max_num_unique_sys_prompt=max_num_unique_sys_prompt,
            distribution_type=distribution_type,
            min_input_len=min_input_len,
            min_output_len=min_output_len,
            random_state=random_state,
        )

        follow_up_turns = self.sample_requests(
            num_requests=num_chat_sessions * (num_turns - 1),
            tokenizer=tokenizer,
            input_len=input_len,
            output_len=output_len,
            coefficient_of_variation=coefficient_of_variation,
            sys_prompt_ratio=0,
            max_num_unique_sys_prompt=1,
            distribution_type=distribution_type,
            min_input_len=min_input_len,
            min_output_len=min_output_len,
            random_state=random_state,
        )

        sessions: list[ChatSession] = []
        for session_id, first_turn in enumerate(first_turns):
            assert isinstance(first_turn.prompt_formatted, str)
            messages = [
                build_chat_message(
                    "user", first_turn.prompt_formatted, tokenizer
                ),
                build_chat_message(
                    "assistant", "", tokenizer, first_turn.output_len
                ),
            ]

            num_turns_this_session = np.random.randint(
                low=int(num_turns / 2), high=num_turns + 1
            )

            for i in range(num_turns_this_session - 1):
                follow_up_turn = follow_up_turns[
                    session_id * (num_turns - 1) + i
                ]
                assert isinstance(follow_up_turn.prompt_formatted, str)
                messages.append(
                    build_chat_message(
                        "user", follow_up_turn.prompt_formatted, tokenizer
                    )
                )
                messages.append(
                    build_chat_message(
                        "assistant", "", tokenizer, follow_up_turn.output_len
                    )
                )

            sessions.append(ChatSession(session_id, messages))

        return sessions

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> Sequence[SampledRequest]:
        # Extract required parameters from kwargs
        input_len = kwargs.get("input_len")
        output_len = kwargs.get("output_len")
        coefficient_of_variation = kwargs.get("coefficient_of_variation")
        random_state = kwargs.get("random_state")
        sys_prompt_ratio = kwargs.get("sys_prompt_ratio", 0.0)
        max_num_unique_sys_prompt = kwargs.get("max_num_unique_sys_prompt", 1)
        distribution_type = kwargs.get("distribution_type", "uniform")
        min_input_len = kwargs.get("min_input_len", 4)
        min_output_len = kwargs.get("min_output_len", 1)
        image_size = kwargs.get("image_size", "")
        image_count = kwargs.get("image_count", 0)
        model_max_length = min(
            tokenizer.model_max_length, np.iinfo(np.int64).max
        )

        # Validate required parameters
        if input_len is None:
            raise ValueError("input_len is required for RandomBenchmarkDataset")
        if output_len is None:
            raise ValueError(
                "output_len is required for RandomBenchmarkDataset"
            )
        if coefficient_of_variation is None:
            raise ValueError(
                "coefficient_of_variation is required for"
                " RandomBenchmarkDataset"
            )
        if (image_size and not image_count) or (not image_size and image_count):
            raise ValueError(
                "both image_size and image_count are required when generating"
                " an image benchmark"
            )
        if distribution_type == "gamma":
            if len(coefficient_of_variation.split(";")) != 2:
                raise ValueError(
                    "For right-skewed gamma distributions, coefficient_of_variation must"
                    " be two lists of percentiles:values separated by a semicolon"
                    " for input and output (e.g., 'p5:7000,p95:19000;p5:70,p95:190')."
                    f" Instead got: {coefficient_of_variation}"
                )
            if random_state is None:
                raise ValueError(
                    f"random_state is required for RandomBenchmarkDataset with {distribution_type} distribution"
                )

        logger.info(f"Random samples in {distribution_type} distribution")

        # Parse coefficient_of_variation based on distribution type
        if distribution_type == "gamma":  # asymmetric distribution
            # Parse percentile specifications for input and output lengths.
            input_spec, output_spec = coefficient_of_variation.split(";")
            input_percentiles = _parse_percentile_spec(input_spec)
            output_percentiles = _parse_percentile_spec(output_spec)
            # Add P50 to the input and output percentiles.
            input_percentiles[0.5] = input_len
            output_percentiles[0.5] = output_len

            # Sample input and output lengths from gamma distributions fit to percentiles.
            assert random_state is not None, (
                "random_state is required for gamma distribution"
            )
            input_lens = _sample_gamma_lengths(
                percentiles=input_percentiles,
                num_samples=num_requests,
                min_len=min_input_len,
                random_state=random_state,
            )
            output_lens = _sample_gamma_lengths(
                percentiles=output_percentiles,
                num_samples=num_requests,
                min_len=min_output_len,
                random_state=random_state,
            )

        elif distribution_type in [
            "normal",
            "uniform",
        ]:  # symmetric distribution
            if len(coefficient_of_variation.split(",")) == 2:
                input_ratio, output_ratio = map(
                    float, coefficient_of_variation.split(",")
                )
                input_scale = input_len * input_ratio
                output_scale = output_len * output_ratio
            else:
                inout_ratio = float(coefficient_of_variation)
                input_scale = input_len * inout_ratio
                output_scale = output_len * inout_ratio

            if distribution_type == "normal":
                input_lens = np.random.normal(
                    loc=input_len, scale=input_scale, size=num_requests
                ).tolist()
                input_lens = np.round(input_lens).astype(int).tolist()
                input_lens = [
                    max(input_len, min_input_len) for input_len in input_lens
                ]
                output_lens = np.random.normal(
                    loc=output_len, scale=output_scale, size=num_requests
                ).tolist()
                output_lens = np.round(output_lens).astype(int).tolist()
                output_lens = [
                    max(output_len, min_output_len)
                    for output_len in output_lens
                ]
            elif distribution_type == "uniform":
                input_scale = min(input_scale, input_len)  # full length cap
                output_scale = min(output_scale, output_len)  # full length cap
                input_lens = np.random.randint(
                    max(int(input_scale), min_input_len),
                    input_len + 1,
                    size=num_requests,
                ).tolist()
                output_lens = np.random.randint(
                    max(int(output_scale), min_output_len),
                    output_len + 1,
                    size=num_requests,
                ).tolist()
        else:
            raise ValueError(
                f"Unknown probability distribution type: {distribution_type}"
            )

        image_width, image_height = None, None
        if image_size:
            if len(image_size.split(",")) == 2:
                image_width, image_height = map(int, image_size.split(","))
            else:
                raise ValueError(
                    "Expected image size to be 2 ints separated by a comma,"
                    f" instead got: {image_size}"
                )

        vocab_size = tokenizer.vocab_size
        max_context_length = int(model_max_length * MAX_CONTEXT_USAGE_RATIO)

        sys_prompt_len = min(
            max_context_length,
            np.floor(input_len * sys_prompt_ratio).astype(int),
        )
        sys_prompts = []
        for i in range(max_num_unique_sys_prompt):  # noqa: B007
            sys_prompt = np.random.randint(0, vocab_size, size=sys_prompt_len)
            sys_prompts.append(sys_prompt.tolist())

        input_requests = []
        for i in range(num_requests):
            input_len_cur = input_lens[i]
            output_len_cur = (
                int(output_lens[i]) if output_lens[i] is not None else 0
            )
            if input_len_cur + output_len_cur > max_context_length:
                # Cap over-length sequences.
                logger.info(
                    f"Capping too long sequences ({input_len_cur} + {output_len_cur})"
                    f" > {max_context_length})..."
                )
                input_len_cur = max_context_length - output_len_cur

            sys_prompt_id = np.random.randint(0, max_num_unique_sys_prompt)
            user_prompt_offset = np.random.randint(0, vocab_size)
            user_prompt_len = input_len_cur - sys_prompt_len
            prompt_ids = sys_prompts[sys_prompt_id] + [
                (user_prompt_offset + i + j) % vocab_size
                for j in range(user_prompt_len)
            ]

            # Remove special tokens from the prompt.
            special_ids = set(tokenizer.all_special_ids)
            replacement = tokenizer.encode(" ", add_special_tokens=False)[0]
            prompt_ids = [
                (replacement if (id in special_ids) else id)
                for id in prompt_ids
            ]
            prompt = tokenizer.decode(prompt_ids)

            images = []
            image_token_len = 0
            for _ in range(image_count):
                assert image_height is not None
                assert image_width is not None
                raw_image = self._generate_random_image(
                    image_height, image_width
                )
                images.append(encode_image(raw_image))
                # TODO: figure out how to account for image tokens and chat prompts in this length.
                # For now, just hardcoding to the internvl 512x512 image token count.
                image_token_len += 256

            # We change to use the tokenizer to count the actual number of
            # input tokens encoded on the serving backends instead of looking at
            # int(input_lens[i]) that we randomly generated since multiple
            # input tokens may be bundled together in one pass
            input_len_actual = (
                len(tokenizer(prompt, add_special_tokens=False).input_ids)
                + image_token_len
            )
            input_requests.append(
                SampledRequest(
                    prompt_formatted=prompt,
                    prompt_len=input_len_actual,
                    output_len=int(output_lens[i]),
                    encoded_images=images,
                    ignore_eos=(output_lens[i] is not None),
                )
            )

        return input_requests

    def _generate_random_image(self, height: int, width: int) -> Image.Image:
        # Truly random images end up being too large and incompressible.
        # Instead create a much more limited block based random image with limited color palette.
        block_size = 16
        colors = np.array([0, 64, 128, 192, 255], dtype=np.uint8)

        blocks_h = (height + block_size - 1) // block_size
        blocks_w = (width + block_size - 1) // block_size

        # Generate colors for all blocks
        block_colors = np.random.choice(
            len(colors), size=(blocks_h, blocks_w, 3)
        )
        block_array = colors[block_colors]

        # repeat blocks to create image
        array = np.repeat(
            np.repeat(block_array, block_size, axis=0), block_size, axis=1
        )

        # crop
        array = array[:height, :width]

        return Image.fromarray(array)
