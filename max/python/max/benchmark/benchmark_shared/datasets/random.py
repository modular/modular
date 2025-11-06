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
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .local import LocalBenchmarkDataset
from .types import ChatSession, SampledRequest, build_chat_message, encode_image

logger = logging.getLogger(__name__)


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
        sys_prompt_ratio = kwargs.get("sys_prompt_ratio", 0.0)
        max_num_unique_sys_prompt = kwargs.get("max_num_unique_sys_prompt", 1)
        distribution_type = kwargs.get("distribution_type", "uniform")
        min_input_len = kwargs.get("min_input_len", 4)
        min_output_len = kwargs.get("min_output_len", 1)
        image_size = kwargs.get("image_size", "")
        image_count = kwargs.get("image_count", 0)

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

        logger.info(f"Random samples in {distribution_type} distribution")

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

        image_width, image_height = None, None
        if image_size:
            if len(image_size.split(",")) == 2:
                image_width, image_height = map(int, image_size.split(","))
            else:
                raise ValueError(
                    "Expected image size to be 2 ints separated by a comma,"
                    f" instead got: {image_size}"
                )

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
                max(output_len, min_output_len) for output_len in output_lens
            ]
        elif distribution_type == "uniform":
            input_scale = min(input_scale, input_len)  # full length cap
            output_scale = min(output_scale, output_len)  # full length cap
            input_lens = np.random.randint(
                max(int(input_scale), min_input_len),
                input_len + 1,
                size=num_requests,
            )
            output_lens = np.random.randint(
                max(int(output_scale), min_output_len),
                output_len + 1,
                size=num_requests,
            )
        else:
            raise ValueError(
                f"Unknown probability distribution type: {distribution_type}"
            )

        vocab_size = tokenizer.vocab_size

        sys_prompt_len = np.floor(input_len * sys_prompt_ratio).astype(int)
        sys_prompts = []
        for i in range(max_num_unique_sys_prompt):  # noqa: B007
            sys_prompt = np.random.randint(0, vocab_size, size=sys_prompt_len)
            sys_prompts.append(sys_prompt.tolist())

        input_requests = []
        for i in range(num_requests):
            sys_prompt_id = np.random.randint(0, max_num_unique_sys_prompt)
            user_prompt_offset = np.random.randint(0, vocab_size)
            user_prompt_len = input_lens[i] - sys_prompt_len
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
