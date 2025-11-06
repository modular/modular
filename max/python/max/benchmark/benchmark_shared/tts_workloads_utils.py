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

import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger("tts_workloads_utils")

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class SampleTTSRequest:
    """
    Represents a single TTS request for benchmarking.
    """

    input_len_set: tuple[int, int, int]  # (transcription, tts, speech)
    tts_prompt: str
    audio_prompt_transcription: str = ""
    audio_prompt_speech_tokens: list[int] = field(default_factory=list)
    audio_prompt_speech_tokens_message: str | None = None
    output_len: int | None = None
    speaker_id: str | None = None

    def get_language(self) -> str:
        """Get the language code from speaker_id or default to 'en'."""
        if not self.speaker_id:
            return "en"
        return self.speaker_id.split("_")[0]


# -----------------------------------------------------------------------------
# Main Workload Generator Function
# -----------------------------------------------------------------------------


def generate_workloads_from_config(
    workload_config: str,
    num_prompts: int | None,
    tokenizer: PreTrainedTokenizerBase,
    shuffle_data: bool,
) -> list[SampleTTSRequest]:
    if not os.path.exists(workload_config):
        raise FileNotFoundError(
            f"Error: FILE:<{workload_config}> not found. Abort loading."
        )

    with open(workload_config) as f:
        workload_configurations = yaml.safe_load(f)
    logger.info(f"WorkloadConfig {workload_config}")

    input_requests = []
    if workload_configurations.get("tts-dataset"):  # TTS Dataset
        tts_dataset_config = workload_configurations["tts-dataset"]

        # If dataset-file is relative, then it's relative to the YAML location.
        # Convert to absolute before passing to loading the file.
        tts_dataset_jsonl_file = Path(tts_dataset_config["dataset-file"])
        tts_dataset_jsonl_file = Path(
            os.path.expandvars(tts_dataset_jsonl_file)
        )
        if not tts_dataset_jsonl_file.is_absolute():
            workload_dir = Path(workload_config).parent.resolve()
            tts_dataset_jsonl_file = (
                workload_dir / tts_dataset_config["dataset-file"]
            )

        tts_dataset = load_tts_dataset_from_jsonl(tts_dataset_jsonl_file)
        speech_to_text_ratio = tts_dataset_config.get(
            "speech-to-text-ratio", None
        )
        if num_prompts is None:
            num_prompts = len(tts_dataset)

        input_requests = sample_dataset_tts_requests(
            tts_dataset,
            num_prompts,
            speech_to_text_ratio,
            tokenizer,
            shuffle_data=shuffle_data,
        )
    elif workload_configurations.get("text-to-synthesize"):  # Synthetic
        tts_shape_config = workload_configurations["text-to-synthesize"]
        audio_shape_config = workload_configurations.get("audio-prompt", None)
        if num_prompts is None:
            num_prompts = workload_configurations["default-num-prompts"]

        input_requests = sample_synthetic_tts_requests(
            tts_shape_config, audio_shape_config, num_prompts, tokenizer
        )
    else:
        raise ValueError(
            "Failed to generate workloads. Lack `tts-dataset` or "
            f"`text-to-synthesize` in File: {workload_config}"
        )

    return input_requests


# -----------------------------------------------------------------------------
# Helper functions of
# restriving and converting data from TTS Dataset
# -----------------------------------------------------------------------------


def load_tts_dataset_from_jsonl(jsonl_filename: Path) -> list[dict[str, Any]]:
    if not os.path.exists(jsonl_filename):
        raise FileNotFoundError(
            f"Error: FILE:<{jsonl_filename}> not found. Abort loading."
        )

    tts_dataset = []
    with open(jsonl_filename) as f:
        for line in f:
            data = json.loads(line.strip())
            tts_dataset.append(data)
    return tts_dataset


def sample_dataset_tts_requests(
    tts_dataset: list[dict[str, Any]],
    num_prompts: int,
    speech_to_text_ratio: int | None,
    tokenizer: PreTrainedTokenizerBase,
    shuffle_data: bool = True,
) -> list[SampleTTSRequest]:
    if len(tts_dataset) < num_prompts:
        raise ValueError(
            f"TTS dataset only contains {len(tts_dataset)} samples, less than "
            f"{num_prompts}, requested. Decrease num_prompts or Consider "
            "switching to a larger dataset."
        )
    tts_dataset_part = tts_dataset[:num_prompts]

    logger.info("Converting into SampleTTSRequest formats...")
    # Convert into SampleTTSRequest formats
    input_requests = []
    for data in tts_dataset_part:
        tts_prompt = data["tts_prompt"]
        audio_prompt_transcription = data["audio_prompt_transcription"]
        audio_prompt_tokens = data["audio_prompt_tokens"]
        speaker_id = data.get("speaker_id")

        audio_prompt_transcription_len = len(
            tokenizer(
                audio_prompt_transcription, add_special_tokens=False
            ).input_ids
        )
        tts_prompt_len = len(
            tokenizer(tts_prompt, add_special_tokens=False).input_ids
        )
        audio_prompt_tokens_len = len(audio_prompt_tokens)

        input_len_set = (
            audio_prompt_transcription_len,
            tts_prompt_len,
            audio_prompt_tokens_len,
        )

        output_len = None
        if speech_to_text_ratio:
            output_len = tts_prompt_len * speech_to_text_ratio

        input_requests.append(
            SampleTTSRequest(
                input_len_set=input_len_set,
                output_len=output_len,
                tts_prompt=data["tts_prompt"],
                audio_prompt_transcription=data["audio_prompt_transcription"],
                audio_prompt_speech_tokens=data["audio_prompt_tokens"],
                speaker_id=speaker_id,
            )
        )

    # Shuffle the combinations
    if shuffle_data:
        random.shuffle(input_requests)

    return input_requests


# -----------------------------------------------------------------------------
# Helper functions of
# restriving/generating samples from synthetic random data
# -----------------------------------------------------------------------------


def get_random_prompt_in_text(
    num_tokens: int,
    tokenizer: PreTrainedTokenizerBase,
    vocab_size: int,
    token_offset: int = 0,
) -> tuple[str, int]:
    """
    This function constructs a list of num_tokens consecutive numbers, beginning with
    a random base plus token_offset and increasing by 1 with each step. Then,
    detokenize into text for the return together with an actual prompt length for the
    re-tokenization.
    """
    token_base = np.random.randint(0, vocab_size) + token_offset

    prompt_in_text = tokenizer.decode(
        [(token_base + j) % vocab_size for j in range(num_tokens)]
    )
    prompt_len_actual = len(
        tokenizer(prompt_in_text, add_special_tokens=False).input_ids
    )
    return (prompt_in_text, prompt_len_actual)


def sample_synthetic_tts_requests(
    tts_shape_config: dict[str, Any],
    audio_shape_config: dict[str, Any] | None,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> list[SampleTTSRequest]:
    # Prepare shape distribution
    text_to_synthesize_lens = np.random.normal(
        loc=tts_shape_config["mean"],
        scale=tts_shape_config["stddev"],
        size=num_requests,
    )
    text_to_synthesize_lens = np.clip(
        text_to_synthesize_lens,
        a_min=tts_shape_config["min"],
        a_max=tts_shape_config["max"],
    )
    if tts_shape_config.get("speech-to-text-ratio"):
        generated_audio_tokens_lens = (
            text_to_synthesize_lens * tts_shape_config["speech-to-text-ratio"]
        )

    audio_prompt_transcription_lens: npt.NDArray[np.float64]
    if (
        audio_shape_config is None or audio_shape_config["mean"] == 0
    ):  # No voice cloning
        audio_prompt_transcription_lens = np.zeros(num_requests)
        audio_prompt_speech_tokens_lens = np.zeros(num_requests)
    else:
        audio_prompt_transcription_lens = np.random.normal(
            loc=audio_shape_config["mean"],
            scale=audio_shape_config["stddev"],
            size=num_requests,
        )
        audio_prompt_transcription_lens = np.clip(
            audio_prompt_transcription_lens,
            a_min=audio_shape_config["min"],
            a_max=audio_shape_config["max"],
        )
        audio_prompt_speech_tokens_lens = (
            audio_prompt_transcription_lens
            * audio_shape_config["speech-to-text-ratio"]
        )

    # Generate requests
    vocab_size = tokenizer.vocab_size
    input_requests: list[SampleTTSRequest] = []
    for i in range(num_requests):
        # Generate prompts and get the actual length after re-tokenization
        audio_prompt_transcription, audio_prompt_transcription_len_actual = (
            get_random_prompt_in_text(
                num_tokens=int(audio_prompt_transcription_lens[i]),
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                token_offset=i,
            )
        )

        tts_prompt, tts_prompt_len_actual = get_random_prompt_in_text(
            num_tokens=int(text_to_synthesize_lens[i]),
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            token_offset=i,
        )

        (
            audio_prompt_speech_tokens_message,
            audio_prompt_speech_tokens_len_actual,
        ) = get_random_prompt_in_text(
            num_tokens=int(audio_prompt_speech_tokens_lens[i]),
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            token_offset=i,
        )

        # For voice cloning with PrependPromptSpeechTokens != Never, we need to
        # set *both* audio_prompt_tokens *and* _assistant_message_override in
        # the request. So we take the string that we just generated through
        # reverse tokenization, and re-tokenize it once more to get the array of
        # integers back. This is messy & ineffecient; see PAQ-992
        audio_prompt_speech_tokens = tokenizer(
            audio_prompt_speech_tokens_message, add_special_tokens=False
        ).input_ids
        assert (
            len(audio_prompt_speech_tokens)
            == audio_prompt_speech_tokens_len_actual
        ), "Unexpected length after re-tokenization"

        speech_lm_input_len_set_actual = (
            audio_prompt_transcription_len_actual,
            tts_prompt_len_actual,
            audio_prompt_speech_tokens_len_actual,
        )

        output_len = None
        if tts_shape_config.get("speech-to-text-ratio"):
            output_len = int(generated_audio_tokens_lens[i])

        # Pack into a request.
        # Synthetic benchmark uses message in text as the replacement of speech token pattern
        input_requests.append(
            SampleTTSRequest(
                tts_prompt=tts_prompt,
                audio_prompt_speech_tokens=audio_prompt_speech_tokens,
                audio_prompt_speech_tokens_message=audio_prompt_speech_tokens_message,
                audio_prompt_transcription=audio_prompt_transcription,
                input_len_set=speech_lm_input_len_set_actual,
                output_len=output_len,
            )
        )

    return input_requests
