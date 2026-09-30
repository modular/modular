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
"""GPQA-Diamond samples built the way OpenRouter's benchmark harness builds them.

OpenRouter publishes the harness it benchmarks provider endpoints with at
https://github.com/OpenRouterTeam/benchmark-harness. It is TypeScript and calls
the OpenRouter Responses API, so it cannot be pointed at a local server; this
module ports its GPQA sample construction (``src/benchmarks/gpqa.ts`` and
``src/benchmarks/scorers/mcq/shuffle.ts`` at :data:`HARNESS_COMMIT`) so an
Inspect task can ask a local endpoint exactly the questions it asks.

The port matters for one reason above all: each question's options are
shuffled with a permutation seeded by its row index, so the correct answers
spread over A-D. openbench's ``gpqa_diamond`` reseeds with 0 for every
question, which puts every correct answer at 'B'.

Stdlib only, so it unit-tests under bazel and imports into the harness venv.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

HARNESS = "OpenRouterTeam/benchmark-harness"
#: The harness commit this module was ported from.
HARNESS_COMMIT = "232356dc8132f9664011ee73ec2087ab958c6a66"

DATASET = "nmayorga7/gpqa_diamond"
#: The dataset commit whose row order was checked against the HF
#: datasets-server rows the harness reads. The seed is the row index, so a
#: reordered revision would silently change every question's permutation.
DATASET_REVISION = "c63e9ba02dc3da4c698e2a8485551b35041c3900"
SPLIT = "train"
NUM_QUESTIONS = 198

#: ``GPQA_META`` in ``src/benchmarks/benchmark-meta.ts``.
TEMPERATURE = 0.5
EPOCHS = 10
#: ``DEFAULT_REASONING_EFFORT`` in ``src/harness/constants.ts``, sent on every
#: request unless the run overrides it.
REASONING_EFFORT = "high"

SYSTEM_MESSAGE = "You are a helpful assistant."

PROMPT_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without "
    "quotes) where LETTER is one of ABCD.\n"
    "\n"
    "{prompt}\n"
    "\n"
    "A) {option_a}\n"
    "B) {option_b}\n"
    "C) {option_c}\n"
    "D) {option_d}"
)

#: Record fields in their original order; index 0 is the correct answer.
OPTION_FIELDS = (
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
)

_MASK32 = 0xFFFFFFFF


def _mulberry32(seed: int) -> Callable[[], float]:
    # JavaScript's bitwise operators work on 32-bit integers, so every step is
    # reduced mod 2**32. Kept unsigned throughout: `>>>` is a logical shift,
    # and Math.imul keeps the low 32 bits of the product.
    state = seed & _MASK32

    def random() -> float:
        nonlocal state
        state = (state + 1831565813) & _MASK32
        t = ((state ^ (state >> 15)) * (1 | state)) & _MASK32
        t ^= (t + (((t ^ (t >> 7)) * (61 | t)) & _MASK32)) & _MASK32
        return ((t ^ (t >> 14)) & _MASK32) / 4294967296

    return random


def _mix_seed(seed: int) -> int:
    x = (seed ^ 2654435769) & _MASK32
    x = ((x ^ (x >> 16)) * 73244475) & _MASK32
    x = ((x ^ (x >> 16)) * 73244475) & _MASK32
    return x ^ (x >> 16)


def seeded_permutation(length: int, seed: int) -> list[int]:
    """Returns the harness's Fisher-Yates permutation of ``range(length)``.

    Args:
        length: Number of items to permute.
        seed: The seed; the harness passes the question's row index.

    Returns:
        ``perm`` such that position ``i`` holds original item ``perm[i]``.
    """
    indices = list(range(length))
    random = _mulberry32(_mix_seed(seed))
    for i in range(length - 1, 0, -1):
        j = int(random() * (i + 1))
        indices[i], indices[j] = indices[j], indices[i]
    return indices


def record_to_prompt(
    record: Mapping[str, object], index: int
) -> tuple[str, str]:
    """Builds one question's user prompt and correct letter.

    Mirrors ``gpqaRecordToSample``, including its token-by-token fill: each
    placeholder is replaced once, in order, so a question that itself contains
    ``{option_a}`` renders the way the harness renders it.

    Args:
        record: One dataset row.
        index: The row's index in the split, which seeds the shuffle.

    Returns:
        The user prompt and the correct answer's letter.

    Raises:
        TypeError: If a question or option field is not a string.
    """
    question = record["Question"]
    options = [record[field] for field in OPTION_FIELDS]
    for name, value in (
        ("Question", question),
        *zip(OPTION_FIELDS, options, strict=True),
    ):
        if not isinstance(value, str):
            raise TypeError(f'gpqa record field "{name}" was not a string')
    permutation = seeded_permutation(len(OPTION_FIELDS), index)
    shuffled = [options[original] for original in permutation]
    prompt = PROMPT_TEMPLATE
    for token, value in zip(
        ("{prompt}", "{option_a}", "{option_b}", "{option_c}", "{option_d}"),
        (question, *shuffled),
        strict=True,
    ):
        prompt = prompt.replace(token, str(value), 1)
    return prompt, "ABCD"[permutation.index(0)]
