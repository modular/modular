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
"""Tests for the port of OpenRouter's GPQA sample construction."""

import exacto_gpqa_openrouter as gpqa
import pytest

# seededPermutation(4, i) for i in 0..197, printed by the harness's own
# shuffle.ts (commit gpqa.HARNESS_COMMIT) under node, four digits per question.
HARNESS_PERMUTATIONS = (
    "1203320101231230123001233210321010323201023102130321130210322301"
    "1203320131201032123021301230201321033210203121300312310220310123"
    "2130102321031023031202131203210332100123012321033102132001323102"
    "1023210313022031301213021023310213022301132032100321310201231320"
    "0132210320133120023113200231230132103021203110323201312003212013"
    "3120130221302310023101322013120303123120031231203201230131022013"
    "1302201310232301301232102031023102311230012303212103203132010213"
    "2103103223010213312031202130120323012301102303213210231010321023"
    "0231213012302301312010320312213003120213103203123210203130211032"
    "3102103202311320032102130231023103210312301231023120023101231230"
    "1320321012303012312010232103013223012103132031021032132030121230"
    "1230021321032130201303211023213003212031021302133012201323101032"
    "201303213201013220312310"
)

RECORD = {
    "Question": "What is 2+2?",
    "Correct Answer": "four",
    "Incorrect Answer 1": "three",
    "Incorrect Answer 2": "five",
    "Incorrect Answer 3": "six",
}


def test_permutations_match_the_harness_for_every_question() -> None:
    expected = [
        [int(d) for d in HARNESS_PERMUTATIONS[i : i + 4]]
        for i in range(0, len(HARNESS_PERMUTATIONS), 4)
    ]
    assert len(expected) == gpqa.NUM_QUESTIONS
    assert [
        gpqa.seeded_permutation(4, i) for i in range(gpqa.NUM_QUESTIONS)
    ] == expected


def test_correct_answers_spread_over_all_letters() -> None:
    # openbench's task put all 198 at 'B'; this is the property the port fixes.
    letters = [
        gpqa.record_to_prompt(RECORD, i)[1] for i in range(gpqa.NUM_QUESTIONS)
    ]
    assert {letter: letters.count(letter) for letter in "ABCD"} == {
        "A": 51,
        "B": 47,
        "C": 48,
        "D": 52,
    }


def test_prompt_puts_the_correct_answer_at_its_letter() -> None:
    prompt, letter = gpqa.record_to_prompt(RECORD, 0)
    # seededPermutation(4, 0) == [1, 2, 0, 3].
    assert letter == "C"
    assert prompt.endswith("A) three\nB) five\nC) four\nD) six")
    assert prompt.startswith(
        "Answer the following multiple choice question. The last line of "
        "your response should be of the following format: 'Answer: $LETTER' "
        "(without quotes) where LETTER is one of ABCD.\n\nWhat is 2+2?\n\n"
    )


def test_placeholders_in_the_question_fill_like_the_harness() -> None:
    # The harness replaces each token once, in order, so a literal
    # "{option_a}" in the question takes option A's text and the real slot
    # keeps its placeholder. str.format would have filled both.
    record = {**RECORD, "Question": "Pick {option_a}."}
    prompt, _ = gpqa.record_to_prompt(record, 0)
    assert "Pick three." in prompt
    assert "A) {option_a}" in prompt


def test_non_string_fields_are_rejected() -> None:
    with pytest.raises(TypeError, match="Incorrect Answer 2"):
        gpqa.record_to_prompt({**RECORD, "Incorrect Answer 2": 5}, 0)
