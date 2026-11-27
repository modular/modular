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
"""Pins which JSON Schema requests are served and which are refused.

Serve cases must return conformant content, as response_format and, where a
case asks, as the arguments of a named tool. Refuse cases cover unsatisfiable
or unenforceable schemas. Reasoning is disabled so the verdict
reflects the compiler rather than model behavior.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from jsonschema import Draft7Validator, Draft202012Validator

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario
from scenarios._constrained_stream import classify_served_json, grade_admission
from scenarios._schema_admission_cases import (
    CASES,
    DRAFT_2020_12,
    REFUSE,
    SERVE,
    AdmissionCase,
)

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

_N_PER_CASE = 3
_MAX_OUTPUT_TOKENS = 512


_GRADE_TO_VERDICT = {
    "pass": Verdict.PASS,
    "fail": Verdict.FAIL,
    "interesting": Verdict.INTERESTING,
}


def _payload(
    model: str, name: str, schema: dict[str, object], prompt: str
) -> dict[str, object]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return ONLY a JSON object."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": schema},
        },
        "max_tokens": _MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
    }


def _tool_payload(
    model: str, name: str, schema: dict[str, object], prompt: str
) -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "Record the requested value.",
                    "parameters": schema,
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": name}},
        "max_tokens": _MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
    }


def _classify_tool_arguments(
    body: str, case: AdmissionCase
) -> tuple[str, str | None]:
    """Grades every returned tool call as ``classify_served_json`` grades
    content: each must name ``case.name`` and carry conformant arguments.

    A length finish with no call is ``truncated``. A call's arguments keep
    the finish reason, so a length-finished call whose arguments overran the
    grammar is ``invalid``, not ``truncated``.
    """
    try:
        choice = json.loads(body)["choices"][0]
        finish = choice.get("finish_reason")
        calls = choice["message"].get("tool_calls") or []
    except (
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
        AttributeError,
    ):
        return "invalid", "served but body was not a chat completion"
    if not calls:
        if finish == "length":
            return "truncated", None
        return "invalid", "served but carried no tool call"
    grades: list[tuple[str, str | None]] = []
    for index, call in enumerate(calls):
        try:
            name = call["function"]["name"]
            arguments = call["function"]["arguments"]
        except (KeyError, TypeError):
            return "invalid", f"tool_calls[{index}] is malformed"
        if name != case.name:
            return "invalid", (
                f"tool_calls[{index}] names {name!r}, expected {case.name!r}"
            )
        content = {
            "choices": [
                {"message": {"content": arguments}, "finish_reason": finish}
            ]
        }
        grades.append(
            classify_served_json(
                json.dumps(content), case.schema, _validator(case)
            )
        )
    for grade in grades:
        if grade[0] == "invalid":
            return grade
    for grade in grades:
        if grade[0] == "truncated":
            return grade
    return "conformant", None


def _validator(
    case: AdmissionCase,
) -> type[Draft7Validator | Draft202012Validator]:
    return (
        Draft202012Validator if case.draft == DRAFT_2020_12 else Draft7Validator
    )


@register_scenario
class SchemaAdmissionPolicy(BaseScenario):
    name = "schema_admission_policy"
    description = (
        "Pins admission for multi-byte text, enum siblings, and schemas that "
        "are unenforceable, plus shapes the grammar compiler once refused"
    )
    tags = [
        "structured",
        "json",
        "schema",
        "admission",
        "regression",
        "correctness",
    ]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        case_verdicts: list[Verdict] = []
        summary_parts: list[str] = []

        runs = [(case, case.name, False) for case in CASES] + [
            (case, f"{case.name}_as_tool", True)
            for case in CASES
            if case.as_tool
        ]
        for case, label, as_tool in runs:
            refused = served = conformant = truncated = 0
            other: list[str] = []
            build = _tool_payload if as_tool else _payload
            for _ in range(_N_PER_CASE):
                resp = await client.post_json(
                    build(config.model, case.name, case.schema, case.prompt),
                    timeout=config.timeout * 2,
                )
                if resp.status == 400:
                    refused += 1
                    if case.expect == SERVE and len(other) < 2:
                        other.append(f"400: {(resp.body or '')[:160]!r}")
                    continue
                if resp.status != 200 or resp.error:
                    if len(other) < 2:
                        other.append(f"status={resp.status} err={resp.error}")
                    continue
                served += 1
                grade, detail = (
                    _classify_tool_arguments(resp.body, case)
                    if as_tool
                    else classify_served_json(
                        resp.body, case.schema, _validator(case)
                    )
                )
                if grade == "truncated":
                    truncated += 1
                elif grade == "conformant":
                    conformant += 1
                elif len(other) < 2 and detail:
                    other.append(detail)

            judged = served - truncated
            fv = _GRADE_TO_VERDICT[
                grade_admission(
                    case.expect,
                    _N_PER_CASE,
                    refused,
                    served,
                    conformant,
                    truncated,
                )
            ]
            if case.expect == REFUSE:
                detail = (
                    f"{label}: refused {refused}/{_N_PER_CASE}"
                    f" (expected: {case.why})"
                )
                summary_parts.append(
                    f"{label}=[refused {refused}/{_N_PER_CASE}]"
                )
            elif served == _N_PER_CASE and truncated == _N_PER_CASE:
                detail = (
                    f"{label}: served {served}/{_N_PER_CASE},"
                    f" truncated {truncated} (expected: {case.why})"
                )
                summary_parts.append(
                    f"{label}=[served {served}/{_N_PER_CASE}, truncated]"
                )
            else:
                detail = (
                    f"{label}: served {served}/{_N_PER_CASE},"
                    f" conformant {conformant}/{judged}"
                    f" (expected: {case.why})"
                )
                summary_parts.append(
                    f"{label}=[served {served}/{_N_PER_CASE},"
                    f" conf {conformant}/{judged}]"
                )
            case_verdicts.append(fv)
            print(f"    {detail}")
            for ex in other:
                print(f"      {ex}")
            results.append(
                self.make_result(self.name, f"case_{label}", fv, detail=detail)
            )

        summary = " ".join(summary_parts)
        print(f"    SUMMARY: {summary}")
        if Verdict.FAIL in case_verdicts:
            overall = Verdict.FAIL
        elif Verdict.INTERESTING in case_verdicts:
            overall = Verdict.INTERESTING
        else:
            overall = Verdict.PASS
        results.append(
            self.make_result(
                self.name, "schema_admission_policy", overall, detail=summary
            )
        )

        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_schema_admission_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
            )
        )
        return results
