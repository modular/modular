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
"""Tool calling schema enforcement: constrained decoding and JSON schema support.

Verifies that tool calls match the requested tool call schema, and that the
complete JSON schema specification is supported.

Each test sends a tool-calling request with tool_choice=required and a
carefully designed schema, then validates the tool calls against
the requested schema.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from helpers import budget_exhausted, make_tool

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


def _validate_args(
    args: dict[str, Any],
    schema: dict[str, Any],
) -> list[str]:
    """Validate tool call args against a JSON schema. Returns error strings."""
    errors: list[str] = []
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    for req in required:
        if req not in args:
            errors.append(f"missing required field: {req}")

    for key, val in args.items():
        if key not in props:
            continue
        prop_schema = props[key]

        if "enum" in prop_schema:
            if val not in prop_schema["enum"]:
                errors.append(
                    f"{key}: {val!r} not in enum {prop_schema['enum']}"
                )

        expected_type = prop_schema.get("type")
        if expected_type == "integer":
            if not isinstance(val, int) or isinstance(val, bool):
                errors.append(
                    f"{key}: expected integer, got {type(val).__name__}"
                )
            if isinstance(val, float) and val != int(val):
                errors.append(f"{key}: got float {val}, expected integer")
        elif expected_type == "number":
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                errors.append(
                    f"{key}: expected number, got {type(val).__name__}"
                )
        elif expected_type == "string":
            if not isinstance(val, str):
                errors.append(
                    f"{key}: expected string, got {type(val).__name__}"
                )
        elif expected_type == "boolean":
            if not isinstance(val, bool):
                errors.append(
                    f"{key}: expected boolean, got {type(val).__name__}"
                )
        elif expected_type == "object" and isinstance(val, dict):
            nested_errors = _validate_args(val, prop_schema)
            errors.extend(f"{key}.{e}" for e in nested_errors)
        elif expected_type == "array" and isinstance(val, list):
            items_schema = prop_schema.get("items", {})
            for i, item in enumerate(val):
                if (
                    isinstance(item, dict)
                    and items_schema.get("type") == "object"
                ):
                    nested_errors = _validate_args(item, items_schema)
                    errors.extend(f"{key}[{i}].{e}" for e in nested_errors)

    return errors


def _extract_tc_args(resp: Any) -> tuple[dict[str, Any] | None, str]:
    """Extract parsed tool call arguments from an OpenAI SDK response.

    Returns (args_dict, error_string). On success error_string is empty.
    """
    tcs = resp.choices[0].message.tool_calls
    if not tcs:
        return None, "no tool_calls returned"
    raw = tcs[0].function.arguments
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON args: {e}"
    if not isinstance(args, dict):
        return None, f"args not dict: {type(args).__name__}"
    return args, ""


@register_scenario
class TCSchemaEnforcement(BaseScenario):
    name = "tc_schema_enforcement"
    description = (
        "Tool calling schema enforcement: required fields, enums, "
        "integer vs number, null, nested objects, fixed property order"
    )
    tags = ["validation", "tool_calling", "schema", "grammar"]
    requires_validator = True
    scenario_type = "validation"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        v = config.validator
        if not v:
            results.append(
                self.make_result(
                    self.name,
                    "setup",
                    Verdict.ERROR,
                    detail="No validator client available",
                )
            )
            return results
        loop = asyncio.get_running_loop()

        results.extend(await self._required_fields(v, loop))
        results.extend(await self._enum_enforcement(v, loop))
        results.extend(await self._integer_vs_number(v, loop))
        results.extend(await self._nested_object_typed(v, loop))
        results.extend(await self._many_properties(v, loop))
        results.extend(await self._required_consistency(v, loop))
        results.extend(await self._scientific_notation(v, loop))
        results.extend(await self._null_value(v, loop))
        results.extend(await self._multiline_string(v, loop))
        results.extend(await self._hyphenated_keys_freeform(v, loop))
        results.extend(await self._enum_with_object_value(v, loop))
        results.extend(await self._integer_scientific_notation(v, loop))
        results.extend(await self._no_type_field(v, loop))
        results.extend(await self._type_array_number_null(v, loop))
        results.extend(await self._ref_defs_fail_open(v, loop))
        results.extend(await self._const_value_fail_open(v, loop))
        results.extend(await self._nullable_type_list(v, loop))
        results.extend(await self._null_type_standalone(v, loop))
        results.extend(await self._anyof_nullable_string(v, loop))
        results.extend(await self._anyof_multi_object(v, loop))
        results.extend(await self._oneof_string_int(v, loop))
        results.extend(await self._enum_mixed_with_null(v, loop))
        results.extend(await self._enum_bool_null(v, loop))
        results.extend(await self._additional_properties_true(v, loop))
        results.extend(await self._deep_nesting_5_levels(v, loop))
        results.extend(await self._required_optional_mix(v, loop))
        results.extend(await self._all_optional_empty_args(v, loop))
        results.extend(await self._additional_properties_false(v, loop))
        results.extend(await self._ref_defs_recursive(v, loop))
        results.extend(await self._type_list_object_with_properties(v, loop))
        results.extend(await self._type_list_array_with_items(v, loop))
        results.extend(await self._properties_without_type_object(v, loop))
        results.extend(await self._enum_dict_literal_exact(v, loop))

        return results

    def _tc(
        self,
        v: Any,
        loop: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        return loop.run_in_executor(
            None,
            lambda: v.tc_chat(
                messages,
                tools,
                tool_choice="required",
                max_tokens=1024,
                **kwargs,
            ),
        )

    async def _run_tc_test(
        self,
        v: Any,
        loop: Any,
        test_name: str,
        schema: dict[str, Any],
        tool_name: str,
        tool_desc: str,
        user_message: str,
        validate: Callable[[dict[str, Any]], tuple[Verdict, str]] | None = None,
    ) -> list[ScenarioResult]:
        """Run a single tool-call test with standard boilerplate.

        Handles the try/budget_exhausted/extract_args/except envelope.
        If ``validate`` is provided, it receives the parsed args dict and
        must return ``(verdict, detail)``.  Otherwise falls back to
        ``_validate_args`` against the schema.
        """
        results: list[ScenarioResult] = []
        tool = make_tool(tool_name, schema, tool_desc)
        try:
            resp = await self._tc(
                v,
                loop,
                [{"role": "user", "content": user_message}],
                [tool],
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
                return results
            args, err = _extract_tc_args(resp)
            if err:
                results.append(
                    self.make_result(
                        self.name, test_name, Verdict.FAIL, detail=err
                    )
                )
                return results
            assert args is not None
            if validate is not None:
                verdict, detail = validate(args)
            else:
                errors = _validate_args(args, schema)
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                detail = "; ".join(errors) or f"OK: {json.dumps(args)}"
            results.append(
                self.make_result(self.name, test_name, verdict, detail=detail)
            )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, test_name, Verdict.ERROR, error=str(e)
                )
            )
        return results

    # ------------------------------------------------------------------
    # 1. Required fields always present
    # ------------------------------------------------------------------

    async def _required_fields(self, v: Any, loop: Any) -> list[ScenarioResult]:
        return await self._run_tc_test(
            v,
            loop,
            test_name="required_fields_present",
            schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "country": {"type": "string"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "country", "unit"],
                "additionalProperties": False,
            },
            tool_name="get_weather",
            tool_desc="Get weather for a city",
            user_message="Weather in Paris, France in celsius?",
        )

    # ------------------------------------------------------------------
    # 2. Enum enforcement
    # ------------------------------------------------------------------

    async def _enum_enforcement(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location", "unit"],
            "additionalProperties": False,
        }
        tool = make_tool("get_temperature", schema, "Get temperature")

        # Run 5 times to increase chance of catching non-deterministic violations
        enum_pass = 0
        enum_fail_details: list[str] = []
        for i in range(5):
            try:
                resp = await self._tc(
                    v,
                    loop,
                    [
                        {
                            "role": "user",
                            "content": f"Temperature in city #{i + 1}: Tokyo?",
                        }
                    ],
                    [tool],
                )
                if budget_exhausted(resp):
                    continue
                args, err = _extract_tc_args(resp)
                if err:
                    enum_fail_details.append(f"run {i}: {err}")
                    continue
                assert args is not None
                unit = args.get("unit")
                if unit in ("celsius", "fahrenheit"):
                    enum_pass += 1
                else:
                    enum_fail_details.append(f"run {i}: unit={unit!r}")
            except Exception as e:
                enum_fail_details.append(f"run {i}: {e}")

        if enum_fail_details:
            results.append(
                self.make_result(
                    self.name,
                    "enum_string_enforcement",
                    Verdict.FAIL,
                    detail=f"{enum_pass}/5 valid; {'; '.join(enum_fail_details[:3])}",
                )
            )
        else:
            results.append(
                self.make_result(
                    self.name,
                    "enum_string_enforcement",
                    Verdict.PASS,
                    detail=f"{enum_pass}/5 all valid enum values",
                )
            )

        # Integer enum
        int_schema = {
            "type": "object",
            "properties": {
                "status_code": {
                    "type": "integer",
                    "enum": [200, 404, 500],
                },
            },
            "required": ["status_code"],
            "additionalProperties": False,
        }
        int_tool = make_tool(
            "get_status",
            int_schema,
            "Return an HTTP status code for a URL check",
        )

        try:
            resp = await self._tc(
                v,
                loop,
                [
                    {
                        "role": "user",
                        "content": "Check if example.com is reachable and return the status code.",
                    }
                ],
                [int_tool],
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "enum_integer_enforcement",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                args, err = _extract_tc_args(resp)
                if err:
                    verdict = Verdict.FAIL
                    detail = err
                else:
                    assert args is not None
                    sc = args.get("status_code")
                    if sc in (200, 404, 500):
                        verdict = Verdict.PASS
                        detail = f"status_code={sc}"
                    else:
                        verdict = Verdict.FAIL
                        detail = (
                            f"status_code={sc!r} not in enum [200, 404, 500]"
                        )
                results.append(
                    self.make_result(
                        self.name,
                        "enum_integer_enforcement",
                        verdict,
                        detail=detail,
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "enum_integer_enforcement",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        return results

    # ------------------------------------------------------------------
    # 3. Integer vs number typing
    # ------------------------------------------------------------------

    async def _integer_vs_number(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            dur = args.get("duration_minutes")
            if isinstance(dur, float):
                return Verdict.FAIL, (
                    f"duration_minutes is float ({dur}), expected int"
                )
            if isinstance(dur, int) and not isinstance(dur, bool):
                return Verdict.PASS, (
                    f"duration_minutes={dur} (int), rating={args.get('rating')}"
                )
            return Verdict.FAIL, (
                f"duration_minutes has type {type(dur).__name__}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="integer_no_decimal",
            schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "duration_minutes": {"type": "integer"},
                    "rating": {"type": "number"},
                },
                "required": ["title", "duration_minutes", "rating"],
                "additionalProperties": False,
            },
            tool_name="create_event",
            tool_desc="Create an event with a title, duration in minutes, and rating",
            user_message=(
                "Create an event titled 'Team Standup' that lasts "
                "30 minutes with a rating of 4.5."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 4. Nested object inner types enforced
    # ------------------------------------------------------------------

    async def _nested_object_typed(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        schema = {
            "type": "object",
            "properties": {
                "server": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer"},
                        "tls": {"type": "boolean"},
                    },
                    "required": ["host", "port", "tls"],
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["server", "tags"],
            "additionalProperties": False,
        }

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors = _validate_args(args, schema)
            port = args.get("server", {}).get("port")
            if isinstance(port, str):
                errors.append(
                    f"server.port is string '{port}', expected integer"
                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, "; ".join(errors) or f"OK: {json.dumps(args)}"

        return await self._run_tc_test(
            v,
            loop,
            test_name="nested_object_types",
            schema=schema,
            tool_name="configure_server",
            tool_desc="Configure a server connection with host, port, TLS, and tags",
            user_message=(
                "Configure server at api.example.com on port 443 "
                "with TLS enabled, tags: production, primary."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 5. Many properties (20 required)
    # ------------------------------------------------------------------

    async def _many_properties(self, v: Any, loop: Any) -> list[ScenarioResult]:
        field_names = [
            "name",
            "email",
            "phone",
            "address",
            "city",
            "state",
            "zip_code",
            "country",
            "company",
            "title",
            "department",
            "team",
            "manager",
            "start_date",
            "office",
            "floor",
            "desk",
            "badge_id",
            "role",
            "level",
        ]
        schema = {
            "type": "object",
            "properties": {f: {"type": "string"} for f in field_names},
            "required": field_names,
            "additionalProperties": False,
        }

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors = _validate_args(args, schema)
            missing = [f for f in field_names if f not in args]
            if missing:
                errors.append(f"missing {len(missing)} fields: {missing[:5]}")
            if len(args) < len(field_names):
                errors.append(
                    f"only {len(args)} keys, expected {len(field_names)}"
                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, "; ".join(
                errors
            ) or f"OK: {len(args)} fields present"

        return await self._run_tc_test(
            v,
            loop,
            test_name="many_properties_20",
            schema=schema,
            tool_name="register_employee",
            tool_desc="Register a new employee with all fields",
            user_message=(
                "Register employee: Jane Smith, jane@acme.com, "
                "555-1234, 100 Main St, Springfield, IL, 62701, "
                "US, Acme Corp, Senior Engineer, Engineering, "
                "Platform, Bob Jones, 2025-01-15, HQ, 3, D42, "
                "B-9876, engineer, senior."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 6. Required fields consistency (N runs)
    # ------------------------------------------------------------------

    async def _required_consistency(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
                "c": {"type": "boolean"},
            },
            "required": ["a", "b", "c"],
            "additionalProperties": False,
        }
        tool = make_tool(
            "submit_form",
            schema,
            "Submit a form with fields a (string), b (integer), c (boolean)",
        )

        n_runs = 5
        passes = 0
        fail_details: list[str] = []
        for i in range(n_runs):
            try:
                resp = await self._tc(
                    v,
                    loop,
                    [
                        {
                            "role": "user",
                            "content": f"Submit form: a='hello{i}', b={i + 1}, c=true.",
                        }
                    ],
                    [tool],
                )
                if budget_exhausted(resp):
                    continue
                args, err = _extract_tc_args(resp)
                if err:
                    fail_details.append(f"run {i}: {err}")
                    continue
                assert args is not None
                errors = _validate_args(args, schema)
                if errors:
                    fail_details.append(f"run {i}: {'; '.join(errors)}")
                else:
                    passes += 1
            except Exception as e:
                fail_details.append(f"run {i}: {e}")

        if fail_details:
            results.append(
                self.make_result(
                    self.name,
                    "required_consistency_5_runs",
                    Verdict.FAIL,
                    detail=f"{passes}/{n_runs} valid; {'; '.join(fail_details[:3])}",
                )
            )
        else:
            results.append(
                self.make_result(
                    self.name,
                    "required_consistency_5_runs",
                    Verdict.PASS,
                    detail=f"{passes}/{n_runs} all valid across runs",
                )
            )

        return results

    # ------------------------------------------------------------------
    # 7. Scientific notation accepted in number fields
    # ------------------------------------------------------------------

    async def _scientific_notation(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify that number fields accept scientific notation (e.g. 1.5e10).

        We can't force the model to emit sci notation, so we validate that
        the returned number field parses as a valid Python float/int -- the
        grammar allows sci notation and the parser handles it.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            dist = args.get("distance_meters")
            pop = args.get("population")
            errors: list[str] = []
            if not isinstance(dist, (int, float)) or isinstance(dist, bool):
                errors.append(
                    f"distance_meters: expected number, got {type(dist).__name__}"
                )
            if not isinstance(pop, int) or isinstance(pop, bool):
                errors.append(
                    f"population: expected integer, got {type(pop).__name__}"
                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors) or f"OK: distance={dist}, population={pop}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="number_field_valid",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "distance_meters": {"type": "number"},
                    "population": {"type": "integer"},
                },
                "required": ["name", "distance_meters", "population"],
                "additionalProperties": False,
            },
            tool_name="record_celestial_body",
            tool_desc="Record a celestial body with name, distance in meters, and population",
            user_message=(
                "Record celestial body: Earth, distance from Sun "
                "is 149597870700 meters, population 8000000000."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 8. Null value accepted for optional fields
    # ------------------------------------------------------------------

    async def _null_value(self, v: Any, loop: Any) -> list[ScenarioResult]:
        """Verify that the grammar does not crash when a field could be null.

        The model may or may not emit null -- we just verify the request
        succeeds and produces valid JSON args with correct required fields.
        The grammar's null_val rule ensures null is parseable if emitted.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "name" not in args:
                errors.append("missing required field: name")
            if "age" not in args:
                errors.append("missing required field: age")
            nickname = args.get("nickname")
            if nickname is not None and not isinstance(nickname, str):
                errors.append(
                    f"nickname: expected string or null, got {type(nickname).__name__}"
                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, "; ".join(errors) or f"OK: nickname={nickname!r}"

        return await self._run_tc_test(
            v,
            loop,
            test_name="optional_field_nullable",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "nickname": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
                "additionalProperties": False,
            },
            tool_name="register_person",
            tool_desc="Register a person. nickname is optional and may be omitted.",
            user_message=(
                "Register person: name is John Doe, age 30. No nickname."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 9. Multiline strings accepted in string fields
    # ------------------------------------------------------------------

    async def _multiline_string(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify the grammar accepts newlines inside string values.

        STRING_CONTENT must match newline characters so the model can
        produce multi-line strings (e.g. poems, addresses, code).
        """
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["title", "content"],
            "additionalProperties": False,
        }

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors = _validate_args(args, schema)
            content = args.get("content", "")
            has_newline = "\n" in content if isinstance(content, str) else False
            if errors:
                return Verdict.FAIL, "; ".join(errors)
            if has_newline:
                return (
                    Verdict.PASS,
                    f"OK: content has newline, len={len(content)}",
                )
            return Verdict.FAIL, (
                f"grammar likely blocked newlines: "
                f"content={content!r} (no newline)"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="multiline_string_value",
            schema=schema,
            tool_name="save_note",
            tool_desc="Save a note with a title and multi-line content",
            user_message=(
                "Save a note titled 'Grocery List'. The content "
                "field MUST contain actual newline characters "
                "between items. Format it exactly like:\n"
                "- milk\n- eggs\n- bread\n"
                "Include the newlines in the content string."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 10. Hyphenated keys in free-form object properties
    # ------------------------------------------------------------------

    async def _hyphenated_keys_freeform(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify free-form objects accept property names with hyphens/dots.

        The KEY terminal must allow characters beyond [a-zA-Z0-9_] so
        that keys like 'Content-Type' or 'x.custom' work in untyped
        object properties.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            headers = args.get("headers")
            if not isinstance(headers, dict):
                return Verdict.FAIL, (
                    f"headers not a dict: {type(headers).__name__}"
                )
            if any("-" in k for k in headers):
                return Verdict.PASS, f"keys={list(headers.keys())}"
            return Verdict.FAIL, (
                f"grammar likely blocked hyphens in keys: "
                f"keys={list(headers.keys())}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="freeform_object_hyphen_keys",
            schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "headers": {"type": "object"},
                },
                "required": ["url", "headers"],
                "additionalProperties": False,
            },
            tool_name="send_request",
            tool_desc="Send an HTTP request with a URL and custom headers object",
            user_message=(
                "Send a request to https://api.example.com/data "
                "with headers Content-Type=application/json and "
                "Accept=text/html."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 11. Enum with object/array values accepted
    # ------------------------------------------------------------------

    async def _enum_with_object_value(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify enum alternatives that are objects or arrays are not dropped.

        _enum_value_rule must handle dict/list enum values instead of
        silently discarding them, which would make a valid alternative
        impossible to produce.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            config = args.get("config")
            if isinstance(config, dict):
                return Verdict.PASS, f"config={config!r} (object selected)"
            return Verdict.FAIL, (
                f"grammar likely dropped object enum alternative: "
                f"config={config!r} (type={type(config).__name__})"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="enum_with_object_value",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "config": {
                        "enum": [
                            "default",
                            {"mode": "advanced", "level": 3},
                        ],
                    },
                },
                "required": ["name", "config"],
                "additionalProperties": False,
            },
            tool_name="apply_config",
            tool_desc=(
                "Apply a configuration. config is either 'default' or "
                "an object {mode: 'advanced', level: 3}."
            ),
            user_message=(
                "Apply the advanced configuration (mode=advanced, "
                "level=3) for component 'router'."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 12. Integer field accepts scientific notation
    # ------------------------------------------------------------------

    async def _integer_scientific_notation(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify integer fields accept scientific notation (e.g. 1e50).

        Uses astronomically large numbers that are impractical to write
        in digit form, forcing the model to use scientific notation.
        The INTEGER terminal must allow an exponent suffix for this to work.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            qty = args.get("quantity")
            if isinstance(qty, float) and qty >= 1e20:
                return Verdict.PASS, (
                    f"quantity={qty} (sci notation parsed as float)"
                )
            if isinstance(qty, int) and qty >= 10**20:
                return Verdict.PASS, f"quantity={qty} (large integer)"
            return Verdict.FAIL, (
                f"quantity={qty!r} (type={type(qty).__name__}) -- "
                f"expected a very large number via sci notation"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="integer_scientific_notation",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "quantity": {"type": "integer"},
                },
                "required": ["name", "quantity"],
                "additionalProperties": False,
            },
            tool_name="record_quantity",
            tool_desc=(
                "Record a named quantity. The quantity field is an integer. "
                "You MUST use scientific notation (e.g. 1e50) for large values."
            ),
            user_message=(
                "Record quantity: name='atoms_in_universe', "
                "quantity=1e80. Write the quantity using "
                "scientific notation exactly as 1e80."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 13. Nullable type list (e.g. ["string", "null"])
    # ------------------------------------------------------------------

    async def _nullable_type_list(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify schemas with union types like ["string", "null"] don't crash.

        JSON Schema allows type to be a list. When the grammar generator
        receives a list for type, dict.get() raises TypeError (unhashable
        type: 'list') unless handled explicitly.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "name" not in args:
                errors.append("missing required field: name")
            if "nickname" not in args:
                errors.append("missing required field: nickname")
            else:
                nn = args["nickname"]
                if nn is not None and not isinstance(nn, str):
                    errors.append(
                        f"nickname: expected string or null, got {type(nn).__name__}"
                    )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors)
                or f"OK: name={args.get('name')!r}, nickname={args.get('nickname')!r}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="nullable_type_list",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "nickname": {"type": ["string", "null"]},
                },
                "required": ["name", "nickname"],
                "additionalProperties": False,
            },
            tool_name="greet_user",
            tool_desc="Greet a user. nickname can be a string or null.",
            user_message=(
                "Greet user: name is 'Alice', nickname is null "
                "(she has no nickname). Set nickname to null."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 14. Standalone null type property
    # ------------------------------------------------------------------

    async def _null_type_standalone(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify a property with type "null" produces null and uses null_val rule.

        _JSON_TYPE_TO_GRAMMAR_RULE must have an entry for "null" mapping
        to "null_val", otherwise the property falls to generic "value"
        (fail-open).
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "action" not in args:
                errors.append("missing required field: action")
            if "reason" not in args:
                errors.append("missing required field: reason")
            elif args["reason"] is not None:
                errors.append(
                    f"reason: expected null, got "
                    f"{type(args['reason']).__name__}={args['reason']!r}"
                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors)
                or f"OK: action={args.get('action')!r}, reason=null"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="null_type_standalone",
            schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "reason": {"type": "null"},
                },
                "required": ["action", "reason"],
                "additionalProperties": False,
            },
            tool_name="cancel_request",
            tool_desc="Cancel a request. reason must always be null.",
            user_message=(
                "Cancel the pending request. The action is 'cancel' "
                "and reason is null (must be null, not a string)."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 15. anyOf with nullable string: anyOf: [{type:string},{type:null}]
    # ------------------------------------------------------------------

    async def _anyof_nullable_string(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "name" not in args:
                errors.append("missing required field: name")
            if "middle_name" not in args:
                errors.append("missing required field: middle_name")
            else:
                mn = args["middle_name"]
                if mn is not None and not isinstance(mn, str):
                    errors.append(
                        f"middle_name: expected string or null, "
                        f"got {type(mn).__name__}"
                    )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors)
                or f"OK: middle_name={args.get('middle_name')!r}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="anyof_nullable_string",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "middle_name": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                    },
                },
                "required": ["name", "middle_name"],
                "additionalProperties": False,
            },
            tool_name="register_name",
            tool_desc="Register a name. middle_name can be a string or null.",
            user_message=(
                "Register: name='Alice', middle_name is null "
                "(she has no middle name). Set middle_name to null."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 16. anyOf with multiple object branches
    # ------------------------------------------------------------------

    async def _anyof_multi_object(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            payment = args.get("payment")
            if not isinstance(payment, dict):
                return Verdict.FAIL, (
                    f"payment not dict: {type(payment).__name__}"
                )
            has_method = isinstance(payment.get("method"), str)
            has_card = isinstance(payment.get("card_last4"), str)
            has_bank = isinstance(payment.get("bank_name"), str)
            if has_method and (has_card or has_bank):
                return Verdict.PASS, f"OK: payment={payment!r}"
            return Verdict.FAIL, (
                f"payment missing expected fields: {payment!r}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="anyof_multi_object",
            schema={
                "type": "object",
                "properties": {
                    "payment": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "method": {"type": "string"},
                                    "card_last4": {"type": "string"},
                                },
                                "required": ["method", "card_last4"],
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "method": {"type": "string"},
                                    "bank_name": {"type": "string"},
                                },
                                "required": ["method", "bank_name"],
                            },
                        ],
                    },
                },
                "required": ["payment"],
                "additionalProperties": False,
            },
            tool_name="process_payment",
            tool_desc=(
                "Process a payment. Payment is either a card (with card_last4) "
                "or bank transfer (with bank_name)."
            ),
            user_message=(
                "Process a credit card payment, card last 4 digits are 1234."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 17. oneOf with string or integer
    # ------------------------------------------------------------------

    async def _oneof_string_int(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            ident = args.get("identifier")
            if isinstance(ident, (str, int)) and not isinstance(ident, bool):
                return Verdict.PASS, (
                    f"OK: identifier={ident!r} ({type(ident).__name__})"
                )
            return Verdict.FAIL, (
                f"identifier={ident!r} ({type(ident).__name__}) "
                f"-- expected string or integer"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="oneof_string_int",
            schema={
                "type": "object",
                "properties": {
                    "identifier": {
                        "oneOf": [{"type": "string"}, {"type": "integer"}],
                    },
                },
                "required": ["identifier"],
                "additionalProperties": False,
            },
            tool_name="lookup_entity",
            tool_desc="Look up an entity by identifier (string name or integer ID).",
            user_message="Look up entity with ID 42.",
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 18. Enum with null among string values
    # ------------------------------------------------------------------

    async def _enum_mixed_with_null(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            status = args.get("status")
            if status in ("active", "inactive", None):
                return Verdict.PASS, f"OK: status={status!r}"
            return Verdict.FAIL, (
                f"status={status!r} not in enum ['active','inactive',null]"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="enum_mixed_with_null",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "status": {"enum": ["active", "inactive", None]},
                },
                "required": ["name", "status"],
                "additionalProperties": False,
            },
            tool_name="set_status",
            tool_desc="Set user status. Status is 'active', 'inactive', or null.",
            user_message=(
                "Set status for user 'Bob' to null (no status assigned)."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 19. Enum with boolean and null values
    # ------------------------------------------------------------------

    async def _enum_bool_null(self, v: Any, loop: Any) -> list[ScenarioResult]:
        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            enabled = args.get("enabled")
            if enabled in (True, False, None):
                return Verdict.PASS, f"OK: enabled={enabled!r}"
            return Verdict.FAIL, (
                f"enabled={enabled!r} ({type(enabled).__name__}) "
                f"not in [true, false, null]"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="enum_bool_null",
            schema={
                "type": "object",
                "properties": {
                    "feature": {"type": "string"},
                    "enabled": {"enum": [True, False, None]},
                },
                "required": ["feature", "enabled"],
                "additionalProperties": False,
            },
            tool_name="toggle_feature",
            tool_desc="Toggle a feature. enabled is true, false, or null (unknown).",
            user_message="Toggle feature 'dark_mode' to enabled=true.",
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 20. $ref/$defs
    # ------------------------------------------------------------------

    async def _ref_defs_fail_open(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify that $ref/$defs are resolved and enforced by the grammar."""

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            item = args.get("item")
            if not isinstance(item, dict):
                return Verdict.FAIL, (
                    f"item={item!r} is not a dictionary with name and price"
                )
            has_name = isinstance(item.get("name"), str)
            has_price = isinstance(item.get("price"), (int, float))
            if has_name and has_price:
                return Verdict.PASS, f"OK: item={item!r}"
            return Verdict.FAIL, (
                f"item={item!r} is not a dictionary with name and price"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_fail_open",
            schema={
                "type": "object",
                "properties": {
                    "item": {"$ref": "#/$defs/Item"},
                },
                "$defs": {
                    "Item": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "number"},
                        },
                        "required": ["name", "price"],
                    },
                },
                "required": ["item"],
                "additionalProperties": False,
            },
            tool_name="add_item",
            tool_desc="Add an item with name and price.",
            user_message="Add item: name='Widget', price=9.99.",
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 21. const keyword
    # ------------------------------------------------------------------

    async def _const_value_fail_open(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify that the const keyword is enforced."""

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            version = args.get("version")
            if version == "2.0":
                return Verdict.PASS, (
                    f"OK: version={version!r} (model produced correct const)"
                )
            return Verdict.FAIL, (
                f"version={version!r} is not '2.0' "
                f"(model did not produce correct const)"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="const_value_fail_open",
            schema={
                "type": "object",
                "properties": {
                    "version": {"const": "2.0"},
                    "name": {"type": "string"},
                },
                "required": ["version", "name"],
                "additionalProperties": False,
            },
            tool_name="create_spec",
            tool_desc="Create a spec. version must be exactly '2.0'.",
            user_message="Create a spec named 'my-api' with version '2.0'.",
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 22. additionalProperties: true (default) — extra props allowed
    # ------------------------------------------------------------------

    async def _additional_properties_true(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """When additionalProperties is not set (defaults to true in JSON
        Schema), the grammar should not reject undeclared properties.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            if "name" not in args:
                return Verdict.FAIL, "missing required field: name"
            extra_keys = set(args.keys()) - {"name"}
            if extra_keys:
                return Verdict.PASS, (
                    f"OK: extra properties accepted: {sorted(extra_keys)}"
                )
            return Verdict.FAIL, (
                "grammar likely over-constrained: only 'name' "
                "returned, no additional properties despite "
                "additionalProperties defaulting to true"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="additional_properties_true",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            },
            tool_name="flexible_input",
            tool_desc=(
                "Accept flexible input. name is required, but you may include "
                "any additional fields like age, email, etc."
            ),
            user_message=(
                "Create input: name='Alice', age=30, "
                "email='alice@example.com'. Include all three fields."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 23. Deep nesting (5 levels of nested objects)
    # ------------------------------------------------------------------

    async def _deep_nesting_5_levels(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            try:
                l5_val = args["l1"]["l2"]["l3"]["l4"]["l5"]
                if not isinstance(l5_val, str):
                    errors.append(
                        f"l5: expected string, got {type(l5_val).__name__}"
                    )
            except (KeyError, TypeError) as e:
                errors.append(f"nesting traversal failed: {e}")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors)
                or f"OK: l1.l2.l3.l4.l5={args['l1']['l2']['l3']['l4']['l5']!r}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="deep_nesting_5_levels",
            schema={
                "type": "object",
                "properties": {
                    "l1": {
                        "type": "object",
                        "properties": {
                            "l2": {
                                "type": "object",
                                "properties": {
                                    "l3": {
                                        "type": "object",
                                        "properties": {
                                            "l4": {
                                                "type": "object",
                                                "properties": {
                                                    "l5": {"type": "string"},
                                                },
                                                "required": ["l5"],
                                            },
                                        },
                                        "required": ["l4"],
                                    },
                                },
                                "required": ["l3"],
                            },
                        },
                        "required": ["l2"],
                    },
                },
                "required": ["l1"],
                "additionalProperties": False,
            },
            tool_name="deep_store",
            tool_desc="Store a deeply nested value at l1.l2.l3.l4.l5.",
            user_message=(
                "Store the value 'deep_value' at the path l1.l2.l3.l4.l5."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 24. Mix of required and optional properties
    # ------------------------------------------------------------------

    async def _required_optional_mix(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            for req in ("first_name", "last_name", "email"):
                if req not in args:
                    errors.append(f"missing required field: {req}")
                elif not isinstance(args[req], str):
                    errors.append(
                        f"{req}: expected string, got {type(args[req]).__name__}"
                    )
            for opt in ("phone", "bio", "website"):
                if opt in args and not isinstance(args[opt], str):
                    errors.append(
                        f"{opt}: expected string, got {type(args[opt]).__name__}"
                    )
            allowed = {
                "first_name",
                "last_name",
                "email",
                "phone",
                "bio",
                "website",
            }
            extra = set(args.keys()) - allowed
            if extra:
                errors.append(f"undeclared properties: {sorted(extra)}")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors) or f"OK: keys={sorted(args.keys())}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="required_optional_mix",
            schema={
                "type": "object",
                "properties": {
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                    "bio": {"type": "string"},
                    "website": {"type": "string"},
                },
                "required": ["first_name", "last_name", "email"],
                "additionalProperties": False,
            },
            tool_name="create_profile",
            tool_desc=(
                "Create a user profile. first_name, last_name, email are "
                "required. phone, bio, website are optional."
            ),
            user_message=(
                "Create profile: first_name='Jane', "
                "last_name='Doe', email='jane@example.com'. "
                "Skip the optional fields."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 25. All optional — model may produce empty args
    # ------------------------------------------------------------------

    async def _all_optional_empty_args(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "note" in args and not isinstance(args["note"], str):
                errors.append(
                    f"note: expected string, got {type(args['note']).__name__}"
                )
            if "priority" in args:
                p = args["priority"]
                if not isinstance(p, int) or isinstance(p, bool):
                    errors.append(
                        f"priority: expected integer, got {type(p).__name__}"
                    )
            extra = set(args.keys()) - {"note", "priority"}
            if extra:
                errors.append(f"undeclared properties: {sorted(extra)}")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors)
                or f"OK: args={args!r} (empty or optional fields)"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="all_optional_empty_args",
            schema={
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                    "priority": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            tool_name="ping",
            tool_desc="Send a ping. All parameters are optional.",
            user_message="Send a simple ping with no parameters.",
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 26. Property with no type field
    # ------------------------------------------------------------------

    async def _no_type_field(self, v: Any, loop: Any) -> list[ScenarioResult]:
        """A property with no 'type' key should fall back to generic 'value'
        and accept any JSON. Must not crash.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "label" not in args:
                errors.append("missing required field: label")
            if "data" not in args:
                errors.append("missing required field: data")
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors)
                or f"OK: data={args.get('data')!r} ({type(args.get('data')).__name__})"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="no_type_field",
            schema={
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "data": {
                        "description": "Arbitrary data, no type specified."
                    },
                },
                "required": ["label", "data"],
                "additionalProperties": False,
            },
            tool_name="store_data",
            tool_desc="Store data. The data field has no type -- it can be anything.",
            user_message="Store data: label='test', data=42.",
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 27. type: ["number", "null"] — another type-as-array variant
    # ------------------------------------------------------------------

    async def _type_array_number_null(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Like nullable_type_list but with number instead of string.
        Exercises the same dict.get() crash path with a different type.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "label" not in args:
                errors.append("missing required field: label")
            if "score" not in args:
                errors.append("missing required field: score")
            else:
                sc = args["score"]
                if sc is not None and (
                    not isinstance(sc, (int, float)) or isinstance(sc, bool)
                ):
                    errors.append(
                        f"score: expected number or null, "
                        f"got {type(sc).__name__}={sc!r}"
                    )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors) or f"OK: score={args.get('score')!r}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="type_array_number_null",
            schema={
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "score": {"type": ["number", "null"]},
                },
                "required": ["label", "score"],
                "additionalProperties": False,
            },
            tool_name="record_score",
            tool_desc="Record a score. score is a number or null.",
            user_message="Record: label='test_run', score=95.5.",
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 28. additionalProperties: false — grammar rejects undeclared keys
    # ------------------------------------------------------------------

    async def _additional_properties_false(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify additionalProperties: false prevents undeclared properties.

        The prompt deliberately asks the model to include extra fields that
        are NOT in the schema. The grammar should prevent them from appearing.
        """
        schema = {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"},
            },
            "required": ["city", "country"],
            "additionalProperties": False,
        }

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "city" not in args:
                errors.append("missing required field: city")
            if "country" not in args:
                errors.append("missing required field: country")
            extra = set(args.keys()) - {"city", "country"}
            if extra:
                errors.append(
                    f"additionalProperties violated: "
                    f"undeclared keys {sorted(extra)}"
                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors)
                or f"OK: keys={sorted(args.keys())} (no extras)"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="additional_properties_false",
            schema=schema,
            tool_name="get_location",
            tool_desc=(
                "Get a location. Only city and country are accepted. "
                "No other fields."
            ),
            user_message=(
                "Get the location for Tokyo, Japan. Also include the "
                "population (1400000), the continent (Asia), and "
                "the timezone (JST). Include all five fields."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 29. $ref/$defs recursive schema (linked list)
    # ------------------------------------------------------------------

    async def _ref_defs_recursive(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify recursive $ref/$defs schemas work (linked list pattern).

        This is the key use case from the OpenAI structured outputs docs:
        a node type that references itself via $ref inside anyOf.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            node = args.get("list")
            if not isinstance(node, dict):
                return Verdict.FAIL, (
                    f"list: expected object, got {type(node).__name__}"
                )
            depth = 0
            current = node
            while isinstance(current, dict):
                depth += 1
                if not isinstance(current.get("value"), str):
                    return Verdict.FAIL, (
                        f"node at depth {depth}: value is "
                        f"{type(current.get('value')).__name__}, "
                        f"expected string"
                    )
                nxt = current.get("next")
                if nxt is None:
                    break
                if not isinstance(nxt, dict):
                    return Verdict.FAIL, (
                        f"node at depth {depth}: next is "
                        f"{type(nxt).__name__}, expected object or null"
                    )
                current = nxt
            if depth < 2:
                return Verdict.FAIL, (
                    f"only {depth} node(s), expected at least 2"
                )
            return Verdict.PASS, f"OK: linked list with {depth} nodes"

        return await self._run_tc_test(
            v,
            loop,
            test_name="ref_defs_recursive",
            schema={
                "type": "object",
                "properties": {
                    "list": {"$ref": "#/$defs/node"},
                },
                "$defs": {
                    "node": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                            "next": {
                                "anyOf": [
                                    {"$ref": "#/$defs/node"},
                                    {"type": "null"},
                                ],
                            },
                        },
                        "required": ["value", "next"],
                    },
                },
                "required": ["list"],
                "additionalProperties": False,
            },
            tool_name="build_list",
            tool_desc=(
                "Build a linked list. Each node has a string value and "
                "a next pointer (another node or null)."
            ),
            user_message=(
                "Build a linked list with 3 items: 'alpha', 'beta', "
                "'gamma'. The last node's next should be null."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 30. type: ["object", "null"] with properties (bug: constraints dropped)
    # ------------------------------------------------------------------

    async def _type_list_object_with_properties(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify that type: ["object", "null"] preserves property constraints.

        Bug: when type is a list containing "object", the grammar enters
        the union branch and maps to generic object_val, silently dropping
        properties/required constraints from the same schema level.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "label" not in args:
                errors.append("missing required field: label")
            if "metadata" not in args:
                errors.append("missing required field: metadata")
            else:
                md = args["metadata"]
                if md is None:
                    pass
                elif isinstance(md, dict):
                    if not isinstance(md.get("author"), str):
                        errors.append(
                            f"metadata.author: expected string, "
                            f"got {type(md.get('author')).__name__}"
                        )
                    ver = md.get("version")
                    if not isinstance(ver, int) or isinstance(ver, bool):
                        errors.append(
                            f"metadata.version: expected integer, "
                            f"got {type(ver).__name__}"
                        )
                else:
                    errors.append(
                        f"metadata: expected object or null, "
                        f"got {type(md).__name__}"
                    )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors) or f"OK: metadata={args.get('metadata')!r}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="type_list_object_with_properties",
            schema={
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "metadata": {
                        "type": ["object", "null"],
                        "properties": {
                            "author": {"type": "string"},
                            "version": {"type": "integer"},
                        },
                        "required": ["author", "version"],
                    },
                },
                "required": ["label", "metadata"],
                "additionalProperties": False,
            },
            tool_name="save_document",
            tool_desc=(
                "Save a document with a label and metadata. "
                "metadata is an object with author (string) and "
                "version (integer), or null."
            ),
            user_message=(
                "Save document: label='report', "
                "metadata with author='Alice' and version=3."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 30b. type: ["array", "null"] with items (bug: constraints dropped)
    # ------------------------------------------------------------------

    async def _type_list_array_with_items(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify that type: ["array", "null"] preserves items constraints.

        Same root cause as test 30: when type is a list containing "array",
        the grammar enters the union branch and maps to generic array_val,
        silently dropping items type constraints from the same schema level.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if "label" not in args:
                errors.append("missing required field: label")
            if "tags" not in args:
                errors.append("missing required field: tags")
            else:
                tags = args["tags"]
                if tags is None:
                    pass
                elif isinstance(tags, list):
                    for i, item in enumerate(tags):
                        if not isinstance(item, str):
                            errors.append(
                                f"tags[{i}]: expected string, "
                                f"got {type(item).__name__}={item!r}"
                            )
                else:
                    errors.append(
                        f"tags: expected array or null, "
                        f"got {type(tags).__name__}"
                    )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors) or f"OK: tags={args.get('tags')!r}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="type_list_array_with_items",
            schema={
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "tags": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                },
                "required": ["label", "tags"],
                "additionalProperties": False,
            },
            tool_name="tag_resource",
            tool_desc=("Tag a resource. tags is an array of strings, or null."),
            user_message=(
                "Tag resource: label='server-01', "
                "tags=['production', 'us-east', 'critical']."
            ),
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 31. properties without type: "object" (bug: constraints dropped)
    # ------------------------------------------------------------------

    async def _properties_without_type_object(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify schemas with properties but no explicit type: "object" work.

        Bug: _generate_property_value_rule checks
        json_type == "object" and prop_schema.get("properties"), so a
        schema without type falls through to generic value rule, losing
        all property constraints.
        """

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            errors: list[str] = []
            if not isinstance(args.get("name"), str):
                errors.append(
                    f"name: expected string, "
                    f"got {type(args.get('name')).__name__}"
                )
            age = args.get("age")
            if not isinstance(age, int) or isinstance(age, bool):
                errors.append(
                    f"age: expected integer, got {type(age).__name__}"
                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            return verdict, (
                "; ".join(errors) or f"OK: name={args.get('name')!r}, age={age}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="properties_without_type_object",
            schema={
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
                "additionalProperties": False,
            },
            tool_name="register_user",
            tool_desc="Register a user with name and age.",
            user_message="Register user: name='Bob', age=25.",
            validate=validate,
        )

    # ------------------------------------------------------------------
    # 32. Enum with dict literals — exact match enforcement
    # ------------------------------------------------------------------

    async def _enum_dict_literal_exact(
        self, v: Any, loop: Any
    ) -> list[ScenarioResult]:
        """Verify enum with dict values constrains to the exact literal.

        Bug: _enum_value_rule maps dict enum values to the generic
        object_val rule, which accepts ANY object instead of only the
        specific literals declared in the enum.
        """
        valid_presets = [
            {"mode": "fast", "threads": 4},
            {"mode": "safe", "threads": 1},
        ]

        def validate(args: dict[str, Any]) -> tuple[Verdict, str]:
            preset = args.get("preset")
            if not isinstance(preset, dict):
                return Verdict.FAIL, (
                    f"preset: expected dict, got {type(preset).__name__}"
                )
            if preset in valid_presets:
                return Verdict.PASS, f"OK: preset={preset!r} (exact match)"
            return Verdict.FAIL, (
                f"preset={preset!r} does not exactly match any "
                f"enum literal: {valid_presets}"
            )

        return await self._run_tc_test(
            v,
            loop,
            test_name="enum_dict_literal_exact",
            schema={
                "type": "object",
                "properties": {
                    "preset": {
                        "enum": valid_presets,
                    },
                },
                "required": ["preset"],
                "additionalProperties": False,
            },
            tool_name="apply_preset",
            tool_desc=(
                "Apply a preset. preset must be exactly one of: "
                "{mode:'fast', threads:4} or {mode:'safe', threads:1}."
            ),
            user_message=("Apply the fast preset (mode='fast', threads=4)."),
            validate=validate,
        )
