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
import json
import re
from typing import Any, ClassVar

from max.pipelines.lib.tool_parsing import (
    StructuralTagToolParser,
    canonicalize_lark_rule_name,
    escape_for_lark_string,
    generate_call_id,
    get_token_id,
    grammar_rule_for_json_type,
    maybe_name_from_tool,
    names_from_tools,
    register,
    resolve_lark_token_reference,
)
from max.pipelines.modeling.types import (
    ParsedToolCall,
    PipelineTokenizer,
)

from .tokenizer import SpecialToken

TOOL_CALL_PATTERN = re.compile(
    re.escape(SpecialToken.TOOL_CALL_START)
    + r"call:([\w\-\.]+)\{(.*?)\}"
    + re.escape(SpecialToken.TOOL_CALL_END),
    re.DOTALL,
)


def _tool_call_rule(
    func_ref: str,
    body: str,
    tcs_ref: str = SpecialToken.TOOL_CALL_START.name,
    tce_ref: str = SpecialToken.TOOL_CALL_END.name,
) -> str:
    """Build a Lark rule fragment for a single tool call alternative."""
    return f'{tcs_ref} "call:" {func_ref} "{{" {body} "}}" {tce_ref}'


def _has_schema_constraints(schema: dict[str, Any]) -> bool:
    """Return True if *schema* declares properties or additionalProperties."""
    return (
        bool(schema.get("properties"))
        or schema.get("additionalProperties", False) is not False
    )


def _resolve_refs(
    node: Any,
    defs: dict[str, Any],
    _depth: int = 0,
    _max_depth: int = 10,
) -> Any:
    """Inline ``$ref`` pointers using definitions from ``$defs``.

    Walks the schema tree and replaces ``{"$ref": "#/$defs/Name"}`` with
    the corresponding entry from *defs*.  Recursion is capped at
    *_max_depth* to handle self-referential schemas safely — unresolved
    ``$ref`` nodes are left as-is and will fall through to the generic
    ``value`` rule in the grammar generator.

    The ``$defs`` key itself is stripped from the returned schema so
    downstream code never sees it.
    """
    if not isinstance(node, dict):
        if isinstance(node, list):
            return [
                _resolve_refs(item, defs, _depth, _max_depth) for item in node
            ]
        return node

    if "$ref" in node:
        ref = node["$ref"]
        if (
            isinstance(ref, str)
            and ref.startswith("#/$defs/")
            and _depth < _max_depth
        ):
            ref_name = ref[len("#/$defs/") :]
            if ref_name in defs:
                return _resolve_refs(
                    defs[ref_name], defs, _depth + 1, _max_depth
                )
        return node

    return {
        k: _resolve_refs(v, defs, _depth, _max_depth)
        for k, v in node.items()
        if k != "$defs"
    }


def _extract_tool_schemas(
    tools: list[dict[str, Any]],
) -> dict[str, dict[str, Any]] | None:
    """Extract parameter schemas from an OpenAI-style tools list."""
    schemas: dict[str, dict[str, Any]] = {}
    for t in tools:
        name = maybe_name_from_tool(t)
        if not name:
            continue
        params = t.get("function", {}).get("parameters")
        if params:
            defs = params.get("$defs", {})
            schemas[name] = _resolve_refs(params, defs) if defs else params
    return schemas or None


def _parse_gemma4_value(value_str: str) -> object:
    """Parse a single Gemma4 value (after key:) into a Python object."""
    value_str = value_str.strip()
    if not value_str:
        return value_str

    if value_str == "null":
        return None

    # Boolean
    if value_str == "true":
        return True
    if value_str == "false":
        return False

    # Number (int or float)
    try:
        if "." in value_str or "e" in value_str or "E" in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # Bare string (no <|"|> delimiters — shouldn't happen but be safe)
    return value_str


def _parse_gemma4_args(
    args_str: str, *, partial: bool = False
) -> dict[str, Any]:
    """Parse Gemma4's custom key:value format into a Python dict.

    Format examples::

        location:<|"|>Tokyo<|"|>
        location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>
        count:42,flag:true
        nested:{inner_key:<|"|>val<|"|>}
        items:[<|"|>a<|"|>,<|"|>b<|"|>]

    Args:
        args_str: The raw Gemma4 argument string.
        partial: When True (streaming), bare values at end of string are
            omitted because they may be incomplete and type-unstable
            (e.g. partial boolean parsed as bare string).

    Returns a dict ready for ``json.dumps()``.
    """
    if not args_str or not args_str.strip():
        return {}

    result: dict[str, Any] = {}
    i = 0
    n = len(args_str)

    while i < n:
        # Skip whitespace and commas
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # Parse key (unquoted, ends at ':')
        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        i += 1  # skip ':'

        # Parse value
        if i >= n:
            result[key] = ""
            break

        # Skip whitespace after ':'
        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            result[key] = ""
            break

        # String value: <|"|>...<|"|>
        if args_str[i:].startswith(SpecialToken.STRING_DELIM):
            i += len(SpecialToken.STRING_DELIM)
            val_start = i
            end_pos = args_str.find(SpecialToken.STRING_DELIM, i)
            if end_pos == -1:
                # Unterminated string — take rest
                result[key] = args_str[val_start:]
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + len(SpecialToken.STRING_DELIM)

        # Nested object: {...}
        elif args_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(SpecialToken.STRING_DELIM):
                    # Skip over string contents to avoid counting { inside strings
                    i += len(SpecialToken.STRING_DELIM)
                    next_delim = args_str.find(SpecialToken.STRING_DELIM, i)
                    i = (
                        n
                        if next_delim == -1
                        else next_delim + len(SpecialToken.STRING_DELIM)
                    )
                    continue
                if args_str[i] == "{":
                    depth += 1
                elif args_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                # Incomplete nested object — use i (not i-1) to avoid
                # dropping the last char, and recurse as partial.
                result[key] = _parse_gemma4_args(
                    args_str[obj_start:i], partial=True
                )
            else:
                result[key] = _parse_gemma4_args(args_str[obj_start : i - 1])

        # Array: [...]
        elif args_str[i] == "[":
            depth = 1
            arr_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(SpecialToken.STRING_DELIM):
                    i += len(SpecialToken.STRING_DELIM)
                    next_delim = args_str.find(SpecialToken.STRING_DELIM, i)
                    i = (
                        n
                        if next_delim == -1
                        else next_delim + len(SpecialToken.STRING_DELIM)
                    )
                    continue
                if args_str[i] == "[":
                    depth += 1
                elif args_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_array(
                    args_str[arr_start:i], partial=True
                )
            else:
                result[key] = _parse_gemma4_array(args_str[arr_start : i - 1])

        # Bare value (number, boolean, etc.)
        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            if partial and i >= n:
                # Value may be incomplete (e.g. partial boolean) —
                # withhold to avoid type instability during streaming.
                break
            result[key] = _parse_gemma4_value(args_str[val_start:i])

    return result


def _parse_gemma4_array(arr_str: str, *, partial: bool = False) -> list[Any]:
    """Parse a Gemma4 array content string into a Python list."""
    items: list[Any] = []
    i = 0
    n = len(arr_str)

    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # String element
        if arr_str[i:].startswith(SpecialToken.STRING_DELIM):
            i += len(SpecialToken.STRING_DELIM)
            end_pos = arr_str.find(SpecialToken.STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + len(SpecialToken.STRING_DELIM)

        # Nested object
        elif arr_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(SpecialToken.STRING_DELIM):
                    i += len(SpecialToken.STRING_DELIM)
                    nd = arr_str.find(SpecialToken.STRING_DELIM, i)
                    i = nd + len(SpecialToken.STRING_DELIM) if nd != -1 else n
                    continue
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(
                    _parse_gemma4_args(arr_str[obj_start:i], partial=True)
                )
            else:
                items.append(_parse_gemma4_args(arr_str[obj_start : i - 1]))

        # Nested array
        elif arr_str[i] == "[":
            depth = 1
            sub_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(
                    _parse_gemma4_array(arr_str[sub_start:i], partial=True)
                )
            else:
                items.append(_parse_gemma4_array(arr_str[sub_start : i - 1]))

        # Bare value
        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            if partial and i >= n:
                break
            items.append(_parse_gemma4_value(arr_str[val_start:i]))

    return items


def _enum_value_rule(
    rule_name: str,
    enum_values: list[Any],
    sd_ref: str,
    rules_parts: list[str],
) -> str:
    """Generate a Lark rule matching only the given enum literals."""
    alternatives: list[str] = []
    for val in enum_values:
        if isinstance(val, bool):
            alternatives.append('"true"' if val else '"false"')
        elif isinstance(val, str):
            alternatives.append(
                f'{sd_ref} "{escape_for_lark_string(val)}" {sd_ref}'
            )
        elif isinstance(val, int):
            alternatives.append(f'"{val}"')
        elif isinstance(val, float):
            alternatives.append(f'"{val}"')
        elif val is None:
            alternatives.append('"null"')
        elif isinstance(val, dict):
            alternatives.append("object_val")
        elif isinstance(val, list):
            alternatives.append("array_val")
    if not alternatives:
        return "value"
    rules_parts.append(f"{rule_name}: " + " | ".join(alternatives))
    return rule_name


def _generate_ordered_args_rule(
    prefix: str,
    prop_rule_names: list[str],
    required: set[str],
    prop_names: list[str],
    rules_parts: list[str],
    ap_value_rule: str | None = None,
) -> str:
    """Generate Lark suffix rules enforcing fixed property order.

    Properties must appear in schema-definition order. Required properties
    cannot be skipped; optional ones may be omitted. Duplicates are
    impossible by construction since each property has exactly one slot.
    """
    n = len(prop_rule_names)

    if ap_value_rule is not None:
        n += 1
        ap_rule_name = f"ap_{prefix}"
        ap_arg_name = f"ap_{prefix}_arg"
        rules_parts.append(f'{ap_arg_name}: KEY ":" {ap_value_rule}')
        rules_parts.append(
            f'{ap_rule_name}: {ap_arg_name} ("," {ap_arg_name})*'
        )
        prop_names.append(ap_rule_name)
        prop_rule_names.append(ap_rule_name)

    if n == 0:
        return ""

    is_req = [name in required for name in prop_names]

    has_req_after = [False] * n
    for i in range(n - 2, -1, -1):
        has_req_after[i] = is_req[i + 1] or has_req_after[i + 1]

    for i in range(n - 1, -1, -1):
        sfx = f"{prefix}_sfx_{i}"
        prop = prop_rule_names[i]

        if i == n - 1:
            rules_parts.append(f"{sfx}: {prop}")
        else:
            next_sfx = f"{prefix}_sfx_{i + 1}"
            if has_req_after[i]:
                if is_req[i]:
                    rules_parts.append(f'{sfx}: {prop} "," {next_sfx}')
                else:
                    rules_parts.append(
                        f'{sfx}: {prop} "," {next_sfx} | {next_sfx}'
                    )
            else:
                if is_req[i]:
                    rules_parts.append(f'{sfx}: {prop} ("," {next_sfx})?')
                else:
                    rules_parts.append(
                        f'{sfx}: {prop} ("," {next_sfx})? | {next_sfx}'
                    )

    top_sfx = f"{prefix}_sfx_0"
    if any(is_req):
        return top_sfx
    return f"{top_sfx}?"


@register("gemma4")
class Gemma4ToolParser(StructuralTagToolParser):
    """Gemma 4 tool parser using flat ``<|tool_call>`` … ``<tool_call|>`` pairs.

    Uses the flat (no-section-wrapper) mode of :class:`StructuralTagToolParser`:
    only ``CALL_BEGIN``/``CALL_END`` are set. Arguments are emitted atomically
    (withheld until the close marker) because Gemma4's ``<|"|>`` string
    delimiters make incremental JSON conversion non-monotonic.
    """

    CALL_BEGIN: ClassVar[str] = SpecialToken.TOOL_CALL_START
    CALL_END: ClassVar[str] = SpecialToken.TOOL_CALL_END

    # ----- StructuralTagToolParser hooks ----------------------------------

    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        tool_calls: list[ParsedToolCall] = []
        for match in TOOL_CALL_PATTERN.finditer(tool_section):
            func_name = match.group(1)
            args_str = match.group(2)
            args_obj = _parse_gemma4_args(args_str)
            tool_calls.append(
                ParsedToolCall(
                    id=generate_call_id(),
                    name=func_name,
                    arguments=json.dumps(args_obj, ensure_ascii=False),
                )
            )
        return tool_calls

    def _split_tool_call_body(
        self, body: str, is_complete: bool
    ) -> tuple[str | None, str | None]:
        prefix = "call:"
        if not body.startswith(prefix):
            return (None, None)
        brace_pos = body.find("{")
        if brace_pos == -1:
            return (None, None)
        header = body[:brace_pos]
        if is_complete and body.endswith("}"):
            args = body[brace_pos + 1 : -1]
        else:
            args = body[brace_pos + 1 :]
        return (header, args)

    def _extract_tool_id_and_name(
        self, header: str
    ) -> tuple[str | None, str | None]:
        prefix = "call:"
        if not header.startswith(prefix):
            return (None, None)
        name = header[len(prefix) :].strip()
        if not name:
            return (None, None)
        return generate_call_id(), name

    def _format_args_for_streaming(
        self, args_text: str, is_complete: bool
    ) -> str:
        if not is_complete:
            return ""
        try:
            args_obj = _parse_gemma4_args(args_text)
            return json.dumps(args_obj, ensure_ascii=False)
        except Exception:
            return "{}"

    # ----- Constrained decoding grammar (Gemma4-specific) ---------------

    @staticmethod
    def _build_func_name_pattern(
        tool_names: list[str] | None = None,
    ) -> str:
        """Return a Lark regex terminal for the function name."""
        if tool_names is not None:
            escaped = [re.escape(n) for n in tool_names]
            return "(" + "|".join(escaped) + ")"
        return r"[a-zA-Z0-9_\-\.]+"

    @staticmethod
    def _resolve_ap_value_rule(
        schema: dict[str, Any],
        rule_prefix: str,
        sd_ref: str,
        rules_parts: list[str],
        depth: int = 0,
    ) -> str | None:
        """Resolve ``additionalProperties`` to a Lark value rule name."""
        ap = schema.get("additionalProperties", False)
        if ap is False:
            return None
        if ap is True:
            return "value"
        return Gemma4ToolParser._generate_property_value_rule(
            ap, f"{rule_prefix}_ap_val", sd_ref, rules_parts, depth
        )

    @staticmethod
    def _generate_property_value_rule(
        prop_schema: dict[str, Any],
        rule_prefix: str,
        sd_ref: str,
        rules_parts: list[str],
        depth: int = 0,
        max_depth: int = 5,
    ) -> str:
        """Return the Lark rule name for a property's value, recursing for objects/arrays."""
        if depth > max_depth:
            return "value"

        enum_values = prop_schema.get("enum")
        if enum_values is not None and len(enum_values) > 0:
            return _enum_value_rule(
                f"{rule_prefix}_enum", enum_values, sd_ref, rules_parts
            )

        any_of = prop_schema.get("anyOf")
        if any_of and isinstance(any_of, list):
            branch_rules: list[str] = []
            for i, branch in enumerate(any_of):
                if isinstance(branch, dict):
                    branch_rules.append(
                        Gemma4ToolParser._generate_property_value_rule(
                            branch,
                            f"{rule_prefix}_branch{i}",
                            sd_ref,
                            rules_parts,
                            depth + 1,
                            max_depth,
                        )
                    )
            unique = list(dict.fromkeys(branch_rules))
            if len(unique) == 1:
                return unique[0]
            anyof_rule = f"{rule_prefix}_anyof"
            rules_parts.append(f"{anyof_rule}: " + " | ".join(unique))
            return anyof_rule

        json_type = prop_schema.get("type", "")

        if isinstance(json_type, list):
            alternatives: list[str] = []
            for t in json_type:
                alt = Gemma4ToolParser._generate_property_value_rule(
                    {**prop_schema, "type": t},
                    f"{rule_prefix}_{t}",
                    sd_ref,
                    rules_parts,
                    depth,
                )
                alternatives.append(alt)
            alternatives = list(dict.fromkeys(alternatives))
            if len(alternatives) == 1:
                return alternatives[0]
            union_rule = f"{rule_prefix}_union"
            rules_parts.append(f"{union_rule}: " + " | ".join(alternatives))
            return union_rule

        if json_type == "object" and _has_schema_constraints(prop_schema):
            nested_props = prop_schema.get("properties", {})
            nested_required = set(prop_schema.get("required", []))
            nested_prop_rules: list[str] = []
            nested_prop_names: list[str] = []
            for nested_name, nested_schema in nested_props.items():
                nested_rule = (
                    f"{rule_prefix}_{canonicalize_lark_rule_name(nested_name)}"
                )
                nested_val = Gemma4ToolParser._generate_property_value_rule(
                    nested_schema,
                    nested_rule,
                    sd_ref,
                    rules_parts,
                    depth + 1,
                )
                rules_parts.append(
                    f'{nested_rule}: "{escape_for_lark_string(nested_name)}" ":" {nested_val}'
                )
                nested_prop_rules.append(nested_rule)
                nested_prop_names.append(nested_name)

            obj_rule = f"{rule_prefix}_obj"
            args_rule = _generate_ordered_args_rule(
                rule_prefix,
                nested_prop_rules,
                nested_required,
                nested_prop_names,
                rules_parts,
                Gemma4ToolParser._resolve_ap_value_rule(
                    prop_schema, rule_prefix, sd_ref, rules_parts, depth + 1
                ),
            )
            rules_parts.append(f'{obj_rule}: "{{" {args_rule} "}}"')
            return obj_rule

        if json_type == "array" and prop_schema.get("items"):
            items_val = Gemma4ToolParser._generate_property_value_rule(
                prop_schema["items"],
                f"{rule_prefix}_item",
                sd_ref,
                rules_parts,
                depth + 1,
            )
            arr_rule = f"{rule_prefix}_arr"
            rules_parts.append(
                f'{arr_rule}: "[" ({items_val} ("," {items_val})*)? "]"'
            )
            return arr_rule

        return grammar_rule_for_json_type(json_type)

    @staticmethod
    def _generate_schema_aware_rules(
        tool_names: list[str],
        tool_schemas: dict[str, dict[str, Any]],
        tcs_ref: str = SpecialToken.TOOL_CALL_START.name,
        tce_ref: str = SpecialToken.TOOL_CALL_END.name,
        sd_ref: str = SpecialToken.STRING_DELIM.name,
    ) -> tuple[str, str]:
        """Generate per-tool argument rules based on parameter schemas."""
        tool_call_alternatives: list[str] = []
        rules_parts: list[str] = []

        for name in set(tool_names):
            safe = canonicalize_lark_rule_name(name)
            schema = tool_schemas.get(name, {})
            properties = schema.get("properties", {})

            if not _has_schema_constraints(schema):
                tool_call_alternatives.append(
                    _tool_call_rule(
                        f'"{escape_for_lark_string(name)}"',
                        "args_body",
                        tcs_ref,
                        tce_ref,
                    )
                )
                continue

            prefix = f"tc_{safe}"
            required = set(schema.get("required", []))
            prop_rule_names: list[str] = []
            prop_names: list[str] = []
            for prop_name, prop_schema in properties.items():
                rule_name = f"{prefix}_{canonicalize_lark_rule_name(prop_name)}"
                value_rule = Gemma4ToolParser._generate_property_value_rule(
                    prop_schema, rule_name, sd_ref, rules_parts
                )
                rules_parts.append(
                    f'{rule_name}: "{escape_for_lark_string(prop_name)}" ":" {value_rule}'
                )
                prop_rule_names.append(rule_name)
                prop_names.append(prop_name)

            args_rule = _generate_ordered_args_rule(
                prefix,
                prop_rule_names,
                required,
                prop_names,
                rules_parts,
                Gemma4ToolParser._resolve_ap_value_rule(
                    schema, prefix, sd_ref, rules_parts
                ),
            )
            tool_call_alternatives.append(
                _tool_call_rule(
                    f'"{escape_for_lark_string(name)}"',
                    args_rule,
                    tcs_ref,
                    tce_ref,
                )
            )

        tool_call_rule = "tool_call: " + " | ".join(tool_call_alternatives)
        extra_rules = "\n".join(rules_parts)
        return tool_call_rule, extra_rules

    @staticmethod
    def _get_special_token_ids(
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> dict[str, int] | None:
        """Resolve Gemma4 special token IDs from the tokenizer."""
        result: dict[str, int] = {}
        for token in SpecialToken:
            tid = get_token_id(tokenizer, token.value)
            if tid is not None:
                result[token.name] = tid
        return result if result else None

    @staticmethod
    def generate_tool_call_grammar(
        response_format_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tokenizer: PipelineTokenizer[Any, Any, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generates a Lark grammar for constrained decoding of Gemma4 tool calls."""
        tool_names = names_from_tools(tools)

        special_token_ids = (
            Gemma4ToolParser._get_special_token_ids(tokenizer)
            if tokenizer is not None
            else None
        )
        if not special_token_ids:
            raise ValueError(
                "tokenizer is required for grammar generation; "
                "it must resolve Gemma4 special token IDs"
            )
        tool_schemas = _extract_tool_schemas(tools) if tools else None

        sd_ref = resolve_lark_token_reference(
            special_token_ids[SpecialToken.STRING_DELIM.name]
        )
        tcs_ref = resolve_lark_token_reference(
            special_token_ids[SpecialToken.TOOL_CALL_START.name]
        )
        tce_ref = resolve_lark_token_reference(
            special_token_ids[SpecialToken.TOOL_CALL_END.name]
        )
        te_ref = resolve_lark_token_reference(
            special_token_ids[SpecialToken.TURN_END.name]
        )
        trs_ref = resolve_lark_token_reference(
            special_token_ids[SpecialToken.TOOL_RESPONSE_START.name]
        )

        use_schema_aware = (
            tool_schemas is not None
            and tool_names is not None
            and any(
                _has_schema_constraints(tool_schemas.get(n, {}))
                for n in tool_names
            )
        )

        if use_schema_aware:
            assert tool_names is not None
            assert tool_schemas is not None
            tool_call_rule, schema_rules = (
                Gemma4ToolParser._generate_schema_aware_rules(
                    tool_names, tool_schemas, tcs_ref, tce_ref, sd_ref
                )
            )
            func_name_terminal = ""
        else:
            func_name_pattern = Gemma4ToolParser._build_func_name_pattern(
                tool_names
            )
            tool_call_rule = "tool_call: " + _tool_call_rule(
                "FUNC_NAME", "args_body", tcs_ref, tce_ref
            )
            schema_rules = ""
            func_name_terminal = f"FUNC_NAME: /{func_name_pattern}/"

        rules = [
            f"tool_calls: tool_call+ ({te_ref} | {trs_ref})",
            tool_call_rule,
            schema_rules,
            'args_body: (arg ("," arg)*)?',
            'arg: KEY ":" value',
            "value: string_val | number_val | bool_val | object_val | array_val | null_val",
            f"string_val: {sd_ref} STRING_CONTENT {sd_ref}",
            "number_val: NUMBER",
            "integer_val: INTEGER",
            "bool_val: BOOL",
            'null_val: "null"',
            'object_val: "{" args_body "}"',
            'array_val: "[" (value ("," value)*)? "]"',
        ]
        terminals = [
            r"STRING_CONTENT: /[\s\S]*/",
            func_name_terminal,
            r"KEY: /[a-zA-Z_][-a-zA-Z0-9_.]*/",
            r"NUMBER: /\-?[0-9]+(\.[0-9]+)?([eE][\+\-]?[0-9]+)?/",
            r"INTEGER: /\-?[0-9]+([eE][\+\-]?[0-9]+)?/",
            'BOOL: "true" | "false"',
        ]
        rule_lines = "\n".join(line for line in rules if line)
        terminal_lines = "\n".join(line for line in terminals if line)
        tool_grammar = f"\n{rule_lines}\n\n{terminal_lines}\n"

        if response_format_schema is None:
            return f"\nstart: tool_calls\n{tool_grammar}"

        schema_with_opts = {
            **response_format_schema,
            "x-guidance": {"whitespace_pattern": ""},
        }
        schema_json = json.dumps(schema_with_opts, separators=(",", ":"))
        return (
            f"\nstart: tool_calls | json_response\n"
            f"json_response: %json {schema_json}\n{tool_grammar}"
        )
