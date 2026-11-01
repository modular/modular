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

"""Maps checkpoint weights onto graph weights with declarative rules."""

from __future__ import annotations

import dataclasses
import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from typing_extensions import Self

__all__ = [
    "Bitcast",
    "CheckpointRole",
    "Ignore",
    "WeightEntry",
    "WeightNaming",
    "WeightRenaming",
    "WeightRule",
    "WeightTransform",
    "apply_weight_entries",
]

logger = logging.getLogger("max.pipelines")

# Mach applies serialized rules with Rust's regex crate, which has neither
# look-around nor back-references.
_UNPORTABLE_SYNTAX = re.compile(r"\(\?(?:<?[=!]|P=)|\\[1-9]")


@dataclass(frozen=True)
class WeightRenaming:
    r"""Renames weights by regular expression substitution.

    Replaces every match of :attr:`pattern` in a weight's name, as
    :func:`re.sub` does, so :attr:`replacement` can refer to groups as
    ``\1`` or ``\g<name>``. A pattern matches anywhere in the name unless
    it is anchored.

    Raises:
        ValueError: If :attr:`pattern` is not a regular expression, uses
            look-around or a back-reference, or can match the empty string.
    """

    pattern: str
    """The regular expression to replace."""

    replacement: str
    """The text that replaces each match."""

    _regex: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        regex = _compile(self.pattern)
        # re.sub replaces an empty match right after a non-empty one, where
        # Rust's replace_all skips it, so Mach would rename differently.
        if regex.fullmatch("") is not None:
            raise ValueError(
                f"pattern {self.pattern!r} can match the empty string, which"
                " re.sub and Rust's replace_all substitute differently"
            )
        object.__setattr__(self, "_regex", regex)

    @classmethod
    def prefix(cls, prefix: str) -> Self:
        """Returns the rule that puts every name under ``prefix``.

        A pipeline model appends it to an entry's rules to place a module in
        a graph, as a speculative-decoding graph places its target under
        ``target.``.

        Args:
            prefix: The text every name starts with afterwards.
        """
        return cls(r"^(.+)$", prefix.replace("\\", "\\\\") + r"\1")


@dataclass(frozen=True)
class Bitcast:
    """Reinterprets the bytes of matching weights as another dtype.

    Applies only to weights stored as :attr:`src`, such as block scales that
    a checkpoint stores as raw ``uint8`` bytes. The result is a view of the
    same memory, so the shape and the bytes are unchanged.

    Raises:
        ValueError: If :attr:`pattern` is not a regular expression or uses
            look-around or a back-reference, or :attr:`src` and :attr:`dst`
            differ in width.
    """

    pattern: str
    """The regular expression a weight's name must match."""

    src: DType
    """The dtype a weight must be stored as."""

    dst: DType
    """The dtype the weight's bytes are reinterpreted as."""

    _regex: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.src.size_in_bits != self.dst.size_in_bits:
            raise ValueError(
                f"cannot bitcast {self.src} to {self.dst}: their elements"
                " differ in width"
            )
        object.__setattr__(self, "_regex", _compile(self.pattern))


@dataclass(frozen=True)
class Ignore:
    """Drops matching weights, for a stated reason.

    Raises:
        ValueError: If :attr:`pattern` is not a regular expression or uses
            look-around or a back-reference, or :attr:`reason` is empty.
    """

    pattern: str
    """The regular expression a weight's name must match."""

    reason: str
    """Why the graph does not load these weights, shown in the load log."""

    _regex: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError(f"Ignore({self.pattern!r}) needs a reason")
        object.__setattr__(self, "_regex", _compile(self.pattern))


WeightRule: TypeAlias = WeightRenaming | Bitcast | Ignore
"""One step of a :class:`WeightTransform`."""


@dataclass(frozen=True)
class WeightTransform:
    """Maps a checkpoint's weights onto the names and dtypes a graph loads.

    Each checkpoint weight goes through the rules in order, and each rule
    sees the name the rules before it produced. An :class:`Ignore` drops the
    weight and ends its rules. A weight that no rule touches keeps its name.
    No rule copies data: the adapted weights are views of the checkpoint's.

    A transform holds only patterns and plain values, so it serializes to
    plain data and back. Patterns stay within the syntax Python's :mod:`re`
    shares with Rust's ``regex`` crate, so Mach can apply the same rules:
    look-around and back-references are refused.

    Args:
        rules: The rules, in the order they apply.
    """

    rules: Sequence[WeightRule]
    """The rules, in the order they apply."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", tuple(self.rules))

    def __call__(self, weights: Mapping[str, Weights]) -> dict[str, WeightData]:
        """Returns the weights no rule ignores, under their adapted names.

        Args:
            weights: The checkpoint's weights, keyed by checkpoint name.

        Returns:
            The adapted weights, keyed by adapted name.

        Raises:
            ValueError: If two checkpoint weights adapt to the same name.
        """
        adapted: dict[str, WeightData] = {}
        sources: dict[str, str] = {}
        ignored: Counter[Ignore] = Counter()
        for source, weight in weights.items():
            result = self._adapt(source, weight)
            if isinstance(result, Ignore):
                ignored[result] += 1
                continue
            name, data = result
            if name in adapted:
                raise ValueError(
                    f"checkpoint weights {sources[name]!r} and {source!r}"
                    f" both adapt to {name!r}"
                )
            adapted[name] = data
            sources[name] = source
        for rule, count in ignored.items():
            logger.info(
                "Ignored %d checkpoint weights matching %r: %s",
                count,
                rule.pattern,
                rule.reason,
            )
        return adapted

    def _adapt(
        self, name: str, weight: Weights
    ) -> tuple[str, WeightData] | Ignore:
        """Returns the adapted name and data, or the Ignore that drops it."""
        # Loaded only when a rule or the result needs it, so a weight that is
        # ignored by name is never read.
        data: WeightData | None = None
        for rule in self.rules:
            if isinstance(rule, WeightRenaming):
                name = rule._regex.sub(rule.replacement, name)
            elif not rule._regex.search(name):
                continue
            elif isinstance(rule, Ignore):
                return rule
            else:
                if data is None:
                    data = weight.data()
                if data.dtype == rule.src:
                    data = _bitcast(data, rule.dst)
        if data is None:
            data = weight.data()
        return name, data

    def __serialize__(self) -> dict[str, list[dict[str, str]]]:
        """Returns the rules as plain data, in order."""
        return {"rules": [_rule_to_json(rule) for rule in self.rules]}

    @classmethod
    def __deserialize__(cls, value: Mapping[str, object]) -> Self:
        """Rebuilds a transform from the output of :meth:`__serialize__`.

        Args:
            value: The serialized transform.

        Returns:
            A transform equal to the one serialized.

        Raises:
            ValueError: If the rules are missing or a rule is malformed.
        """
        rules = value.get("rules")
        if not isinstance(rules, list):
            raise ValueError(f"rules must be a list, got {rules!r}")
        return cls([_rule_from_json(record) for record in rules])


class CheckpointRole(str, Enum):
    """A checkpoint, by its role in the pipeline.

    An artifact names checkpoints by role rather than path, so whoever serves
    it supplies each one.
    """

    TARGET = "target"
    """The pipeline's model."""

    DRAFT = "draft"
    """The speculative-decoding draft model's own checkpoint."""


@dataclass(frozen=True)
class WeightEntry:
    """Names one checkpoint's weights in a graph.

    A graph binds the names it declares and skips the rest, so an entry
    whose rules name a whole checkpoint serves every graph that loads part
    of it.
    """

    checkpoint: CheckpointRole
    """The checkpoint whose weights the entry names."""

    transform: WeightTransform
    """The rules that name them."""

    def with_prefix(self, prefix: str) -> WeightEntry:
        """Returns the entry with every name under ``prefix``.

        Args:
            prefix: The text every name starts with afterwards.
        """
        return WeightEntry(
            self.checkpoint,
            WeightTransform(
                [*self.transform.rules, WeightRenaming.prefix(prefix)]
            ),
        )

    def __serialize__(self) -> dict[str, object]:
        """Returns the checkpoint and the rules as plain data."""
        return {
            "checkpoint": self.checkpoint.value,
            **self.transform.__serialize__(),
        }

    @classmethod
    def __deserialize__(cls, value: Mapping[str, object]) -> Self:
        """Rebuilds an entry from the output of :meth:`__serialize__`.

        Args:
            value: The serialized entry.

        Returns:
            An entry equal to the one serialized.

        Raises:
            ValueError: If a field is missing or unknown, the checkpoint is
                not a :class:`CheckpointRole`, or a rule is malformed.
        """
        if value.keys() != {"checkpoint", "rules"}:
            raise ValueError(
                "an entry takes fields ['checkpoint', 'rules'], got"
                f" {sorted(value)}"
            )
        try:
            checkpoint = CheckpointRole(value["checkpoint"])
        except ValueError as e:
            raise ValueError(
                f"unknown checkpoint {value['checkpoint']!r}"
            ) from e
        transform = WeightTransform.__deserialize__({"rules": value["rules"]})
        return cls(checkpoint, transform)


def apply_weight_entries(
    entries: Sequence[WeightEntry],
    checkpoints: Mapping[CheckpointRole, Mapping[str, Weights]],
) -> dict[str, WeightData]:
    """Returns the weights a graph's entries name.

    Args:
        entries: The graph's entries.
        checkpoints: The weights of the checkpoints the entries name, keyed by
            role and then by checkpoint name.

    Returns:
        Every entry's weights, keyed by the names the entries give them.

    Raises:
        ValueError: If two weights get one name, or an entry names a
            checkpoint ``checkpoints`` lacks.
    """
    weights: dict[str, WeightData] = {}
    for entry in entries:
        if entry.checkpoint not in checkpoints:
            raise ValueError(
                f"an entry names the {entry.checkpoint.value} checkpoint,"
                " which the pipeline does not load"
            )
        named = entry.transform(checkpoints[entry.checkpoint])
        if repeated := sorted(named.keys() & weights.keys()):
            raise ValueError(f"two entries name the weights {repeated[:3]}")
        weights.update(named)
    return weights


@dataclass(frozen=True)
class WeightNaming:
    """The entries that name each graph's weights, keyed by graph role.

    A pipeline artifact stores it as its ``weight_loader`` section, so the
    artifact alone names every graph's weights.
    """

    graphs: Mapping[str, Sequence[WeightEntry]]
    """Each graph's entries, keyed by role."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "graphs",
            {role: tuple(entries) for role, entries in self.graphs.items()},
        )

    def __serialize__(self) -> dict[str, list[dict[str, object]]]:
        """Returns each graph's entries as plain data."""
        return {
            role: [entry.__serialize__() for entry in entries]
            for role, entries in sorted(self.graphs.items())
        }

    @classmethod
    def __deserialize__(cls, value: Mapping[str, object]) -> Self:
        """Rebuilds the naming from the output of :meth:`__serialize__`.

        Args:
            value: The serialized naming.

        Returns:
            A naming equal to the one serialized.

        Raises:
            ValueError: If a graph's entries are not a list of entries, or an
                entry is malformed.
        """
        graphs: dict[str, tuple[WeightEntry, ...]] = {}
        for role, entries in value.items():
            if not isinstance(entries, list) or not all(
                isinstance(entry, Mapping) for entry in entries
            ):
                raise ValueError(
                    f"{role} must be a list of entries, got {entries!r}"
                )
            graphs[role] = tuple(
                WeightEntry.__deserialize__(entry) for entry in entries
            )
        return cls(graphs)


def _compile(pattern: str) -> re.Pattern[str]:
    if _UNPORTABLE_SYNTAX.search(pattern):
        raise ValueError(
            f"pattern {pattern!r} uses look-around or a back-reference,"
            " which Rust's regex crate does not support"
        )
    try:
        return re.compile(pattern)
    except re.error as e:
        raise ValueError(f"invalid pattern {pattern!r}: {e}") from e


def _bitcast(data: WeightData, dtype: DType) -> WeightData:
    buffer = (
        data.data
        if isinstance(data.data, Buffer)
        else Buffer.from_dlpack(data.data)
    )
    # Relabel both: max.nn checks the WeightData's dtype and ModuleV3 the
    # buffer's.
    return dataclasses.replace(data, data=buffer.view(dtype), dtype=dtype)


def _rule_to_json(rule: WeightRule) -> dict[str, str]:
    if isinstance(rule, WeightRenaming):
        return {
            "rule": "WeightRenaming",
            "pattern": rule.pattern,
            "replacement": rule.replacement,
        }
    if isinstance(rule, Bitcast):
        return {
            "rule": "Bitcast",
            "pattern": rule.pattern,
            "src": rule.src.name,
            "dst": rule.dst.name,
        }
    return {"rule": "Ignore", "pattern": rule.pattern, "reason": rule.reason}


_RULE_FIELDS = {
    "WeightRenaming": {"pattern", "replacement"},
    "Bitcast": {"pattern", "src", "dst"},
    "Ignore": {"pattern", "reason"},
}


def _rule_from_json(record: object) -> WeightRule:
    if not isinstance(record, dict) or not all(
        isinstance(value, str) for value in record.values()
    ):
        raise ValueError(f"a rule must be an object of strings: {record!r}")
    fields = dict(record)
    kind = fields.pop("rule", None)
    expected = _RULE_FIELDS.get(kind) if isinstance(kind, str) else None
    if expected is None:
        raise ValueError(f"unknown rule {kind!r} in {record!r}")
    if fields.keys() != expected:
        raise ValueError(
            f"{kind} takes fields {sorted(expected)}, got {record!r}"
        )
    if kind == "WeightRenaming":
        return WeightRenaming(fields["pattern"], fields["replacement"])
    if kind == "Ignore":
        return Ignore(fields["pattern"], fields["reason"])
    try:
        src, dst = DType[fields["src"]], DType[fields["dst"]]
    except KeyError as e:
        raise ValueError(f"unknown dtype in {record!r}") from e
    return Bitcast(fields["pattern"], src, dst)
