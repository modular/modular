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

"""Guard rails for running eager code in production."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from types import TracebackType
from typing import Any, Protocol, TypeVar

from max import _validation_hooks
from max._validation_hooks import active_validator
from max.driver import Buffer, Device
from max.engine import Model
from max.experimental.realization_context import (
    EagerRealizationContext,
    set_default_realization_context,
)
from max.experimental.tensor import Tensor

__all__ = [
    "EagerUsageValidator",
    "active_validator",
]

_logger = logging.getLogger("max.pipelines")

# Packages to pass over when finding the call site in the stack trace.
_FRAMEWORK_MODULES = (
    "max._core",
    "max._validation_hooks",
    "max.driver",
    "max.engine",
    "max.experimental",
    "max.graph",
    "asyncio",
    "concurrent.",
    "contextlib",
    "functools",
)

# Maximum number of reported findings per category.
_MAX_REPORTED = 20


class _Reportable(Protocol):
    """A finding that remembers whether the validator has logged it."""

    reported: bool


_Finding = TypeVar("_Finding", bound=_Reportable)


def _call_site() -> str:
    """Returns ``file:line`` of the innermost frame outside the framework."""
    frame: Any = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if not module.startswith(_FRAMEWORK_MODULES):
            filename = os.path.basename(frame.f_code.co_filename)
            return f"{filename}:{frame.f_lineno}"
        frame = frame.f_back
    return "<unknown>"


def _buffers(value: Any) -> Iterator[Buffer]:
    """Yields the device buffers reachable through ``value``."""
    if isinstance(value, Tensor):
        if value._storages is not None:
            yield from value._storages
    elif isinstance(value, Buffer):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _buffers(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _buffers(item)


def _describe(value: Tensor | Buffer) -> str:
    """Describes a tensor's or buffer's shape, dtype and device."""
    shape = ", ".join(str(dim) for dim in value.shape)
    # ``Tensor.device`` raises across multiple devices, and a distributed
    # tensor is exactly what the transfer findings are most often about.
    if isinstance(value, Tensor) and value.is_distributed:
        devices = ", ".join(str(d) for d in value._mapping.mesh.devices)
        return f"[{shape}] {value.dtype.name} across {devices}"
    return f"[{shape}] {value.dtype.name} on {value.device}"


@dataclass
class _EagerSite:
    """Eager executions traced back to one call site."""

    site: str
    count: int = 0
    reported: bool = False


@dataclass
class _GraphBreak:
    """A transition from one compiled graph into another."""

    source: int
    target: int
    site: str
    materialized: int
    count: int = 0
    reported: bool = False


@dataclass
class _TrackedTransfer:
    """Device-to-host copies traced back to one call site."""

    api: str
    site: str
    description: str
    count: int = 0
    reported: bool = False


@dataclass
class _UnreadOutput:
    """A compiled-call output that went unread for a whole scope."""

    graph: int
    index: int
    description: str
    site: str
    reported: bool = False


@dataclass(frozen=True)
class _PendingOutput:
    """A compiled-call output awaiting evidence that something reads it.

    A question, not yet a finding: most are answered by the next op that
    reads the buffer. Only what a scope ends still holding becomes one.
    """

    graph: int
    index: int
    description: str
    site: str
    buffer: Buffer
    """Retained so that no later object lands on the same ``id``.

    Without it an unread output's buffer is free to be collected and
    ``_mark_used`` pops whatever reused its address, dropping a real
    finding.
    """

    def unread(self) -> _UnreadOutput:
        """Returns the finding this output becomes if nothing reads it."""
        return _UnreadOutput(
            self.graph, self.index, self.description, self.site
        )


@dataclass(frozen=True)
class _TrackedGraph:
    """A compiled model that the validator has seen called."""

    model: Model
    """Retained so no later object lands on the same ``id``."""

    name: str
    ordinal: int
    """Which model of this name it is, counting from one."""


@dataclass
class _Findings:
    """Every finding a validator has made, for as long as it lives.

    One entry per call site, so re-entering the validator -- which a
    pipeline does once per execution -- adds to what is here rather than
    starting it over.
    """

    eager: dict[str, _EagerSite] = field(default_factory=dict)
    breaks: dict[tuple[int, int, str], _GraphBreak] = field(
        default_factory=dict
    )
    outputs: dict[tuple[int, int, str], _UnreadOutput] = field(
        default_factory=dict
    )

    transfers: dict[tuple[str, str], _TrackedTransfer] = field(
        default_factory=dict
    )
    eager_total: int = 0
    transfer_total: int = 0
    """Executions and transfers seen, whether or not their site is reported."""


class _ValidatingRealizationContext(EagerRealizationContext):
    """An eager context that reports each Tensor realization to the validator.

    Stores the call site that triggered the realization.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._validator = active_validator()

    async def realize_all(self) -> list[Tensor]:
        validator = self._validator
        if validator is None:
            return await super().realize_all()
        # Both are read before ``super()``: the frames naming the caller
        # unwind once it returns.
        site = _call_site()
        # A realized tensor this graph reads is a graph input, which is the
        # evidence that something went on to use it.
        sources = list(self.sources.values())
        realized = await super().realize_all()
        validator._realized(site, sources)
        return realized


class EagerUsageValidator:
    """A scope that tracks production-unsafe API usages.

    Reports:

    1. Eager ops that ran without being fused into a compiled graph.
    2. Graph breaks between compiled graphs.
    3. Compiled outputs that were materialized and never read.
    4. Device-to-host transfers that were not bulk transferred.


    .. code-block:: python

        from max.experimental.validation import EagerUsageValidator

        with EagerUsageValidator():
            ...

    To mark where unsafe API usage is deliberate, use escape hatches. Example:

    .. skip: next

    .. code-block:: python

        from max.pipelines.modeling.eager_validation import eager_validator

        with eager_validator.allow_eager(reason="input batching, off hot path"):
            ...

    Args:
        enabled: When ``False`` the scope does nothing at all, so a caller
            can construct one unconditionally from a config flag.
        label: Names the scope in warnings, e.g. ``"initialization"``.
    """

    def __init__(self, *, enabled: bool = True, label: str = "") -> None:
        self._enabled = enabled
        self.label = label
        self._findings = _Findings()
        self._pending_outputs: dict[int, _PendingOutput] = {}
        """Compiled model outputs that have been materialized but not read."""
        self._graphs: dict[int, _TrackedGraph] = {}
        """The models called within this scope."""
        self._last_graph: int | None = None
        self._last_graph_outputs = 0
        self._allow_eager = 0
        self._allow_transfer = 0
        self._allow_break = 0
        self._exits: list[Any] = []

    @property
    def enabled(self) -> bool:
        """Whether this validator is enabled."""
        return self._enabled

    # Scope management.

    def enter(self) -> EagerUsageValidator:
        """Starts validating. Prefer the context manager form."""
        if not self.enabled:
            return self
        with contextlib.ExitStack() as stack:
            if not self._exits:
                # The findings outlive the scope; the working set does not.
                self._pending_outputs.clear()
                self._graphs.clear()
                self._last_graph = None
                self._last_graph_outputs = 0
                stack.enter_context(_validation_hooks.register(self))
                stack.enter_context(
                    set_default_realization_context(
                        _ValidatingRealizationContext
                    )
                )
            # Only once every context is in: a failure part-way through
            # unwinds with the block rather than leaking a validator.
            self._exits.append(stack.pop_all())
        return self

    def exit(
        self,
        exc_type: type[BaseException] | None = None,
        exc: BaseException | None = None,
        tb: TracebackType | None = None,
    ) -> None:
        """Stops validating and reports what the scope saw."""
        if not self._exits:
            return
        self._exits.pop().close()
        if not self._exits and exc is None:
            # Mark any leftover outputs that were never consumed.
            for pending in self._pending_outputs.values():
                key = (pending.graph, pending.index, pending.site)
                self._findings.outputs.setdefault(key, pending.unread())
            _report_findings(self.label, self._findings, self._graphs)

    def __enter__(self) -> EagerUsageValidator:
        return self.enter()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.exit(exc_type, exc, tb)

    # Escape hatches.

    @contextlib.contextmanager
    def allow_eager(self, *, reason: str) -> Iterator[None]:
        """Opts in to eager execution for the duration of the block.

        Args:
            reason: Why eager execution is the right call here. Recorded for
                the reader, not consulted by the validator.
        """
        del reason
        self._allow_eager += 1
        try:
            yield
        finally:
            self._allow_eager -= 1

    @contextlib.contextmanager
    def graph_break(self, *, reason: str) -> Iterator[None]:
        """Marks the block as an expected boundary between compiled graphs.

        Args:
            reason: Why the model is compiled in pieces here.
        """
        del reason
        self._allow_break += 1
        try:
            yield
        finally:
            self._allow_break -= 1
            # The next call starts a fresh chain: whatever ran inside the
            # block is not a transition anything outside it should answer for.
            self._last_graph = None

    @contextlib.contextmanager
    def allow_device_transfer(self, *, reason: str) -> Iterator[None]:
        """Opts in to device-to-host transfers for the duration of the block.

        Prefer one bulk transfer inside such a block over a transfer per
        element.

        Args:
            reason: Why the host needs this data.
        """
        del reason
        self._allow_transfer += 1
        try:
            yield
        finally:
            self._allow_transfer -= 1

    def discard_output(self, output: Any, *, reason: str) -> None:
        """Marks compiled-model outputs as intentionally unread.

        Args:
            output: A tensor or buffer, or any nesting of tuples, lists and
                dicts holding the outputs to excuse.
            reason: Why the graph produces an output nothing reads.
        """
        del reason
        self._mark_used(output)

    # The following methods are called from max._validation_hooks.

    def _realized(self, site: str, sources: Sequence[Tensor]) -> None:
        """Records one eager realization, called by the installed context."""
        self._mark_used(sources)
        if self._allow_eager:
            return
        findings = self._findings
        findings.eager_total += 1
        eager_site = findings.eager.get(site) or _EagerSite(site)
        findings.eager[site] = replace(eager_site, count=eager_site.count + 1)

    def _device_transfer(self, api: str, tensor: Tensor, dest: Device) -> None:
        self._mark_used(tensor)
        if self._allow_transfer:
            return
        source = tensor._mapping.mesh.devices[0]
        # We only want to warn transfers from device to host, so skip
        # other types of transfers.
        if source.is_host or not dest.is_host:
            return
        site = _call_site()
        described = _describe(tensor)
        self._findings.transfer_total += 1
        key = (api, site)
        found = self._findings.transfers.get(key) or _TrackedTransfer(
            api, site, described
        )
        self._findings.transfers[key] = replace(found, count=found.count + 1)

    def _compiled_call(
        self, engine_model: Model, args: Any, outputs: Any
    ) -> None:
        self._mark_used(args)
        site = _call_site()
        graph = self._register_compiled_graph(engine_model)
        if (
            self._last_graph is not None
            and self._last_graph != graph
            and not self._allow_break
        ):
            key = (self._last_graph, graph, site)
            found = self._findings.breaks.get(key) or _GraphBreak(
                self._last_graph, graph, site, self._last_graph_outputs
            )
            self._findings.breaks[key] = replace(found, count=found.count + 1)

        registered = 0
        for index, buffer in enumerate(_buffers(outputs)):
            registered += 1
            # Not a report cap but a memory one: a pending output pins its
            # buffer until something reads it.
            if len(self._pending_outputs) >= _MAX_REPORTED:
                continue
            self._pending_outputs[id(buffer)] = _PendingOutput(
                graph=graph,
                index=index,
                description=_describe(buffer),
                site=site,
                buffer=buffer,
            )
        self._last_graph = graph
        self._last_graph_outputs = registered

    def _register_compiled_graph(self, engine_model: Model) -> int:
        """Registers a compiled graph and returns the id identifying it."""
        key = id(engine_model)
        if key not in self._graphs:
            name = engine_model.name
            ordinal = (
                sum(1 for g in self._graphs.values() if g.name == name) + 1
            )
            self._graphs[key] = _TrackedGraph(
                model=engine_model, name=name, ordinal=ordinal
            )
        return key

    def _mark_used(self, values: Any) -> None:
        """Marks compiled outputs reachable from ``values`` as read."""
        if not self._pending_outputs:
            return
        for buffer in _buffers(values):
            self._pending_outputs.pop(id(buffer), None)


def _report_findings(
    label: str,
    findings: _Findings,
    graphs: dict[int, _TrackedGraph],
) -> None:
    scope = f" during {label}" if label else ""
    lines = [f"Eager usage validator found{scope}:"]
    lines.extend(_eager_warning(findings))
    lines.extend(_graph_break_warning(findings, graphs))
    lines.extend(_unused_output_lines(findings, graphs))
    lines.extend(_transfer_warning(findings))
    if len(lines) > 1:
        _logger.warning("\n".join(lines))


def _graph_name(graphs: dict[int, _TrackedGraph], graph: int) -> str:
    """Returns the name of a graph."""
    found = graphs.get(graph)
    if found is None:
        return "<unknown>"
    collides = any(
        other.name == found.name and other.ordinal != found.ordinal
        for other in graphs.values()
    )
    return f"{found.name}#{found.ordinal}" if collides else found.name


def _eager_warning(findings: _Findings) -> list[str]:
    eager_sites = sorted(findings.eager.values(), key=lambda t: -t.count)
    lines = _format_findings(
        eager_sites, lambda site: f"    {site.site} x{site.count}"
    )
    if not lines:
        return []
    return [
        f"  {findings.eager_total} eager execution(s). Each "
        "compiles and launches a graph of its own, so nothing fuses "
        "across them; move the work into a compiled function, or mark "
        "it with `eager_validator.allow_eager(reason=...)`.",
        *lines,
    ]


def _graph_break_warning(
    findings: _Findings, graphs: dict[int, _TrackedGraph]
) -> list[str]:
    lines = _format_findings(
        findings.breaks.values(),
        lambda found: (
            f"    {_graph_name(graphs, found.source)} -> "
            f"{_graph_name(graphs, found.target)} at {found.site} "
            f"({found.materialized} tensor(s) materialized, "
            f"x{found.count})"
        ),
    )
    if not lines:
        return []
    return [
        "  Execution crossed a compiled-graph boundary. Each crossing "
        "materializes the tensors between the graphs and blocks fusion "
        "across them; compile the pieces as one graph, or mark the "
        "boundary with `eager_validator.graph_break(reason=...)`.",
        *lines,
    ]


def _unused_output_lines(
    findings: _Findings, graphs: dict[int, _TrackedGraph]
) -> list[str]:
    lines = _format_findings(
        findings.outputs.values(),
        lambda unread: (
            f"    {_graph_name(graphs, unread.graph)} "
            f"output[{unread.index}] {unread.description} "
            f"produced at {unread.site}"
        ),
    )
    if not lines:
        return []
    return [
        "  Compiled-graph outputs were materialized and never read. Drop "
        "them from the graph's outputs, or mark them with "
        "`eager_validator.discard_output(output, reason=...)`.",
        *lines,
    ]


def _transfer_warning(findings: _Findings) -> list[str]:
    transfers = sorted(findings.transfers.values(), key=lambda t: -t.count)
    lines = _format_findings(
        transfers,
        lambda transfer: (
            f"    {transfer.api} at {transfer.site} copies "
            f"{transfer.description} x{transfer.count}"
        ),
    )
    if not lines:
        return []
    return [
        f"  {findings.transfer_total} device-to-host transfer(s). "
        "Each one waits on the accelerator before the host can go on; "
        "keep the value on device, or mark the call site with "
        "`eager_validator.allow_device_transfer(reason=...)`.",
        *lines,
    ]


def _format_findings(
    findings: Iterable[_Finding],
    describe: Callable[[_Finding], str],
) -> list[str]:
    """Filters and formats a list of findings into a list of strings."""
    lines: list[str] = []
    for finding in findings:
        if finding.reported:
            continue
        if len(lines) >= _MAX_REPORTED:
            break
        finding.reported = True
        lines.append(describe(finding))
    return lines
