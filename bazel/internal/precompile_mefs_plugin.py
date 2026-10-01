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

"""Records a GPU test's graph compiles on CPU, then replays them on the GPU.

Two runs of the same pytest invocation, told apart only by the environment
``pytest_record.bzl`` and ``modular_py_test`` set:

- **record**, as a CPU build action. Virtual devices of the lane's arch stand
  in for the accelerator, so every ``session.load`` compiles and writes its
  artifact, and the test stops at the first thing that needs real hardware --
  the expected end of a recorded test, not a failure. What each test and
  fixture exported, and where it stopped, goes in ``record.json``.
- **replay**, as the test. The artifacts are in the runfiles, so every compile
  is answered with the one recorded for it. ``record.json`` decides what a miss
  means: a test that recorded graphs fails loudly, and a test that recorded
  none declines the store and compiles on the GPU as before.

What identifies an artifact is which test or fixture compiled it and in what
order (see :class:`_Units`), which both runs agree on because they run the same
code from the same revision. Nothing correlates a record
shard with a test shard, so the two may be sharded differently.

A test that cannot be recorded -- one that stages device buffers before its
first load, say -- is not a failure and needs no annotation: it compiles on the
GPU and the summary says so. ``@pytest.mark.compile_on_gpu(reason)`` says so
deliberately, for a test whose recording would be misleading rather than
impossible.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
from collections import defaultdict
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import pytest

# pytest does not re-export the reporter its own summary hook is handed.
from _pytest.terminal import TerminalReporter
from max.driver import (
    set_virtual_cpu_target,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.engine import (
    ArtifactBodyMismatch,
    CompileOnlyExecutionError,
    InferenceSession,
    MefStore,
    MissingArtifactError,
)

# Private, and deliberately: what a graph's name and signature digest to is the
# store's business, and this only extends it.
from max.engine._precompiled_mefs import _default_key
from python.runfiles import runfiles

if TYPE_CHECKING:
    from max.graph import Graph

_RECORD_NAME = "record.json"
# What the engine calls a compiled artifact, which the key below extends.
_MEF_SUFFIX = ".mef"
_RECORD_VERSION = 1
_MARKER = "compile_on_gpu"

# Where a test got to under virtual devices, which is what replay reads back.
_RECORDED = "recorded"  # exported artifacts, then needed hardware
_PASSED = "passed"  # ran to the end, whether or not it compiled
_NO_COMPILE = "no_compile"  # needed hardware before compiling anything
_FAILED = "failed"  # failed for a reason hardware does not explain
_OPTED_OUT = "opted_out"  # marked compile_on_gpu
_SKIPPED = "skipped"  # skipped itself, or its fixture did

# Outcomes that leave a test nothing to replay, so it compiles on the GPU.
_SUSPENDED_OUTCOMES = frozenset({_NO_COMPILE, _FAILED, _OPTED_OUT, _SKIPPED})

# A virtual device refuses everything but compiling, and its refusals arrive as
# plain errors carrying one of these. Torch reports the same absence its own
# way, since a record action runs with no accelerator visible.
_NEEDS_HARDWARE_MESSAGES = (
    "VirtualDevice does not",
    "No CUDA GPUs are available",
    "Torch not compiled with CUDA enabled",
    "Found no NVIDIA driver",
)

_MARKER_HINT = (
    "If this test cannot be recorded -- it stages device buffers before its "
    "first load, or executes one graph before building the next -- create its "
    "device buffers after the load that needs them, or mark it "
    "@pytest.mark.compile_on_gpu('reason') to compile it on the GPU."
)


class _TestRecord(TypedDict):
    """How one test ended under virtual devices."""

    outcome: str
    exported: list[str]
    reason: str
    phase: str


class _FixtureRecord(TypedDict):
    """What one fixture setup exported under virtual devices."""

    exported: list[str]


class _Record(TypedDict):
    """A record run's account of itself, as replay reads it."""

    tests: dict[str, _TestRecord]
    fixtures: dict[str, _FixtureRecord]


class _Units:
    """Names each artifact after the unit of work that compiled it.

    A test and a fixture are the units: both runs execute the same code from
    the same revision, so a unit compiles the same graphs in the same order,
    and a compile's position within its unit is usable as identity. That is
    what lets a test loop over variants of one generically named graph without
    renaming anything, and what lets a record shard and a test shard hold
    different tests.

    Process-wide rather than thread-local: the graphs a unit compiles may be
    compiled on worker threads it spawned, and what identifies an artifact is
    which unit asked for it, not which thread got there.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._context = ""
        self._in_process = 0
        self._ordinals: dict[tuple[str, str], int] = {}
        # What each unit claimed: the artifacts record.json attributes to it,
        # and on replay the count of graphs that cost nothing to load.
        self.claimed: dict[str, set[str]] = defaultdict(set)
        self.count = 0

    @contextlib.contextmanager
    def unit(self, context: str) -> Iterator[None]:
        """Attributes the compiles in this block to ``context``.

        Args:
            context: The test or fixture setup being run.

        Yields:
            Nothing; the context is process state, not a value.
        """
        with self._lock:
            previous, self._context = self._context, context
        try:
            yield
        finally:
            with self._lock:
                self._context = previous

    @contextlib.contextmanager
    def compiling_in_process(self) -> Iterator[None]:
        """Declines the store for this block, so its graphs compile here.

        For the work the record run has nothing to say about: a unit that
        recorded no artifacts has to compile, and looking one up would only
        manufacture a miss.

        Yields:
            Nothing; the decision is process state, not a value.
        """
        with self._lock:
            self._in_process += 1
        try:
            yield
        finally:
            with self._lock:
                self._in_process -= 1

    def replay_key(self, graph: Graph) -> str | None:
        """Names the artifact ``graph`` claims, or declines it.

        Args:
            graph: The graph that would otherwise be compiled.

        Returns:
            The artifact's name, or :obj:`None` to compile ``graph`` here.
        """
        with self._lock:
            if self._in_process:
                return None
        return self.export_key(graph)

    def export_key(self, graph: Graph) -> str:
        """Names the artifact ``graph`` is compiled into.

        A record run declines nothing: every graph it compiles is one its
        replay expects to find, and ``compiling_in_process`` is only ever
        entered on replay.

        Args:
            graph: The graph being compiled.

        Returns:
            The artifact's name.
        """
        # The engine's own name for it -- graph name and signature digest --
        # which the unit and the position within it then disambiguate.
        base = _default_key(graph).removesuffix(_MEF_SUFFIX)
        with self._lock:
            context = self._context
            ordinal = self._ordinals.get((context, base), 0)
            self._ordinals[(context, base)] = ordinal + 1
            scope = hashlib.sha256(context.encode()).hexdigest()[:12]
            key = f"{base}-{scope}-{ordinal}{_MEF_SUFFIX}"
            self.claimed[context].add(key)
            self.count += 1
        return key


_UNITS = _Units()


def _replay_directories() -> list[Path]:
    """Resolves the artifact directories the test's runfiles hold.

    Returns:
        One directory per record shard.

    Raises:
        RuntimeError: If runfiles are unavailable or a directory is missing,
            either of which would otherwise degrade into compiling on the GPU.
    """
    resolver = runfiles.Create()
    if resolver is None:
        raise RuntimeError("replaying precompiled MEFs needs runfiles")

    directories = []
    for rlocation in os.environ["PRECOMPILE_MEFS_REPLAY_RLOCATIONS"].split():
        resolved = resolver.Rlocation(rlocation)
        if resolved is None or not Path(resolved).is_dir():
            raise RuntimeError(f"precompiled MEFs missing at {rlocation!r}")
        directories.append(Path(resolved))
    return directories


def _union_records(directories: list[Path]) -> _Record:
    """Unions the per-shard accounts of what each test and fixture exported.

    Args:
        directories: The artifact directories, each holding one record.

    Returns:
        The union, keyed as one shard's record is.

    Raises:
        FileNotFoundError: If a directory holds no record, so was not written
            by a record run.
    """
    tests: dict[str, _TestRecord] = {}
    fixtures: dict[str, _FixtureRecord] = {}
    for directory in directories:
        path = directory / _RECORD_NAME
        if not path.is_file():
            raise FileNotFoundError(
                f"{directory} has no {_RECORD_NAME}, so it was not written by "
                "a record run"
            )
        recorded = json.loads(path.read_text())
        # A test lands in exactly one record shard. A fixture scoped above the
        # test does not: every shard that reached it recorded its own exports.
        tests.update(recorded["tests"])
        for context, entry in recorded["fixtures"].items():
            seen = fixtures.setdefault(context, {"exported": []})
            seen["exported"] = sorted(
                set(seen["exported"]) | set(entry["exported"])
            )
    return {"tests": tests, "fixtures": fixtures}


_MODE = os.environ.get("PRECOMPILE_MEFS_MODE")
_STORE: MefStore | None = None
_RECORD: _Record = {"tests": {}, "fixtures": {}}

if _MODE == "record":
    # Before anything creates a device: the knobs latch at the first one, and
    # `max._interpreter_ops` freezes its device set when it is imported.
    _API, _, _ARCH = os.environ["PRECOMPILE_MEFS_ACCELERATOR"].partition(":")
    if _CPU_TARGET := os.environ["PRECOMPILE_MEFS_CPU_TARGET"]:
        set_virtual_cpu_target(_CPU_TARGET)
    set_virtual_device_api(_API)
    set_virtual_device_target_arch(_ARCH)
    set_virtual_device_count(
        int(os.environ.get("PRECOMPILE_MEFS_DEVICE_COUNT", "1"))
    )
    _STORE = MefStore.for_export(
        os.environ["PRECOMPILE_MEFS_RECORD_DIR"], key=_UNITS.export_key
    )
    InferenceSession.default_mef_store = _STORE
elif _MODE == "replay":
    _DIRECTORIES = _replay_directories()
    # Both runs execute the same code at the same revision, so a graph that
    # differs from the one recorded in its place means they diverged; say so
    # rather than initialize an artifact of some other computation.
    _STORE = MefStore.for_import(
        _DIRECTORIES, key=_UNITS.replay_key, verify_body=True
    )
    InferenceSession.default_mef_store = _STORE
    _RECORD = _union_records(_DIRECTORIES)

# What the record run has seen so far, which is what it writes at the end.
_TESTS: dict[str, _TestRecord] = {}
_FIXTURES: dict[str, _FixtureRecord] = {}
_FIXTURE_CONTEXTS: dict[str, set[str]] = defaultdict(set)

# What the replay run has seen so far, for its summary.
_ON_GPU: dict[str, str] = {}
_FIXTURES_ON_GPU: set[str] = set()
_DIVERGED: list[str] = []


def _marker_reason(item: pytest.Item) -> str:
    """Returns why ``item`` is marked to compile on the GPU.

    Args:
        item: The marked test.

    Returns:
        The reason the marker names.

    Raises:
        UsageError: If the marker names no reason, or not a string. An opt-out
            that outlives its cause is the failure mode here, so the marker
            carries what would have to change to drop it.
    """
    marker = item.get_closest_marker(_MARKER)
    assert marker is not None
    reason = marker.args[0] if marker.args else None
    if not isinstance(reason, str) or not reason:
        raise pytest.UsageError(
            f"{item.nodeid}: @pytest.mark.{_MARKER} needs a reason, as a "
            "string, saying what stops this test from being recorded"
        )
    return reason


def _fixture_context(argname: str, request: pytest.FixtureRequest) -> str:
    """Names the fixture setup whose compiles follow.

    The node the fixture is scoped to is part of the name, so a session-scoped
    fixture keeps one identity while a function-scoped one gets an identity per
    test -- exactly as the compiles they perform do.

    Args:
        argname: The fixture's name.
        request: The request it is being set up for.

    Returns:
        The context to attribute its compiles to.
    """
    index = getattr(request, "param_index", 0)
    return f"fixture:{argname}[{index}]@{request.node.nodeid}"


def _fixture_contexts_of(item: pytest.Item) -> Iterator[str]:
    """Yields the fixture contexts whose exports belong to ``item``.

    Args:
        item: The test to attribute fixture exports to.

    Yields:
        Each context of a fixture this test used, at whatever scope it was
        set up.
    """
    for argname in getattr(item, "fixturenames", ()):
        for context in _FIXTURE_CONTEXTS.get(argname, ()):
            _, _, scope = context.partition("@")
            if item.nodeid.startswith(scope):
                yield context


def _exports_of(item: pytest.Item) -> frozenset[str]:
    """Returns what ``item`` has exported, its fixtures included.

    Args:
        item: The test to attribute exports to.

    Returns:
        The keys of the artifacts attributable to it.
    """
    exported = set(_UNITS.claimed.get(item.nodeid, ()))
    for context in _fixture_contexts_of(item):
        exported |= _UNITS.claimed.get(context, set())
    return frozenset(exported)


def _needs_hardware(error: BaseException | None) -> bool:
    """Reports whether ``error`` is a virtual device refusing to be real.

    Args:
        error: The exception a recorded test ended with.

    Returns:
        Whether the test stopped because it needed the accelerator, rather than
        because something is wrong with it.
    """
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, CompileOnlyExecutionError):
            return True
        if any(needle in str(error) for needle in _NEEDS_HARDWARE_MESSAGES):
            return True
        error = error.__cause__ or error.__context__
    return False


def _note(
    nodeid: str, outcome: str, exported: frozenset[str], reason: str, phase: str
) -> None:
    """Notes how a test ended, unless something earlier already did.

    A failure in setup decides the test; the call phase never runs to say
    otherwise.

    Args:
        nodeid: The test.
        outcome: What became of it.
        exported: The artifacts attributable to it.
        reason: Why, for the summary and for replay to quote.
        phase: The phase that decided it.
    """
    if nodeid in _TESTS:
        return
    _TESTS[nodeid] = {
        "outcome": outcome,
        "exported": sorted(exported),
        "reason": reason,
        "phase": phase,
    }


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{_MARKER}(reason): compile this test's graphs on the GPU instead of "
        "recording and replaying them",
    )


# After pytest-shard has taken its slice: what this deselects must not change
# which tests the other shards run.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not _MODE:
        return

    # Validated in both modes, so a marker that says nothing is a usage error
    # wherever it is introduced rather than only in the record action.
    marked = [item for item in items if item.get_closest_marker(_MARKER)]
    reasons = {item.nodeid: _marker_reason(item) for item in marked}
    if _MODE != "record" or not marked:
        return

    for item in marked:
        _note(
            item.nodeid, _OPTED_OUT, frozenset(), reasons[item.nodeid], "setup"
        )
    items[:] = [item for item in items if item not in marked]
    config.hook.pytest_deselected(items=marked)


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(
    fixturedef: pytest.FixtureDef[object], request: pytest.FixtureRequest
) -> Generator[None, object, object]:
    if not _MODE:
        return (yield)

    context = _fixture_context(fixturedef.argname, request)
    if _MODE == "record":
        try:
            with _UNITS.unit(context):
                return (yield)
        finally:
            # Even when the fixture failed: what it compiled before reaching
            # hardware is what its tests replay.
            _FIXTURE_CONTEXTS[fixturedef.argname].add(context)
            _FIXTURES[context] = {
                "exported": sorted(_UNITS.claimed.get(context, ()))
            }

    # A fixture that recorded nothing has nothing to look up, and one that
    # reached hardware before compiling has to compile where it can.
    recorded = _RECORD["fixtures"].get(context)
    if recorded is not None and recorded["exported"]:
        with _UNITS.unit(context):
            return (yield)
    # Only a fixture the record run never reached is worth reporting: it
    # compiles here and the record cannot say what. One the record did set up
    # that exported nothing compiles nothing here either -- same code, same
    # order -- so declining changes nothing.
    if recorded is None:
        _FIXTURES_ON_GPU.add(context)
    with _UNITS.unit(context), _UNITS.compiling_in_process():
        return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(
    item: pytest.Item, nextitem: pytest.Item | None
) -> Generator[None, object, object]:
    if not _MODE:
        return (yield)

    with _UNITS.unit(item.nodeid):
        return (yield)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    if _MODE != "replay" or item.nodeid in _RECORD["tests"]:
        return
    # Neither strictness nor suspension is right for a test nothing recorded,
    # and quietly compiling it would hide that the two runs disagree about what
    # the test suite contains.
    pytest.fail(
        f"the record run never reached {item.nodeid}, so the two runs disagree "
        "about which tests exist. Parametrize ids have to be stable between "
        "runs; ids derived from an object's repr are not.",
        pytrace=False,
    )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, None, None]:
    if _MODE != "replay":
        return (yield)

    recorded = _RECORD["tests"][item.nodeid]
    if recorded["outcome"] not in _SUSPENDED_OUTCOMES:
        return (yield)

    _ON_GPU[item.nodeid] = recorded["reason"] or recorded["outcome"]
    with _UNITS.compiling_in_process():
        return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    if not _MODE or call.when not in ("setup", "call"):
        return report

    if _MODE == "replay":
        _explain_divergence(item, call, report)
        return report

    exported = _exports_of(item)
    if call.excinfo is None:
        if call.when == "call":
            _note(item.nodeid, _PASSED, exported, "", call.when)
        return report

    error = call.excinfo.value
    if isinstance(error, pytest.skip.Exception):
        _note(item.nodeid, _SKIPPED, exported, str(error), call.when)
        return report
    # An xfail is the test's own verdict on itself, and pytest is stricter
    # about it than this is; leave its report alone.
    if item.get_closest_marker("xfail"):
        return report
    if not _needs_hardware(error):
        _note(
            item.nodeid,
            _FAILED,
            exported,
            f"{type(error).__name__}: {str(error)[:200]}",
            call.when,
        )
        return report

    # Needing hardware is where a recorded test is expected to stop, so it
    # reads as a skip rather than as a failure of a run with no verdict to give.
    if exported:
        outcome = _RECORDED
        graphs = "graph" if len(exported) == 1 else "graphs"
        reason = f"needed hardware after compiling {len(exported)} {graphs}"
    else:
        outcome = _NO_COMPILE
        reason = (
            "needed hardware before compiling anything, so there is nothing "
            f"to replay and this test will compile on the GPU. {_MARKER_HINT}"
        )
    _note(item.nodeid, outcome, exported, reason, call.when)
    report.outcome = "skipped"
    report.longrepr = (item.location[0], item.location[1] or 0, reason)
    return report


def _explain_divergence(
    item: pytest.Item, call: pytest.CallInfo[None], report: pytest.TestReport
) -> None:
    """Attaches what the record run said about a test that failed to replay.

    Args:
        item: The test.
        call: The phase that raised.
        report: Its report, which the explanation is attached to.
    """
    if call.excinfo is None:
        return
    if not isinstance(
        call.excinfo.value, (MissingArtifactError, ArtifactBodyMismatch)
    ):
        return

    _DIVERGED.append(item.nodeid)
    recorded = _RECORD["tests"].get(
        item.nodeid, {"outcome": "nothing", "exported": [], "reason": ""}
    )
    said = (
        f"the record run left this test {recorded['outcome']} after exporting "
        f"{len(recorded['exported'])} graphs"
    )
    if recorded["reason"]:
        said += f": {recorded['reason']}"
    report.sections.append(("MEF replay", f"{said}\n{_MARKER_HINT}"))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _MODE != "record":
        return

    assert _STORE is not None
    # An empty manifest is still a manifest: replay reads every directory, so a
    # shard that recorded nothing has to say so rather than look unwritten.
    _STORE.write_manifest()
    directory = Path(os.environ["PRECOMPILE_MEFS_RECORD_DIR"])
    (directory / _RECORD_NAME).write_text(
        json.dumps(
            {
                "version": _RECORD_VERSION,
                "accelerator": os.environ["PRECOMPILE_MEFS_ACCELERATOR"],
                "cpu_target": os.environ["PRECOMPILE_MEFS_CPU_TARGET"],
                "device_count": int(
                    os.environ.get("PRECOMPILE_MEFS_DEVICE_COUNT", "1")
                ),
                "tests": _TESTS,
                "fixtures": _FIXTURES,
            },
            indent=2,
            sort_keys=True,
        )
    )

    # The verdicts belong to the run on the GPU. This one has produced its
    # artifacts and its account of them, which is all it was asked for.
    if exitstatus in (
        pytest.ExitCode.TESTS_FAILED,
        pytest.ExitCode.NO_TESTS_COLLECTED,
    ):
        session.exitstatus = pytest.ExitCode.OK


def pytest_terminal_summary(terminalreporter: TerminalReporter) -> None:
    if not _MODE:
        return

    write = terminalreporter.write_line
    if _MODE == "record":
        counts: dict[str, int] = defaultdict(int)
        for entry in _TESTS.values():
            counts[entry["outcome"]] += 1
        summary = ", ".join(
            f"{counts[outcome]} {outcome}"
            for outcome in (
                _RECORDED,
                _PASSED,
                _NO_COMPILE,
                _OPTED_OUT,
                _SKIPPED,
                _FAILED,
            )
        )
        write(
            f"precompile-mefs record: {summary}; {_UNITS.count} artifacts "
            "exported"
        )
        for nodeid, entry in sorted(_TESTS.items()):
            if entry["outcome"] in (_NO_COMPILE, _OPTED_OUT, _FAILED):
                write(f"  {entry['outcome']} {nodeid}: {entry['reason']}")
        return

    # Fixtures are counted separately because that is where the surprise
    # lives: a test can replay every graph its body compiles and still spend
    # minutes in a fixture that compiles one the record run never reached.
    write(
        f"precompile-mefs replay: {_UNITS.count} graphs answered from "
        f"artifacts, {len(_ON_GPU)} tests and {len(_FIXTURES_ON_GPU)} fixtures "
        f"compiled on the GPU, {len(_DIVERGED)} diverged from the record"
    )
    for nodeid, reason in sorted(_ON_GPU.items()):
        write(f"  on GPU {nodeid}: {reason}")
    for context in sorted(_FIXTURES_ON_GPU):
        write(f"  on GPU {context}: the record run never set this fixture up")
