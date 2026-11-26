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
"""Tests the claims in mojo-from-python.mdx from a real Python entry point.

Compiler verified: 2026-11-24, Mojo 1.3.0.dev2026112105
Last updated: 2026-11-24

Python reaches the page's modules here the way a reader's program does,
through `mojo.importer` and CPython's own call machinery. `tests.mojo` covers
the claims that need Mojo-side machinery.

Page claims, and where each is verified

Move a hot loop into Mojo
  1. Python and Mojo `count_primes()` both count 148,933 primes below
     2,000,000. -> test_both_versions_count_148933
  2. The Mojo version matches the Python version line for line, so both give
     the same count for every limit. -> test_same_algorithm_same_answer
  3. The Mojo version finishes in a fraction of the time.
     -> test_mojo_is_faster
  4. The wrapper converts its argument with `Int(py=)`, and a failed
     conversion raises a Python exception. -> test_failed_conversion_raises
  5. The builder registers `py_count_primes()` under the Python name
     `count_primes`. -> test_registered_name
  6. `def_function()` accepts plain functions but not closures.
     -> test_closures_dont_bind

How `mojo.importer` works
  7. After `import mojo.importer`, `import fast_primes` finds the `.mojo`
     file on `sys.path`, compiles it, and loads it.
     -> test_importer_caches_next_to_source
  8. Compiled modules land in `__mojocache__` next to the source, as
     `<module>.hash-<hash>.so`. -> test_importer_caches_next_to_source
  9. Later runs reuse the compiled module until the source changes.
     -> test_importer_caches_next_to_source
 10. A changed source recompiles. -> test_importer_caches_next_to_source
 11. Editing any `.mojo` file in the directory or a subdirectory
     invalidates the cache. -> test_importer_caches_next_to_source
 12. Deleting `__mojocache__` is safe; the next import rebuilds.
     -> test_importer_caches_next_to_source
 13. A module that doesn't compile makes `import` raise `ImportError`, with
     the compiler's messages in the traceback.
     -> test_staticmethod_required_inside_struct

Bridge Python to your Mojo types
 14. `def_init_defaultable()` lets Python call `RunningStats()` with no
     arguments. -> test_create_with_no_arguments
 15. `def_py_method()` passes every positional argument as one tuple, so
     `add()` takes one sample or many. -> test_add_takes_one_or_many
 16. The docstrings reach Python. -> test_docstrings
 17. `merge()` accepts another `RunningStats`. -> test_report_values
 18. Passing a list to `merge()` raises an error that names the expected
     type. -> test_merge_rejects_other_types
 19. `from_values()` is called on the type and returns an object of the
     registered type. -> test_from_values_returns_registered_type
 20. `report.py` prints the mean and sample standard deviation of all six
     samples. -> test_report_values
 21. Python prints the object through the struct's `Writable` conformance.
     -> test_print_uses_writable (the exact form is pinned in `tests.mojo`)

Sharp edges
 22. Python sees methods, not fields; the object has no `__dict__`, so
     Python can't add attributes. -> test_methods_not_fields
 23. Interface functions declared inside a struct need `@staticmethod`, or
     the module doesn't compile and `import` raises `ImportError`.
     -> test_staticmethod_required_inside_struct
 24. Keyword arguments arrive through `var **kwargs` and are read by name.
     -> test_keyword_arguments
 25. Mojo keyword-only and variadic parameters don't bind with
     `def_function()`. -> test_keyword_only_and_variadic_dont_bind
 26. `def_py_function()` receives the arguments as one tuple.
     -> test_def_py_function_takes_a_tuple
 27. A wrong argument count raises a plain `Exception` whose message starts
     with `TypeError:`, so `except TypeError` doesn't catch it.
     -> test_wrong_arity_is_plain_exception
 28. `raise_python_exception()` raises a specific Python exception type.
     -> test_raise_python_exception

Build and ship
 29. `mojo build --emit shared-lib` produces a `.so` that Python imports
     without `mojo.importer`. -> test_prebuilt_module_imports_without_importer

Worth knowing
 30. A missing keyword raises an `Exception` that names `DictKeyError` but
     not the key; checking first lets you report it.
     -> test_missing_keyword_doesnt_name_the_key

Not tested, and why
  - The timing numbers in the output block. They're labeled as an example
    and vary by machine; claim 3 checks only the direction.
  - "Python holds the GIL while your Mojo function runs." A test would need
    a timing race between threads, which isn't stable in CI.
  - "The importer runs `mojo build` without extra options." Read from
    `_compile_mojo_to_so()` in `Mojo/python/mojo/importer.py`; there is no
    runtime observable.
  - "RunningStats never holds the samples themselves." Structural, visible
    in the struct's three fields.
  - NumPy borrowing and compile-time parameters. The page states them and
    links elsewhere, with no example of its own.
  - `add_type()` requiring `Writable`. Compile-time; every type the page
    registers conforms.
"""

import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mojo.importer  # noqa: F401

# isort: split

import edges_module  # type: ignore[import-not-found]
import fast_primes  # type: ignore[import-not-found]
import primes
from fast_stats import RunningStats  # type: ignore[import-not-found]
from mojo.run import subprocess_run_mojo

HERE = Path(__file__).resolve().parent
SAMPLES = [12.5, 9.8, 14.1, 11.0, 30.2, 10.7]


def run_python(code: str, *paths: Path) -> subprocess.CompletedProcess[str]:
    """Runs `code` in a fresh interpreter, so each import starts cold."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(p) for p in paths] + [p for p in sys.path if p]
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


def import_fails(source: Path) -> None:
    """Imports `source` from an empty directory and expects a compile error."""
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(source, d)
        result = run_python(
            f"import mojo.importer\nimport {source.stem}", Path(d)
        )
    assert result.returncode != 0, f"{source.name} imported"
    assert "ImportError" in result.stderr, result.stderr
    # The compiler's own message reaches the traceback.
    assert f"{source.name}:" in result.stderr, result.stderr
    assert "error:" in result.stderr, result.stderr


def import_succeeds(source: Path) -> None:
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(source, d)
        result = run_python(
            f"import mojo.importer\nimport {source.stem}", Path(d)
        )
    assert result.returncode == 0, result.stderr


# --- Move a hot loop into Mojo ---


def test_both_versions_count_148933() -> None:
    assert primes.count_primes(2_000_000) == 148_933
    assert fast_primes.count_primes(2_000_000) == 148_933


def test_same_algorithm_same_answer() -> None:
    # Covers the `limit > 2` special case for 2 and the odd-only loop.
    for limit in range(500):
        assert fast_primes.count_primes(limit) == primes.count_primes(limit), (
            limit
        )


def test_mojo_is_faster() -> None:
    start = time.perf_counter()
    primes.count_primes(500_000)
    python_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    fast_primes.count_primes(500_000)
    mojo_elapsed = time.perf_counter() - start

    assert mojo_elapsed < python_elapsed, (mojo_elapsed, python_elapsed)


def test_failed_conversion_raises() -> None:
    try:
        fast_primes.count_primes("two million")
    except Exception as e:
        assert "invalid literal for int()" in str(e), str(e)
    else:
        raise AssertionError("expected the Int(py=) conversion to fail")


def test_registered_name() -> None:
    assert fast_primes.__name__ == "fast_primes"
    assert hasattr(fast_primes, "count_primes")
    assert not hasattr(fast_primes, "py_count_primes")


def test_closures_dont_bind() -> None:
    import_fails(HERE / "compile_checks/capturing_closure.mojo")
    # A nested function with an empty capture list is a plain function.
    import_succeeds(HERE / "compile_checks/nested_plain_function.mojo")


# --- How `mojo.importer` works ---


def test_importer_caches_next_to_source() -> None:
    probe = "import mojo.importer, fast_primes\nprint(fast_primes.count_primes(100))"

    def cached(d: Path) -> list[Path]:
        return sorted((d / "__mojocache__").glob("*"))

    def run(d: Path) -> None:
        result = run_python(probe, d)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "25", result.stdout

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        source = d / "fast_primes.mojo"
        shutil.copy(HERE / "fast_primes.mojo", source)

        run(d)
        (first,) = cached(d)
        assert first.name.startswith("fast_primes.hash-"), first.name
        assert first.suffix == ".so", first.name
        built_at = first.stat().st_mtime_ns

        # Unchanged source: the import reuses the cached module.
        run(d)
        assert cached(d) == [first]
        assert first.stat().st_mtime_ns == built_at

        # Changed source: a new hash replaces the old module.
        source.write_text(source.read_text() + "\n# edited\n")
        run(d)
        (second,) = cached(d)
        assert second != first, second.name

        # A change elsewhere in the directory tree also invalidates it.
        (d / "util").mkdir()
        (d / "util" / "helpers.mojo").write_text("comptime SCALE = 2\n")
        run(d)
        (third,) = cached(d)
        assert third not in (first, second), third.name

        # Deleting the cache is safe; the next import rebuilds the same hash.
        shutil.rmtree(d / "__mojocache__")
        run(d)
        assert cached(d) == [third]


# --- Bridge Python to your Mojo types ---


def test_create_with_no_arguments() -> None:
    stats = RunningStats()
    assert stats.mean() == 0.0
    assert stats.stddev() == 0.0


def test_add_takes_one_or_many() -> None:
    one_at_a_time = RunningStats()
    for value in SAMPLES:
        one_at_a_time.add(value)

    all_at_once = RunningStats()
    all_at_once.add(*SAMPLES)

    assert one_at_a_time.mean() == all_at_once.mean()
    assert one_at_a_time.stddev() == all_at_once.stddev()


def test_docstrings() -> None:
    assert RunningStats.add.__doc__ == "Add one or more samples."
    assert RunningStats.from_values.__doc__ == "Build stats from an iterable."


def test_merge_rejects_other_types() -> None:
    try:
        RunningStats().merge([30.2, 10.7])
    except Exception as e:
        assert "RunningStats" in str(e), str(e)
        assert "list" in str(e), str(e)
    else:
        raise AssertionError("expected merge() to reject a list")


def test_from_values_returns_registered_type() -> None:
    batch = RunningStats.from_values([30.2, 10.7])
    assert type(batch) is RunningStats
    assert batch.mean() == statistics.mean([30.2, 10.7])


def test_report_values() -> None:
    # The page's `report.py`, with asserts in place of the first print.
    latency = RunningStats()
    latency.add(12.5, 9.8, 14.1)
    latency.add(11.0)

    batch = RunningStats.from_values([30.2, 10.7])
    latency.merge(batch)

    assert abs(latency.mean() - statistics.mean(SAMPLES)) < 1e-9
    assert abs(latency.stddev() - statistics.stdev(SAMPLES)) < 1e-9
    assert f"{latency.mean():.2f}" == "14.72"
    assert f"{latency.stddev():.2f}" == "7.73"


def test_print_uses_writable() -> None:
    stats = RunningStats.from_values([1.0, 3.0])
    assert str(stats).startswith("RunningStats("), str(stats)
    assert "count=Int(2)" in str(stats), str(stats)


# --- Sharp edges ---


def test_methods_not_fields() -> None:
    stats = RunningStats()
    assert not hasattr(stats, "count")
    assert not hasattr(stats, "__dict__")
    try:
        stats.label = "latency"
    except AttributeError:
        pass
    else:
        raise AssertionError("expected Python to refuse a new attribute")


def test_staticmethod_required_inside_struct() -> None:
    import_fails(HERE / "compile_checks/method_without_staticmethod.mojo")
    # Adding `@staticmethod` is the whole fix.
    import_succeeds(HERE / "compile_checks/method_with_staticmethod.mojo")


def test_keyword_arguments() -> None:
    assert edges_module.scaled(3, scale=4) == 12


def test_keyword_only_and_variadic_dont_bind() -> None:
    for name in ("keyword_only", "variadic"):
        import_fails(HERE / f"compile_checks/{name}.mojo")


def test_def_py_function_takes_a_tuple() -> None:
    assert edges_module.total() == 0
    assert edges_module.total(1, 2, 3) == 6


def test_wrong_arity_is_plain_exception() -> None:
    try:
        try:
            fast_primes.count_primes()
        except TypeError:
            raise AssertionError("except TypeError caught it") from None
    except Exception as e:
        assert type(e) is Exception, type(e)
        assert str(e).startswith("TypeError:"), str(e)
    else:
        raise AssertionError("expected a missing argument to fail")


def test_raise_python_exception() -> None:
    try:
        edges_module.raise_value_error()
    except ValueError as e:
        assert str(e) == "raised as ValueError", str(e)
    else:
        raise AssertionError("expected a ValueError")


# --- Build and ship ---


def test_prebuilt_module_imports_without_importer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        subprocess_run_mojo(
            [
                "build",
                str(HERE / "fast_primes.mojo"),
                "--emit",
                "shared-lib",
                "-o",
                str(d / "fast_primes.so"),
            ],
            capture_output=True,
            check=True,
        )
        result = run_python(
            "import sys, fast_primes\n"
            "assert 'mojo.importer' not in sys.modules\n"
            "print(fast_primes.__file__.endswith('.so'),"
            " fast_primes.count_primes(100))",
            d,
        )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["True", "25"], result.stdout


# --- Worth knowing ---


def test_missing_keyword_doesnt_name_the_key() -> None:
    try:
        edges_module.scaled(3)
    except Exception as e:
        assert type(e) is Exception, type(e)
        assert "DictKeyError" in str(e), str(e)
        assert "scale" not in str(e), str(e)
    else:
        raise AssertionError("expected the missing keyword to fail")

    try:
        edges_module.scaled_checked(3)
    except Exception as e:
        assert "'scale'" in str(e), str(e)
    else:
        raise AssertionError("expected the checked lookup to fail")


def main() -> None:
    test_both_versions_count_148933()
    test_same_algorithm_same_answer()
    test_mojo_is_faster()
    test_failed_conversion_raises()
    test_registered_name()
    test_closures_dont_bind()
    test_importer_caches_next_to_source()
    test_create_with_no_arguments()
    test_add_takes_one_or_many()
    test_docstrings()
    test_merge_rejects_other_types()
    test_from_values_returns_registered_type()
    test_report_values()
    test_print_uses_writable()
    test_methods_not_fields()
    test_staticmethod_required_inside_struct()
    test_keyword_arguments()
    test_keyword_only_and_variadic_dont_bind()
    test_def_py_function_takes_a_tuple()
    test_wrong_arity_is_plain_exception()
    test_raise_python_exception()
    test_prebuilt_module_imports_without_importer()
    test_missing_keyword_doesnt_name_the_key()
    print("all checks passed")


if __name__ == "__main__":
    main()
