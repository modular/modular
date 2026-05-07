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
"""Integration tests for the three debug configuration mechanisms.

These tests verify that each debug option can be enabled via:
1. Environment variable: ``MODULAR_DEBUG=nan-check,device-sync-mode``
2. Config file: ``[max-debug]`` section in ``modular.cfg``
3. Python API: ``InferenceSession.debug.nan_check = True``

And that the priority ordering (Python API > env var > config file) is
respected.

All env-var and config-file tests run in subprocesses so the parent
process state is not mutated.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def _run_script(
    script: str,
    env_overrides: dict[str, str] | None = None,
    env_removals: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in an isolated subprocess.

    Args:
        script: Python source code to execute.
        env_overrides: Extra env vars to set.
        env_removals: Env vars to unset before running.
    """
    env = os.environ.copy()
    # Remove any debug env vars that could leak from the parent.
    # Also remove MODULAR_HOME / MODULAR_DERIVED_PATH so that
    # TEST_TMPDIR is used for config file discovery in tests.
    for key in [
        "MODULAR_DEBUG",
        "MODULAR_DEVICE_CONTEXT_SYNC_MODE",
        "MODULAR_MAX_DEBUG_ASSERT_LEVEL",
        "MODULAR_HOME",
        "MODULAR_DERIVED_PATH",
    ]:
        env.pop(key, None)
    if env_removals:
        for key in env_removals:
            env.pop(key, None)
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def _assert_pass(result: subprocess.CompletedProcess[str]) -> None:
    """Assert a subprocess printed PASS and exited cleanly."""
    assert result.returncode == 0, (
        f"Script failed (rc={result.returncode}).\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "PASS" in result.stdout, (
        f"Expected PASS in stdout.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# 1. MODULAR_DEBUG env var mechanism
# ---------------------------------------------------------------------------


class TestEnvVarMechanism:
    """Debug features enabled via MODULAR_DEBUG env var."""

    def test_single_boolean_flag(self) -> None:
        result = _run_script(
            """\
            from max.engine import InferenceSession
            assert InferenceSession.debug.nan_check is True
            print("PASS")
            """,
            env_overrides={"MODULAR_DEBUG": "nan-check"},
        )
        _assert_pass(result)

    def test_multiple_comma_separated_flags(self) -> None:
        result = _run_script(
            """\
            from max.engine import InferenceSession
            assert InferenceSession.debug.nan_check is True
            assert InferenceSession.debug.device_sync_mode is True
            assert InferenceSession.debug.stack_trace_on_error is True
            print("PASS")
            """,
            env_overrides={
                "MODULAR_DEBUG": "nan-check,device-sync-mode,stack-trace-on-error"
            },
        )
        _assert_pass(result)

    def test_key_value_flag(self) -> None:
        result = _run_script(
            """\
            from max.engine import InferenceSession
            assert InferenceSession.debug.assert_level == "all"
            print("PASS")
            """,
            env_overrides={"MODULAR_DEBUG": "assert-level=all"},
        )
        _assert_pass(result)

    def test_mixed_boolean_and_key_value(self) -> None:
        result = _run_script(
            """\
            from max.engine import InferenceSession
            assert InferenceSession.debug.nan_check is True
            assert InferenceSession.debug.assert_level == "safe"
            print("PASS")
            """,
            env_overrides={"MODULAR_DEBUG": "nan-check,assert-level=safe"},
        )
        _assert_pass(result)

    def test_whitespace_tolerance(self) -> None:
        result = _run_script(
            """\
            from max.engine import InferenceSession
            assert InferenceSession.debug.nan_check is True
            assert InferenceSession.debug.device_sync_mode is True
            print("PASS")
            """,
            env_overrides={"MODULAR_DEBUG": " nan-check , device-sync-mode "},
        )
        _assert_pass(result)

    def test_sensible_meta_mode(self) -> None:
        result = _run_script(
            """\
            from max.engine import InferenceSession
            from max.graph import Graph
            assert InferenceSession.debug.nan_check is True
            assert InferenceSession.debug.device_sync_mode is True
            assert InferenceSession.debug.stack_trace_on_error is True
            assert InferenceSession.debug.stack_trace_on_crash is True
            assert InferenceSession.debug.assert_level == "all"
            assert Graph.debug.source_tracebacks is True
            # uninitialized_read_check is NOT part of the sensible set
            assert InferenceSession.debug.uninitialized_read_check is False
            print("PASS")
            """,
            env_overrides={"MODULAR_DEBUG": "sensible"},
        )
        _assert_pass(result)

    def test_empty_env_var_is_noop(self) -> None:
        result = _run_script(
            """\
            from max.engine import InferenceSession
            assert InferenceSession.debug.nan_check is False
            assert InferenceSession.debug.device_sync_mode is False
            print("PASS")
            """,
            env_overrides={"MODULAR_DEBUG": ""},
        )
        _assert_pass(result)

    def test_unset_env_var_is_noop(self) -> None:
        result = _run_script(
            """\
            from max.engine import InferenceSession
            assert InferenceSession.debug.nan_check is False
            print("PASS")
            """,
            env_removals=["MODULAR_DEBUG"],
        )
        _assert_pass(result)

    def test_all_value_options(self) -> None:
        result = _run_script(
            """\
            from max.engine import InferenceSession, PrintStyle
            assert InferenceSession.debug.assert_level == "all"
            assert InferenceSession.debug.op_log_level == "DEBUG"
            assert InferenceSession.debug.print_style == PrintStyle.COMPACT
            assert InferenceSession.debug.ir_output_dir == "/tmp/ir"
            print("PASS")
            """,
            env_overrides={
                "MODULAR_DEBUG": (
                    "assert-level=all,op-log-level=DEBUG,"
                    "print-style=compact,ir-output-dir=/tmp/ir"
                )
            },
        )
        _assert_pass(result)


# ---------------------------------------------------------------------------
# 2. Config file mechanism (modular.cfg [max-debug] section)
# ---------------------------------------------------------------------------


class TestConfigFileMechanism:
    """Debug features enabled via [max-debug] section in modular.cfg."""

    @staticmethod
    def _write_config(tmp_path: Path, content: str) -> dict[str, str]:
        """Write a modular.cfg and return env overrides to find it.

        The Config system checks TEST_TMPDIR for a .modular/ subdirectory
        containing modular.cfg. We create that structure in tmp_path.
        """
        config_dir = tmp_path / ".modular"
        config_dir.mkdir()
        config_file = config_dir / "modular.cfg"
        config_file.write_text(content)
        return {"TEST_TMPDIR": str(tmp_path)}

    def test_boolean_option_from_config_file(self, tmp_path: Path) -> None:
        env = self._write_config(
            tmp_path,
            "[max-debug]\nnan-check = true\n",
        )
        result = _run_script(
            """\
            from max.engine import InferenceSession
            assert InferenceSession.debug.nan_check is True
            print("PASS")
            """,
            env_overrides=env,
        )
        _assert_pass(result)

    def test_multiple_options_from_config_file(self, tmp_path: Path) -> None:
        env = self._write_config(
            tmp_path,
            textwrap.dedent("""\
            [max-debug]
            nan-check = true
            device-sync-mode = true
            assert-level = safe
            """),
        )
        result = _run_script(
            """\
            from max.engine import InferenceSession
            assert InferenceSession.debug.nan_check is True
            assert InferenceSession.debug.device_sync_mode is True
            assert InferenceSession.debug.assert_level == "safe"
            print("PASS")
            """,
            env_overrides=env,
        )
        _assert_pass(result)

    def test_config_file_false_values_do_not_enable(
        self, tmp_path: Path
    ) -> None:
        env = self._write_config(
            tmp_path,
            "[max-debug]\nnan-check = false\n",
        )
        result = _run_script(
            """\
            from max.engine import InferenceSession
            assert InferenceSession.debug.nan_check is False
            print("PASS")
            """,
            env_overrides=env,
        )
        _assert_pass(result)

    def test_string_value_from_config_file(self, tmp_path: Path) -> None:
        env = self._write_config(
            tmp_path,
            "[max-debug]\nir-output-dir = /tmp/debug-ir\n",
        )
        result = _run_script(
            """\
            from max.engine import InferenceSession
            assert InferenceSession.debug.ir_output_dir == "/tmp/debug-ir"
            print("PASS")
            """,
            env_overrides=env,
        )
        _assert_pass(result)


# ---------------------------------------------------------------------------
# 3. Python API mechanism
# ---------------------------------------------------------------------------


class TestPythonAPIMechanism:
    """Debug features enabled via the Python API at runtime."""

    def test_boolean_property_roundtrip(self) -> None:
        result = _run_script(
            """\
            from max.engine import InferenceSession
            InferenceSession.debug.nan_check = True
            assert InferenceSession.debug.nan_check is True
            InferenceSession.debug.nan_check = False
            assert InferenceSession.debug.nan_check is False
            print("PASS")
            """,
        )
        _assert_pass(result)

    def test_value_property_roundtrip(self) -> None:
        result = _run_script(
            """\
            from max.engine import InferenceSession
            InferenceSession.debug.assert_level = "all"
            assert InferenceSession.debug.assert_level == "all"
            InferenceSession.debug.assert_level = ""
            assert InferenceSession.debug.assert_level == ""
            print("PASS")
            """,
        )
        _assert_pass(result)

    def test_meta_mode_via_api(self) -> None:
        result = _run_script(
            """\
            from max.engine import InferenceSession
            from max.graph import Graph
            InferenceSession.debug.sensible_mode = True
            assert InferenceSession.debug.nan_check is True
            assert InferenceSession.debug.device_sync_mode is True
            assert InferenceSession.debug.stack_trace_on_error is True
            assert InferenceSession.debug.stack_trace_on_crash is True
            assert InferenceSession.debug.assert_level == "all"
            assert Graph.debug.source_tracebacks is True
            print("PASS")
            """,
        )
        _assert_pass(result)

    def test_reset_clears_all(self) -> None:
        result = _run_script(
            """\
            from max.engine import InferenceSession
            from max.graph import Graph
            InferenceSession.debug.sensible_mode = True
            InferenceSession.debug.reset()
            assert InferenceSession.debug.nan_check is False
            assert InferenceSession.debug.device_sync_mode is False
            assert InferenceSession.debug.assert_level == ""
            assert Graph.debug.source_tracebacks is False
            print("PASS")
            """,
        )
        _assert_pass(result)


# ---------------------------------------------------------------------------
# 4. Priority: Python API > env var > config file
# ---------------------------------------------------------------------------


class TestConfigPriority:
    """When multiple mechanisms set the same option, the highest-priority wins."""

    def test_env_var_overrides_config_file(self, tmp_path: Path) -> None:
        cfg_env = TestConfigFileMechanism._write_config(
            tmp_path,
            "[max-debug]\nassert-level = safe\n",
        )
        cfg_env["MODULAR_DEBUG"] = "assert-level=all"
        result = _run_script(
            """\
            from max.engine import InferenceSession
            assert InferenceSession.debug.assert_level == "all"
            print("PASS")
            """,
            env_overrides=cfg_env,
        )
        _assert_pass(result)

    def test_python_api_overrides_env_var(self) -> None:
        result = _run_script(
            """\
            from max.engine import InferenceSession
            # Env var set nan_check=True, but Python API overrides to False
            assert InferenceSession.debug.nan_check is True
            InferenceSession.debug.nan_check = False
            assert InferenceSession.debug.nan_check is False
            print("PASS")
            """,
            env_overrides={"MODULAR_DEBUG": "nan-check"},
        )
        _assert_pass(result)

    def test_python_api_overrides_config_file(self, tmp_path: Path) -> None:
        cfg_env = TestConfigFileMechanism._write_config(
            tmp_path,
            "[max-debug]\nnan-check = true\n",
        )
        result = _run_script(
            """\
            from max.engine import InferenceSession
            InferenceSession.debug.nan_check = False
            assert InferenceSession.debug.nan_check is False
            print("PASS")
            """,
            env_overrides=cfg_env,
        )
        _assert_pass(result)


# ---------------------------------------------------------------------------
# 5. Reader-side migration: new config keys affect behavior at readers
#
# These spot-check that the migrated reader sites honor the new
# max-debug.* config keys in addition to their legacy env vars.
# ---------------------------------------------------------------------------


class TestReaderMigration:
    """The migrated reader sites see values set via the new Config keys."""

    def test_graph_source_tracebacks_via_modular_debug(self) -> None:
        # graph.py's _SOURCE_TRACEBACKS_ENABLED reads from
        # InferenceSession.debug.source_tracebacks, which goes through
        # Config and picks up MODULAR_DEBUG=source-tracebacks.
        result = _run_script(
            """\
            from max.graph import graph
            assert graph._SOURCE_TRACEBACKS_ENABLED is True
            print("PASS")
            """,
            env_overrides={"MODULAR_DEBUG": "source-tracebacks"},
        )
        _assert_pass(result)

    def test_ir_output_dir_via_modular_debug_dumps_ir(
        self, tmp_path: Path
    ) -> None:
        """Compiling a graph under `MODULAR_DEBUG=ir-output-dir=...`
        should write per-stage MLIR files into the configured directory.

        Regression test for GEX-3684: prior to the fix the option was
        plumbed into `InferenceSession.debug.ir_output_dir` but no
        compiler stage actually consulted it, so files were silently
        not written.
        """
        ir_dir = tmp_path / "ir-dump"
        ir_dir.mkdir()
        # Disable the MEF cache so the gc-pipeline actually runs (and
        # therefore writes IR); a previous test in the same shard could
        # otherwise have cached an MEF for the same graph and skipped
        # compilation entirely.
        result = _run_script(
            """\
            import numpy as np
            from max.driver import CPU
            from max.dtype import DType
            from max.engine import InferenceSession
            from max.graph import DeviceRef, Graph, TensorType

            graph = Graph(
                "tiny_add_modular_debug_dump",
                forward=lambda x, y: x + y,
                input_types=[
                    TensorType(
                        dtype=DType.float32,
                        shape=(4,),
                        device=DeviceRef.CPU(),
                    ),
                    TensorType(
                        dtype=DType.float32,
                        shape=(4,),
                        device=DeviceRef.CPU(),
                    ),
                ],
            )
            session = InferenceSession(devices=[CPU()])
            model = session.load(graph)
            model.execute(
                np.ones((4,), dtype=np.float32),
                np.ones((4,), dtype=np.float32),
            )
            print("PASS")
            """,
            env_overrides={
                "MODULAR_DEBUG": f"ir-output-dir={ir_dir}",
                "MODULAR_MAX_ENABLE_MODEL_IR_CACHE": "false",
            },
        )
        _assert_pass(result)
        files = sorted(p.name for p in ir_dir.iterdir())
        assert files, (
            "Expected per-stage IR dumps in ir-output-dir, but found nothing.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        # The pipeline writes one file per stage with extensions like
        # `.mo.mlir`, `.mo-pre-mogg.mlir`, `.mogg.mlir`, etc.  We just
        # require at least one MLIR file landed.
        assert any(name.endswith(".mlir") for name in files), (
            f"Expected at least one .mlir file in {ir_dir}, got {files}"
        )

    def test_ir_output_dir_via_legacy_temps_dir_still_works(
        self, tmp_path: Path
    ) -> None:
        """`MODULAR_MAX_TEMPS_DIR` continues to drive IR dumping after
        the migration to `max-debug.ir-output-dir`.
        """
        ir_dir = tmp_path / "legacy-ir-dump"
        ir_dir.mkdir()
        # Disable the MEF cache so the gc-pipeline actually runs (and
        # therefore writes IR); a previous test in the same shard could
        # otherwise have cached an MEF for the same graph and skipped
        # compilation entirely.
        result = _run_script(
            """\
            import numpy as np
            from max.driver import CPU
            from max.dtype import DType
            from max.engine import InferenceSession
            from max.graph import DeviceRef, Graph, TensorType

            graph = Graph(
                "tiny_add_legacy_dump",
                forward=lambda x, y: x + y,
                input_types=[
                    TensorType(
                        dtype=DType.float32,
                        shape=(4,),
                        device=DeviceRef.CPU(),
                    ),
                    TensorType(
                        dtype=DType.float32,
                        shape=(4,),
                        device=DeviceRef.CPU(),
                    ),
                ],
            )
            session = InferenceSession(devices=[CPU()])
            model = session.load(graph)
            model.execute(
                np.ones((4,), dtype=np.float32),
                np.ones((4,), dtype=np.float32),
            )
            print("PASS")
            """,
            env_overrides={
                "MODULAR_MAX_TEMPS_DIR": str(ir_dir),
                "MODULAR_MAX_ENABLE_MODEL_IR_CACHE": "false",
            },
        )
        _assert_pass(result)
        files = sorted(p.name for p in ir_dir.iterdir())
        assert files, (
            "Expected per-stage IR dumps under MODULAR_MAX_TEMPS_DIR.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert any(name.endswith(".mlir") for name in files), (
            f"Expected at least one .mlir file in {ir_dir}, got {files}"
        )
