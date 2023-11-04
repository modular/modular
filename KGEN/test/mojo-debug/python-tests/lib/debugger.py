# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Module that holds singleton instances for the lldb module and the debugger."""

import configparser
import subprocess
import importlib
import sys
import os
from typing import Any

lldb: Any = None
debugger: Any = None


def load_lldb() -> Any:
    """Loads the lldb python module and cache it."""
    global lldb
    if lldb is None:
        # Fortunately the path to the module can be gotten from LLDB itself.
        lldb_lib = subprocess.check_output(
            ["mojo", "debug", "-P"], text=True
        ).strip()
        sys.path.insert(0, lldb_lib)
        lldb = importlib.import_module("lldb")
    return lldb


def setup_debugger() -> Any:
    """Creates a global debugger instance and cache it."""
    global debugger
    if debugger is None:
        debugger = lldb.SBDebugger.Create(False)  # no load lldb init files

        # This sets up the debugger for running in the test environment.
        debugger.HandleCommand(
            "command source " + os.environ["LLDB_TEST_INIT_FILE"]
        )

        # This is needed to ensure that we wait for the debugger to stop any
        # time we issue stepping commands.
        debugger.SetAsync(False)

        cfg = configparser.ConfigParser()
        cfg.read(
            os.path.join(os.environ["MODULAR_DERIVED_PATH"], "modular.cfg")
        )
        plugin_path = cfg.get(section="mojo", option="lldb_plugin_path").strip(
            ";"
        )
        debugger.HandleCommand(f"plugin load {plugin_path}")
    return debugger


load_lldb()
setup_debugger()
