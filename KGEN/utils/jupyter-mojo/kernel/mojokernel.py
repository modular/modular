#!/usr/bin/python
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file contains an implementation of a Jupyter kernel for Mojo. It
# communicates to Mojo using the MojoJupyter API library.
#
# ===----------------------------------------------------------------------=== #


import argparse
import ctypes
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ipykernel.kernelapp import IPKernelApp
from ipykernel.kernelbase import Kernel


class MojoKernel(Kernel):
    """A Jupyter kernel for Mojo."""

    def __init__(self, **kwargs):
        """Initialize the Mojo kernel.

        This loads the MojoJupyter library and starts a kernel repl session.
        """
        # Kernel Metadata.
        self.implementation = "MojoKernel"
        self.implementation_version = "0.1"
        self.language = "mojo"
        self.language_version = "0.1"
        self.language_info = {
            "name": "mojo",
            "mimetype": "text/x-mojo",
            "file_extension": ".mojo",
        }
        self.banner = ""
        self.auto_gen_cell_id_count = 0
        super(MojoKernel, self).__init__(**kwargs)

        # Load the MojoJupyter library, and initialize the result types of the
        # functions we use.
        self.lib_mojo_jupyter: ctypes.CDLL = self.load_mojo_lib()
        self.lib_mojo_jupyter.initMojoKernel.restype = ctypes.c_void_p
        self.lib_mojo_jupyter.startMojoExecution.restype = ctypes.c_void_p
        self.lib_mojo_jupyter.checkMojoExecutionFinished.restype = ctypes.c_int

        # The type of the output callback function. It takes a name and a
        # message.
        self.output_callback_type: ctypes.CFUNCTYPE = ctypes.CFUNCTYPE(
            None, ctypes.c_char_p, ctypes.c_char_p
        )

        # Create the output callback function. This is called by the MojoJupyter
        # library to send output back to the Jupyter client.
        def output_callback(name: str, msg: str):
            stream_content = {
                "name": name.decode("utf-8"),
                "text": msg.decode("utf-8"),
            }
            self.send_response(self.iopub_socket, "stream", stream_content)

        self.output_callback = self.output_callback_type(output_callback)

        self.mojo_kernel: ctypes.c_void_p = (
            self.lib_mojo_jupyter.initMojoKernel(
                self.output_callback,
                ctypes.c_char_p(self.mojoReplExe.encode("utf-8")),
            )
        )
        if not self.mojo_kernel:
            raise RuntimeError("Unable to initialize Mojo kernel.")

    def __del__(self):
        """Destroy the Mojo kernel."""
        self.lib_mojo_jupyter.destroyMojoKernel(self.mojo_kernel)

    def load_mojo_lib(self) -> ctypes.CDLL:
        """Load the libMojoJupyter library.

        The location of the library is determined by the location of the
        `mojo-repl-entry-point` executable. The library should either be
        adjacent, or within a relative `../lib/` directory.

        On success, this initializes `mojoReplExe` returns the loaded library.
        """
        # Look for the mojo repl executable. This will have the various
        # necessary libraries adjacent to it.
        mojo_repl_exe_path = (
            Path(os.environ["MODULAR_PATH"]) / ".derived" / "build" / "lib"
        )
        os.environ["PATH"] += os.pathsep + str(mojo_repl_exe_path)
        self.mojoReplExe: Optional[str] = shutil.which("mojo-repl-entry-point")
        if not self.mojoReplExe:
            from distutils.spawn import find_executable

            self.mojoReplExe = find_executable("mojo-repl-entry-point")
            if not self.mojoReplExe:
                raise RuntimeError(
                    "Unable to locate `mojo-repl-entry-point` executable."
                )

        # Load the MojoJupyter library. This library provides the internal
        # implementation, and is located adjacent to the mojo repl executable or
        # within a relative ../lib/ directory.
        mojoReplDir = Path(self.mojoReplExe).parent
        for libDir in [mojoReplDir, mojoReplDir.parent / "lib"]:
            for ext in ["so", "dylib", "dll"]:
                libFilename = libDir / ("libMojoJupyter." + ext)
                if os.path.isfile(libFilename):
                    return ctypes.cdll.LoadLibrary(libFilename)

        raise RuntimeError("Unable to load `libMojoJupyter` library.")

    def do_execute(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[Dict[str, Any]] = None,
        allow_stdin: bool = False,
        *,
        cell_id: Optional[str] = None,
    ):
        """Execute a code cell."""
        # TODO: Better propagate errors from the kernel execution, process
        # provided arguments, etc.

        # jupyter on the cli doesn't provide a cell id, so we need to
        # autogenerate one.
        if cell_id is None:
            cell_id = f"__autogen_cell_id_{self.auto_gen_cell_id_count}"
            self.auto_gen_cell_id_count += 1

        # Start execution of the expression.
        executionState: ctypes.c_void_p = (
            self.lib_mojo_jupyter.startMojoExecution(
                ctypes.c_void_p(self.mojo_kernel),
                ctypes.c_char_p(cell_id.encode("utf-8")),
                ctypes.c_char_p(code.encode("utf-8")),
            )
        )

        # Wait for the execution to finish.
        while True:
            # Sleep for a bit to avoid busy spinning while waiting for the
            # execution to finish.
            time.sleep(0.05)

            # Poll the kernel to see if the execution has finished.
            result: bool = self.lib_mojo_jupyter.checkMojoExecutionFinished(
                ctypes.c_void_p(self.mojo_kernel),
                ctypes.c_void_p(executionState),
            )
            if result:
                break

        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--modular-path",
        required=True,
        help="The value of the env var MODULAR_PATH.",
    )
    args, jupyter_args = parser.parse_known_args()

    os.environ["MODULAR_PATH"] = args.modular_path

    # We pass the kernel name as a command-line arg, since Jupyter gives those
    # highest priority (in particular overriding any system-wide config).
    IPKernelApp.launch_instance(
        argv=jupyter_args + ["--IPKernelApp.kernel_class=__main__.MojoKernel"]
    )
