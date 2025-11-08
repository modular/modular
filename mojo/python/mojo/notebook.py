# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import argparse
import importlib
import re
import sys
import tempfile
import uuid
from pathlib import Path

try:
    # Don't require including IPython as a dependency

    from IPython.core.magic import register_cell_magic  # type: ignore
    from IPython.display import SVG, display
except ImportError:
    SVG, display = None, None

    def register_cell_magic(fn):  # noqa: ANN001, ANN201
        return fn


from .paths import MojoCompilationError
from .run import subprocess_run_mojo

# Enable Mojo import hook for dynamic compilation
try:
    import mojo.importer
except ImportError:
    pass  # Will fall back to subprocess approach


# Template for creating a Mojo module with Python bindings that can return any object
ENTRYPOINT_TEMPLATE = """\
from python import PythonObject
from python.bindings import PythonModuleBuilder
from os import abort

# --- User code:
{USER_CODE}

# --- Python module binding:
@export
fn PyInit_{MODNAME}() -> PythonObject:
    try:
        var m = PythonModuleBuilder("{MODNAME}")
        # Register the entrypoint function that should exist in user code:
        m.def_function[entrypoint]("entrypoint", docstring="Execute cell entry and return a PythonObject")
        return m.finalize()
    except e:
        return abort[PythonObject](String("error creating Python Mojo module:", e))
"""


@register_cell_magic
def mojo(line, cell) -> None:  # noqa: ANN001
    """A Mojo cell with enhanced Python integration.

    Usage:
        - Run Mojo code with entrypoint function (returns Python objects):

            ```mojo
            %%mojo entrypoint
            from python import PythonObject
            from IPython.display import SVG

            fn entrypoint() -> PythonObject:
                # Return any Python object - SVG, DataFrame, etc.
                svg_content = '''<?xml version="1.0"?>
                <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="50" cy="50" r="40" fill="red" />
                </svg>'''
                return SVG(data=svg_content)
            ```

        - Traditional main() approach (prints to stdout):

            ```mojo
            %%mojo
            def main():
                print("Hello from Mojo!")
            ```

        - Compile a python extension SO file:

            ```mojo
            %%mojo build --emit shared-lib -o mojo_module.so

            from python import PythonObject
            from python.bindings import PythonModuleBuilder
            from os import abort

            @export
            fn PyInit_mojo_module() -> PythonObject:
                try:
                    var m = PythonModuleBuilder("thing")
                    m.def_function[hello]("hello", docstring="Hello!")
                    return m.finalize()
                except e:
                    return abort[PythonObject](String("error creating Python Mojo module:", e))

            def hello() -> PythonObject:
                return "Hello from Mojo!"
            ```

            then in another cell

            ```python
            from mojo_module import hello

            hello()
            ```

        - Compile a package for kernel development.
            The following produces a `kernels.mojopkg` which may be included
            as custom ops in a graph via the `custom_extensions` mechanism.

            ```mojo
            %%mojo package -o kernels.mojopkg

            from runtime.asyncrt import DeviceContextPtr
            from tensor import InputTensor, ManagedTensorSlice, OutputTensor

            @compiler.register("histogram")
            struct Histogram:
                @staticmethod
                fn execute[
                    target: StaticString
                ](
                    output: OutputTensor[dtype = DType.int64, rank=1],
                    input: InputTensor[dtype = DType.uint8, rank=1],
                    ctx: DeviceContextPtr,
                ) raises:
                    ...
            ```

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="run")
    parser.add_argument(
        "--entrypoint",
        action="store_true",
        help="Use entrypoint mode for returning Python objects",
    )

    args, extra_args = parser.parse_known_args(line.strip().split())

    # Check if entrypoint mode is requested
    is_entrypoint = args.command == "entrypoint" or args.entrypoint

    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir)

        if is_entrypoint:
            # Use entrypoint approach with Python module binding
            modname = f"mojocell_{uuid.uuid4().hex[:8]}"

            # Validate module name
            if not re.match(r"^[A-Za-z_]\w*$", modname):
                raise ValueError(f"Invalid module name: {modname}")

            # Create Mojo source with entrypoint template
            mojo_content = ENTRYPOINT_TEMPLATE.format(
                USER_CODE=cell, MODNAME=modname
            )
            mojo_path = path / f"{modname}.mojo"
            with open(mojo_path, "w") as f:
                f.write(mojo_content)

            # Add the temp directory to Python path for import
            sys.path.insert(0, str(path))

            try:
                # Import the Mojo module (compiles and loads)
                mod = importlib.import_module(modname)

                # Call entrypoint function and display result
                result = mod.entrypoint()

                # Use IPython display for rich rendering
                if display:
                    display(result)
                else:
                    print(result)

            finally:
                # Clean up path (but keep temp directory for __mojocache__)
                if str(path) in sys.path:
                    sys.path.remove(str(path))

        else:
            # Use traditional approach with subprocess
            mojo_path = path / "cell.mojo"
            with open(mojo_path, "w") as f:
                f.write(cell)
            (path / "__init__.mojo").touch()

            input_path = path if args.command == "package" else mojo_path
            command = [
                args.command,
                str(input_path),
                *extra_args,
            ]

            result = subprocess_run_mojo(command, capture_output=True)

            if not result.returncode:
                stdout = result.stdout.decode()
                print(stdout)
            else:
                raise MojoCompilationError(
                    input_path,
                    command,
                    result.stdout.decode(),
                    result.stderr.decode(),
                )
