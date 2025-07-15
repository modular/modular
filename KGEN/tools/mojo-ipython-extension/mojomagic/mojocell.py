# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import collections
import ctypes
import importlib
import os
import pathlib
import sys
import uuid
from typing import Callable

import IPython  # type: ignore


# https://ipython.readthedocs.io/en/8.27.0/config/extensions/index.html#writing-extensions
def load_ipython_extension(ipython) -> None:  # noqa: ANN001
    ipython.register_magics(_MojoMagic)
    # https://ipython.readthedocs.io/en/8.27.0/config/callbacks.html
    ipython.events.register("pre_run_cell", _pre_exec_hook)


# Global verbose state stack, last element determines state
global _global_verbose_log
_global_verbose_log = [False]

# IPython magic cell execution does not provide a unique cell_id
# which is necessary for cell content mapping
# As a workaround, we can hook into pre cell run events and store
# the cell exec_info in the _global_ipython_exec_info var
global _global_ipython_exec_info
_global_ipython_exec_info = None


def _pre_exec_hook(exec_info) -> None:  # noqa: ANN001
    """pre_run_cell IPython event hook to store exec_info for later retrieval of cell_id"""
    global _global_ipython_exec_info
    _global_ipython_exec_info = exec_info


def _vlog(msg, prefix="") -> None:  # noqa: ANN001
    if not _global_verbose_log[-1]:
        return
    if isinstance(msg, dict):
        for k, v in msg.items():
            print(f'{prefix}["{k}"] = {v}')
    else:
        print(f"{prefix}{msg}")


def _libname() -> str:
    extension = "dylib" if sys.platform == "darwin" else "so"
    return f"libIPythonExtension.{extension}"


def _libpath() -> pathlib.Path:
    # todo: proper install path
    modularpath = pathlib.Path(os.environ.get("MODULAR_PATH", ""))
    libpath = (
        modularpath / "bazel-bin" / "KGEN" / "tools" / "mojo-ipython-extension"
    )
    return libpath


def _dllpath() -> pathlib.Path:
    path = _libpath() / _libname()
    path = path.resolve()
    assert path.is_file()
    return path


def _load_dll():
    dll_path = _dllpath().as_posix()
    sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(dll_path, ctypes.RTLD_GLOBAL)
    assert lib._handle
    return lib


def _load_function():
    funcname = "iPythonMagicMojoCellExecute"
    lib = _load_dll()
    func = getattr(lib, funcname)
    assert func

    # https://docs.python.org/3/library/ctypes.html#fundamental-data-types
    func.restype = ctypes.py_object
    func.argtypes = [ctypes.py_object]

    return func


def _parse_options(line: str) -> dict:
    """Parse string of the form:
    key1=value key2=value ...
    parses value to bool(True) if lowercase value is true, 1, on, yes
    parses value to bool(False) if lowercase value is false, 0, off, no or empty
    """

    def parse_part(part):  # noqa: ANN001
        parts = part.split("=", 1)
        if len(parts) == 1:
            return (parts[0], True)
        lower_val = parts[1].lower()
        if lower_val in ["false", "0", "off", "no", ""]:
            return (parts[0], False)
        if lower_val in ["true", "1", "on", "yes"]:
            return (parts[0], True)
        return (parts[0], parts[1])

    keyvals = (parse_part(part) for part in line.split(" ")) if line else ()
    return collections.defaultdict(lambda: False, keyvals)


@IPython.core.magic.magics_class
class _MojoMagic(IPython.core.magic.Magics):
    """
    https://ipython.readthedocs.io/en/stable/api/generated/IPython.core.magic.html#IPython.core.magic.Magics
    IPython extension to provide %%mojo magic cell execution
    """

    func = None  # function pointer to the DLL IPythonMagicCellExecute function
    cell_contents: dict[str, str] = {}  # ordered map from cell_id -> contents
    symbol_table: dict[
        str, Callable[..., str]
    ] = {}  # function table from symbol_id -> function pointer

    @IPython.core.magic.cell_magic
    # @IPython.core.magic.no_var_expand
    # desired?: https://ipython.readthedocs.io/en/stable/api/generated/IPython.core.magic.html#IPython.core.magic.no_var_expand
    def mojo(self, line, cell) -> None:  # noqa: ANN001
        opts = _parse_options(line)

        # Push logging state, default to current state
        _global_verbose_log.append(opts.get("verbose", _global_verbose_log[-1]))

        use_symbol_table = opts.get("use_symbol_table", True)

        # exec_count is sequentially increasing
        config = get_ipython()  # type: ignore
        exec_count = config.execution_count
        _vlog(f"{exec_count=}")

        # cell_id is not passed in, must get from pre execution hook event
        global _global_ipython_exec_info
        cell_id = _global_ipython_exec_info.cell_id  # type: ignore
        _vlog(f"{cell_id=}")

        # delete any previous cached cell_contents
        # and always append to dict for correct out-of-order cell execution
        if cell_id in self.cell_contents:
            del self.cell_contents[cell_id]

        # Append this cell's contents to internal map from cell_id -> contents
        self.cell_contents[cell_id] = cell

        # Build ordered list of cell contents with comment containing cell_id
        all_cells = [
            f"\n# {id}\n{cell}" for id, cell in self.cell_contents.items()
        ]

        # Join all cell contents into a single unified buffer for compilation
        all_cells = "\n".join(all_cells)  # type: ignore

        # if option reload=True is enabled delete any previous function ptr
        if opts["reload"] and self.func:
            _vlog("unloading %%mojo extension")
            del self.func
            self.func = None

        if not self.func:
            _vlog("reloading %%mojo extension")
            self.func = _load_function()

        uid = uuid.uuid4().hex
        basename = f"mojo_magic_cell_{uid}"
        modulename = f"{basename}.mojo"

        data = {
            "line": line,
            "cell": all_cells,
            "build": opts.get("build", True),
            "uid": uid,
            "modulename": modulename,
        }

        _vlog(data, "data")

        result = self.func(data)

        _vlog({k: v for k, v in result.items() if k != "cell"}, "result")

        if error_msg := result.get("error_msg"):
            print(f"%%mojo error: {error_msg}", file=sys.stderr)
            return

        # functionally equivalent to import module w/ custom name
        module = importlib.import_module(basename)

        symbols = (name for name in dir(module) if not name.startswith("__"))

        def wrapper(*args, name):  # noqa: ANN001
            mojofunc = self.symbol_table[name]
            return mojofunc(*args)

        for sym in symbols:
            mojofunc = getattr(module, sym)
            _vlog(f"# from {basename} import {sym} as {sym}")
            if use_symbol_table:
                self.symbol_table[sym] = mojofunc
                self.shell.user_ns[sym] = lambda *args, name=sym: wrapper(
                    *args, name=name
                )
            else:
                self.shell.user_ns[sym] = mojofunc

        # restore global verbose state
        _global_verbose_log.pop()
        return
