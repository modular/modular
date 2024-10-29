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

import IPython


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
    lib = ctypes.PyDLL(dll_path, ctypes.RTLD_GLOBAL)
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
    def parse_part(part):
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
# https://ipython.readthedocs.io/en/stable/api/generated/IPython.core.magic.html#IPython.core.magic.Magics
class MojoMagic(IPython.core.magic.Magics):
    func = None

    @IPython.core.magic.cell_magic
    # @IPython.core.magic.no_var_expand
    # desired?: https://ipython.readthedocs.io/en/stable/api/generated/IPython.core.magic.html#IPython.core.magic.no_var_expand
    def mojo(self, line, cell):
        if not self.func:
            self.func = _load_function()

        opts = _parse_options(line)

        uid = uuid.uuid4().hex
        basename = f"mojo_magic_cell_{uid}"
        modulename = f"{basename}.mojo"
        verbose = opts["verbose"]

        data = {
            "line": line,
            "cell": cell,
            "build": opts.get("build", True),
            "uid": uid,
            "modulename": modulename,
        }
        if verbose:
            print(data)
        result = self.func(data)
        if verbose:
            print(f"%%mojo {result=}")
        if error_msg := result.get("error_msg"):
            print(f"%%mojo error: {error_msg}", file=sys.stderr)
            return

        mojocellbindings = importlib.import_module(basename)

        # import mojocellbindings

        symbols = (
            name for name in dir(mojocellbindings) if not name.startswith("__")
        )
        for sym in symbols:
            if verbose:
                print(f"from {modulename} import {sym} as {sym}")
            self.shell.user_ns[sym] = getattr(mojocellbindings, sym)

        return

        # print(f'%%mojo magic executed, "mojo" object added to global state.')
