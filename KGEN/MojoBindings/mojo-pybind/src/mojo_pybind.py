# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import subprocess


def generate_mojo_extension_module(
    mojo_path: str,
    verbose: bool = False,
):
    if not os.path.isfile(mojo_path):
        raise Exception(f"Mojo file at path does not exist: {mojo_path}")

    filestem = os.path.splitext(os.path.basename(mojo_path))[0]

    # ----------------------------------
    # Run kgen to build a .a file
    # ----------------------------------

    if workspace := os.environ.get("BUILD_WORKSPACE_DIRECTORY"):
        # Environment during `bazel run` of mojo-pybind
        kgen = os.path.join(workspace, ".derived/build/bin/kgen")

        assert os.path.isfile(kgen)

        # NOTE:
        #   Set MODULAR_DERIVED_PATH so that `kgen` is able to locate the
        #   built Mojo stdlib artifact.
        # FIXME: Better way to indicate this dependency to Bazel?
        os.environ["MODULAR_DERIVED_PATH"] = os.path.join(workspace, ".derived")
    else:
        # Environment during `bazel test` of the integration tests
        # that happen to use mojo-pybind.
        # kgen = os.environ["MODULAR_MOJO_DRIVER_PATH"]
        # Rely on `kgen` being on PATH.
        kgen = "kgen"

    archive_path = filestem + ".a"

    assert os.path.isfile(mojo_path)

    kgen_cmd = [
        kgen,
        mojo_path,
        "-emit",  # Emits object file archive
        # TODO: Define this, might be useful for something else down the line?
        # "-D",
        # "MOJO_PYTHON_EXTENSION_MODULE",
        "-o",
        archive_path,
    ]

    if verbose:
        print("note: invoking kgen:\n\t", kgen_cmd)

    result = subprocess.run(kgen_cmd)

    # TODO: Print a better error if this fails.
    result.check_returncode()

    assert os.path.isfile(archive_path)

    # ----------------------------------
    # Run clang to link a dynamic library
    # ----------------------------------

    # FIXME: Don't hard-code `build-debug` path component here.
    if workspace := os.environ.get("BUILD_WORKSPACE_DIRECTORY"):
        mojo_libs = os.path.join(
            workspace, ".derived/build-debug/bin/libKGENCompilerRT-static.a"
        )
    elif static := os.environ.get("MODULAR_MOJO_MAX_COMPILERRT_STATIC_PATH"):
        mojo_libs = static

    assert os.path.isfile(mojo_libs)

    # Note:
    #   `-force_load` linker option will force all symbols from the immediate
    #   next file (bindings.a) to get loaded by the linker into the final
    #   executable. This is needed because `PyInit_bindings` is otherwise not
    #   included.
    #
    #   TODO: -force_load is probably _too_ many symbols.
    #
    # Note:
    #   Python's dylib module loading logic looks for .so even on macOS
    #   (not .dylib!), so that's why .so is hard-coded here.
    clang_cmd = [
        "clang",
        "-shared",
        "-Wl,-force_load",
        archive_path,
        mojo_libs,
        "-lc++",
        "-o",
        filestem + ".so",
    ]

    if verbose:
        print("note: invoking clang:\n\t", clang_cmd)

    result = subprocess.run(clang_cmd)

    # TODO: Print a better error if this fails.
    result.check_returncode()
