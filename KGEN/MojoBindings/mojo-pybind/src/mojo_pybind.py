# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import platform
import subprocess


def generate_mojo_extension_module(
    # Could be a .mojo or an already built .mlir file
    input_path: str,
    raw_bindings: bool,
    verbose: bool = False,
):
    filestem, fileext = os.path.splitext(os.path.basename(input_path))

    if fileext != ".mojo" and fileext != ".mlir":
        raise Exception(
            f"Expected to get .mojo or .mlir file (got {fileext}): {input_path}"
        )

    if not os.path.isfile(input_path):
        raise Exception(f"File at path does not exist: {input_path}")

    # --------------------------------------------------------
    # Compute paths to `kgen` and `kgen-translate` executables
    # --------------------------------------------------------

    if workspace := os.environ.get("BUILD_WORKSPACE_DIRECTORY"):
        # Environment during `bazel run` of mojo-pybind
        kgen = os.path.join(workspace, ".derived/build/bin/kgen")
        kgen_translate = os.path.join(
            workspace, ".derived/build/bin/kgen-translate"
        )

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
        kgen_translate = "kgen-translate"

    # ------------------------------------------------------------------------
    # Run kgen-translate to generate a .mlir containing the generated bindings
    # ------------------------------------------------------------------------

    # There are three scenarios we need to handle:
    #   1. mojo-pybind foo.mojo
    #   2. mojo-pybind --raw-bindings foo.mojo
    #   3. mojo-pybind foo.mlir
    #
    # In case (1) we have to build the intermediate .mlir file.
    # In case (2) we need to skip `kgen-translate -gen-pybind`.

    # The file we'll pass to `kgen` to build. Could be .mojo or .mlir.
    build_file: str

    if fileext == ".mlir":
        # This is case (3)
        build_file = input_path

    elif raw_bindings:
        # == This is case (2)
        assert fileext == ".mojo"

        build_file = input_path

    else:
        # == This must be case (1)
        assert fileext == ".mojo"

        build_file = filestem + ".mlir"

        kgen_translate_cmd = [
            kgen_translate,
            "-import-mojo",
            input_path,
            # Generate the Python bindings
            "-gen-pybind",
            "--mojo-enable-prebuilt-packages",
            "-o",
            build_file,
        ]

        _run_command(kgen_translate_cmd, verbose=verbose)

        assert os.path.isfile(build_file)

    # ----------------------------------
    # Run kgen to build a .a file
    # ----------------------------------

    archive_path = build_mojo_to_archive(
        build_file,
        filestem,
        kgen=kgen,
        verbose=verbose,
    )

    # ---------------------------------------
    # Run clang to link .a to dynamic library
    # ---------------------------------------

    link_mojo_archive_to_dylib(archive_path, filestem, verbose=verbose)


def build_mojo_to_archive(
    # Could be a .mojo source file, or a .mlir file containing Mojo code built
    # by kgen-translate. or .mlir file
    input_path: str,
    filestem: str,
    *,
    kgen: str,
    verbose: bool,
) -> str:
    """Build a .mlir file to a .a file.

    input_path:
        Either a .mojo source file, or a .mlir file containing prevous output
        from `kgen-translate -import-mojo <foo>.mojo`.

    Returns:
        The path to the .a file
    """

    archive_path = filestem + ".a"

    kgen_cmd = [
        kgen,
        # mojo_path,
        input_path,
        "-emit",  # Emits object file archive
        # TODO: Define this, might be useful for something else down the line?
        # "-D",
        # "MOJO_PYTHON_EXTENSION_MODULE",
        "-o",
        archive_path,
    ]

    _run_command(kgen_cmd, verbose=verbose)

    assert os.path.isfile(archive_path)

    return archive_path


def link_mojo_archive_to_dylib(
    archive_path: str, filestem: str, *, verbose: bool
):
    """Link a .a file containing compiled Mojo code with the Mojo interface
    libraries, and produce a .dylib.
    """

    # ----------------------------------
    # Run clang to link a dynamic library
    # ----------------------------------

    mojo_libs: str

    # FIXME: Don't hard-code `build-debug` path component here.
    if workspace := os.environ.get("BUILD_WORKSPACE_DIRECTORY"):
        ext = "dylib" if platform.system() == "Darwin" else "so"
        mojo_libs = os.path.join(
            workspace, f".derived/build-debug/lib/libKGENCompilerRTShared.{ext}"
        )
    elif static := os.environ.get("MODULAR_MOJO_MAX_COMPILERRT_PATH"):
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
        "clang++",
        "-shared",
        f"-Wl,-rpath,{os.path.dirname(mojo_libs)}",
    ]
    if platform.system() == "Linux":
        clang_cmd.extend(
            [
                "-Wl,--whole-archive",
                archive_path,
                "-Wl,--no-whole-archive",
            ]
        )
    elif platform.system() == "Darwin":
        clang_cmd.extend(
            [
                "-Wl,-force_load",
                archive_path,
            ]
        )

    clang_cmd.extend(
        [
            mojo_libs,
            "-o",
            filestem + ".so",
        ]
    )
    _run_command(clang_cmd, verbose=verbose)


def _run_command(cmd_args: list[str], *, verbose: bool):
    name = os.path.basename(cmd_args[0])

    if verbose:
        print(f"note: invoking {name}:\n\t", cmd_args)

    result = subprocess.run(cmd_args)

    # TODO: Print a better error if this fails.
    result.check_returncode()
