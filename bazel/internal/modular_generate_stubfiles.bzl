"""A wrapper for nanobind's stubgen.py with modular-specific defaults"""

load("@aspect_bazel_lib//lib:write_source_files.bzl", "write_source_files")
load("//bazel:config.bzl", "TOP_LEVEL_TAG")
load(":modular_genrule.bzl", "modular_genrule")
load(":modular_py_binary.bzl", "modular_py_binary")
load(":modular_py_library.bzl", "modular_py_library")

def modular_generate_stubfiles(
        name,
        extension,
        full_name,
        header,
        additional_update_targets = [],
        diff_test_failure_message = "",
        pyi_srcs = [],
        generated_pyi_srcs = [],
        include_private = False,
        pattern_file = None,
        diff_test_exclude = [],
        srcs = [],
        tags = [],
        deps = [],
        is_external = False,
        **kwargs):
    """A wrapper for nanobind's stubgen.py

    Generates stubfiles from a given nanobind extension, creates a py_library,
    and ensures the generated sources are the same as committed ones.

    Args:
        name: The name of the final library
        diff_test_failure_message: See write_source_files
        extension: The modular_nanobind_extension target to generate from
        full_name: The full python import path to the library, such as max._core
        header: The header to place at the top of each stubfile
        pyi_srcs: Stubfiles committed to the repo, to be compared with the generated ones
        additional_update_targets: See write_source_files
        generated_pyi_srcs: Stubfiles generated but not committed to the repo
        include_private: Whether to include private methods in the stubfiles. (Nanobind defaults to False)
        pattern_file: If specified, a nanobind pattern file to pass to stubgen.py
        diff_test_exclude: Optional list of pyi_src filenames to omit from the
            cross-platform diff test. They are still generated (and committed),
            but their diff test is expected to be declared separately at the call
            site under a narrower target_compatible_with -- use this for platform-
            specific stubs (e.g. nixl.pyi, which is only non-empty on linux-x86_64).
        srcs: See py_library, forwarded to the final library target
        deps: See py_library, forwarded to both the intermediate and final library targets
        is_external: True if running from the open source repository
        tags: Forwarded to both the intermediate and final library targets
        **kwargs: Forwarded to both the intermediate and final library targets
    """

    if bool(diff_test_failure_message) != bool(pyi_srcs):
        fail("modular_generate_stubfiles: set both `diff_test_failure_message` and `pyi_srcs` or neither")

    # Intermediate targets

    # Some finicky bits related to how we pull the binaries internally vs externally
    # Typing extensions needed for stubgen on 3.10
    stub_library_kwargs = {
        "deps": deps + [extension, "@modular_pip_lock_file_repo//deps:typing-extensions"],
    } if is_external else {
        "data": [extension],
        "deps": deps + ["@modular_pip_lock_file_repo//deps:typing-extensions"],
    }
    modular_py_library(
        name = name + ".stubfile_generator_library",
        tags = ["no-mypy", "no-pydeps"] + tags,
        **(stub_library_kwargs | kwargs)
    )

    modular_py_binary(
        name = name + ".stubfile_generator",
        srcs = ["@nanobind//:src/stubgen.py"],
        main = "@nanobind//:src/stubgen.py",
        tags = ["no-mypy", "no-pydeps"],
        deps = [":{}.stubfile_generator_library".format(name)],
    )

    outs = [("stubfiles/" + file) for file in pyi_srcs] + generated_pyi_srcs

    # Walk up from one of our own declared outputs to find the directory to
    # generate into. It cannot be derived from the generator's location: when
    # host and target differ, `tools` takes a real exec transition, so the
    # generator lands in a different output tree than these outputs.
    output_dir = "$(location {})".format(outs[0])
    for _ in range(outs[0].count("/") + 1):
        output_dir = "$(dirname {})".format(output_dir)

    modular_genrule(
        name = name + ".stubfiles",
        srcs = ([pattern_file] if pattern_file else []) + [
            # To ensure our linting + formatting aligns with the rest of our project
            "//:pyproject.toml",
        ],
        outs = outs,
        cmd = """
        set -e
        # When profiling is enabled, it replaces the bound methods entirely, which causes weird things to
        # happen with the stubfiles. Avoid the issue entirely by just forcing it off when generating.
        # Only necessary for max._core, but otherwise harmless.
        export MODULAR_ENABLE_PROFILING=OFF
        export BAZEL_TEST=1
        # Do our best to disable asan since we don't care about it for this case
        export ASAN_OPTIONS=verify_asan_link_order=0,detect_leaks=0,symbolize=0,halt_on_error=0,log_path=/dev/null

        export folder={output_dir}/stubfiles/{name}
        $(execpath {stubgen}) -r -m {full_name} -O $folder/.. {pattern_args} {include_private} --quiet

        # Add copyright headers
        for file in $(find $folder -name *.pyi); do
            echo '{header}'"\n$(cat $file)" > $file
        done

        # Lint and format the files. Needs to run twice to ensure lint results are reformatted.
        BUILD_WORKSPACE_DIRECTORY=$folder CHECK=0 FAST=0 $(execpath {ruff}) fix --quiet
        BUILD_WORKSPACE_DIRECTORY=$folder CHECK=0 FAST=0 $(execpath {ruff}) fix --quiet

        # Move generated sources to final location
        for file in {generated}; do
            mv $folder/../$file $folder/../../$file
        done
        """.format(
            name = name,
            full_name = full_name,
            header = header,
            output_dir = output_dir,
            pattern_args = "-p $(location {})".format(pattern_file) if pattern_file else "",
            include_private = "--include-private" if include_private else "",
            ruff = "//bazel/lint:ruff_wrapper",
            stubgen = ":{}.stubfile_generator".format(name),
            generated = " ".join(generated_pyi_srcs),
        ),
        tools = [
            ":{}.stubfile_generator".format(name),
            "//bazel/lint:ruff_wrapper",
        ],
    )

    # Compare the generated stubs to the committed ones.
    # We commit the changes for a few reasons:
    # 1. It is easier to see when they change
    # 2. For other tooling that uses them, such as docs or copybara.

    write_source_files(
        name = name + ".sync-stubfiles",
        additional_update_targets = additional_update_targets,
        diff_test_failure_message = diff_test_failure_message,
        # diff_test_exclude files are still generated above, but their diff test
        # is declared separately at the call site (under a narrower
        # target_compatible_with) rather than in this cross-platform suite.
        files = {
            file: ":stubfiles/" + file
            for file in pyi_srcs
            if file not in diff_test_exclude
        },
        target_compatible_with = select({
            # FIXME: grpc update causes this to fail in some obscure way on asan
            "//:asan": ["@platforms//:incompatible"],
            "//conditions:default": [],
        }),
        testonly = True,
        tags = [
            TOP_LEVEL_TAG,
            "stubfiles",
        ],
    )

    modular_py_library(
        name = name,
        data = [extension],
        pyi_srcs = pyi_srcs + generated_pyi_srcs,
        # Externally we use the max/_mlir/_mlir_libs/__init__.py
        # from the wheel since it contains a relative import
        srcs = [] if is_external else srcs,
        tags = tags,
        deps = deps,
        **kwargs
    )
