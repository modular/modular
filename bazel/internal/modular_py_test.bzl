"""A helper macro for running python tests with pytest"""

load("@rules_python//python:defs.bzl", "py_test")
load("//bazel/internal:config.bzl", "GPU_TEST_ENV", "RUNTIME_SANITIZER_DATA", "env_for_available_tools", "get_default_exec_properties", "get_default_test_env", "runtime_sanitizer_env", "validate_gpu_tags")  # buildifier: disable=bzl-visibility
load("//bazel/pip:pip_requirement.bzl", requirement = "pip_requirement")
load(":modular_py_venv.bzl", "modular_py_venv")
load(":mojo_collect_deps_aspect.bzl", "collect_transitive_mojoinfo")
load(":mojo_test_environment.bzl", "mojo_test_environment")
load(":py_repl.bzl", "py_repl")

def modular_py_test(
        name,
        srcs,
        deps = [],
        env = {},
        args = [],
        data = [],
        mojo_deps = [],
        tags = [],
        exec_properties = {},
        target_compatible_with = [],
        gpu_constraints = [],
        main = None,
        **kwargs):
    """Creates a pytest based python test target.

    Args:
        name: The name of the test target
        srcs: The test source files
        deps: py_library deps of the test target
        env: Any environment variables that should be set during the test runtime
        args: Arguments passed to the test execution
        data: Runtime deps of the test target
        mojo_deps: mojo_library targets the test depends on at runtime
        tags: Tags added to the py_test target
        exec_properties: https://bazel.build/reference/be/common-definitions#common-attributes
        target_compatible_with: https://bazel.build/extending/platforms#skipping-incompatible-targets
        gpu_constraints: GPU requirements for the tests
        main: If provided, this is the main entry point for the test. If not provided, pytest is used.
        **kwargs: Extra arguments passed through to py_test
    """

    validate_gpu_tags(tags, target_compatible_with + gpu_constraints)
    toolchains = [
        "//bazel/internal:current_gpu_toolchain",
        "//bazel/internal:lib_toolchain",
    ]

    has_test = False
    for src in srcs:
        if name == src.split("/")[0]:
            fail("modular_py_test targets cannot have the same 'name' as a directory: {}. Rename the bazel target or the directory".format(name))
        if src.split("/")[-1].startswith("test_"):
            has_test = True

    if not main and not has_test:
        fail("At least 1 file in modular_py_test must start with 'test_' for pytest to discover them")

    extra_env = runtime_sanitizer_env() | {
        "PYTHONUNBUFFERED": "set",
    }
    extra_data = RUNTIME_SANITIZER_DATA
    transitive_mojo_deps = name + ".mojo_deps"
    collect_transitive_mojoinfo(
        name = transitive_mojo_deps,
        deps_to_scan = deps,
        target_compatible_with = gpu_constraints + target_compatible_with,
        testonly = True,
    )

    env_name = name + ".mojo_test_env"
    toolchains.append(env_name)
    extra_data += [env_name]  # buildifier: disable=list-append
    extra_env |= {
        "MODULAR_MOJO_MAX_COMPILERRT_PATH": "$(COMPILER_RT_PATH)",
        "MODULAR_MOJO_MAX_DRIVER_PATH": "$(MOJO_BINARY_PATH)",
        "MODULAR_MOJO_MAX_IMPORT_PATH": "$(COMPUTED_IMPORT_PATH)",
        "MODULAR_MOJO_MAX_LINKER_DRIVER": "$(MOJO_LINKER_DRIVER)",
        "MODULAR_MOJO_MAX_LLD_PATH": "$(LLD_PATH)",
        "MODULAR_MOJO_MAX_SHARED_LIBS": "$(COMPUTED_LIBS)",
        "MODULAR_MOJO_MAX_SYSTEM_LIBS": "$(MOJO_LINKER_SYSTEM_LIBS)",
    }
    mojo_test_environment(
        name = env_name,
        data = mojo_deps + [transitive_mojo_deps],
        testonly = True,
    )

    default_exec_properties = get_default_exec_properties(tags, gpu_constraints)
    extra_env |= get_default_test_env(exec_properties)

    if "requires-network" in tags:
        # Assume networking is used for huggingface and add the cache
        extra_env |= {"HF_ESCAPES_SANDBOX": "1"}

    test_srcs = [src for src in srcs if src.split("/")[-1].startswith("test_")]
    non_test_srcs = [src for src in srcs if not src.split("/")[-1].startswith("test_")]
    extra_env |= GPU_TEST_ENV

    modular_py_venv(
        name = name + ".venv",
        data = data + extra_data,
        deps = deps + [
            requirement("pytest"),
        ],
    )

    for target in [".debug", ".shell"]:
        py_repl(
            name = name + target,
            data = data + extra_data,
            deps = deps + [
                requirement("pytest"),
                "@rules_python//python/runfiles",
            ],
            direct = False,
            env = env_for_available_tools() | extra_env | env | {
                "DEBUG_SRCS": ":".join(["$(location {})".format(src) for src in srcs]),
                # TODO: This should be PYTHONINSPECT but that doesn't work. We're avoiding args so lldb works without --
                "PYTHONSTARTUP": "$(location //bazel/internal:test_debug_shim.py)",
                "REPL_TARGET": target,
            },
            srcs = srcs + ["//bazel/internal:test_debug_shim.py"],
            toolchains = toolchains,
            target_compatible_with = gpu_constraints + target_compatible_with,
        )

    if main:
        kwargs |= {
            "args": args,
            "main": main,
        }
    else:
        kwargs |= {
            "args": [native.package_name(), "-svv", "--color=yes", "--durations=3"] + args,
            "main": "pytest_runner.py",
        }

    if len(test_srcs) > 1:
        test_names = []
        for src in test_srcs:
            test_name = src.replace(".py", "")
            test_names.append(test_name)
            py_test(
                name = test_name,
                data = data + extra_data,
                toolchains = toolchains,
                env = env_for_available_tools() | extra_env | env,
                deps = deps + [
                    requirement("pytest"),
                    "@rules_python//python/runfiles",
                ],
                srcs = [src] + non_test_srcs + ["//bazel/internal:pytest_runner"],
                exec_properties = default_exec_properties | exec_properties,
                target_compatible_with = gpu_constraints + target_compatible_with,
                tags = tags,
                **kwargs
            )

        native.test_suite(
            name = name,
            tests = test_names,
            tags = ["manual"],
        )
    else:
        py_test(
            name = name,
            data = data + extra_data,
            toolchains = toolchains,
            env = env_for_available_tools() | extra_env | env,
            deps = deps + [
                requirement("pytest"),
                "@rules_python//python/runfiles",
            ],
            srcs = srcs + ["//bazel/internal:pytest_runner"],
            exec_properties = default_exec_properties | exec_properties,
            target_compatible_with = gpu_constraints + target_compatible_with,
            tags = tags + (["manual"] if len(test_srcs) > 1 else []),
            **kwargs
        )
