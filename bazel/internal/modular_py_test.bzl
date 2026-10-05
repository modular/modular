"""A helper macro for running python tests with pytest"""

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("//bazel:config.bzl", "ALLOW_UNUSED_TAG")
load("//bazel/internal:config.bzl", "GPU_TEST_ENV", "RUNTIME_SANITIZER_DATA", "env_for_available_tools", "get_default_exec_properties", "get_default_test_env", "get_resources_exec_properties", "get_resources_tags", "runtime_sanitizer_env", "validate_gpu_tags")  # buildifier: disable=bzl-visibility
load("//bazel/pip:pip_requirement.bzl", requirement = "pip_requirement")
load("//bazel/pip/pydeps:pydeps_test.bzl", "pydeps_test")
load(":modular_py_venv.bzl", "modular_py_venv")
load(":mojo_collect_deps_aspect.bzl", "collect_transitive_mojoinfo")
load(":mojo_test_environment.bzl", "mojo_test_environment")
load(":py_imports.bzl", "compute_py_imports")
load(":py_repl.bzl", "py_repl")
load(":pytest_record.bzl", "pytest_record")

def _get_manual_srcs(tags, per_test_tags, srcs):
    # Srcs that default builds skip, so mypy has to see them via a separate
    # library. A no-mypy suppression opts out, at either granularity.
    if "no-mypy" in tags:
        return []

    if "manual" in tags or "postsubmit" in tags:
        return srcs

    result = []
    for src in srcs:
        src_tags = per_test_tags.get(src, [])
        if "no-mypy" in src_tags:
            continue
        if "manual" in src_tags or "postsubmit" in src_tags:
            result.append(src)

    return result

# Apple has no virtual compilation device (MOCO-2411), so nothing there can
# compile for a GPU it does not have; such a lane compiles on the GPU as it did
# before the attribute was set.
_NO_RECORD = select({
    "//:apple_gpu": ["@platforms//:incompatible"],
    "//conditions:default": [],
})

_PLUGIN = "//bazel/internal:precompile_mefs_plugin"

_PLUGIN_MODULE = "precompile_mefs_plugin"

# What a test that is not precompiled adds to itself, so the `py_test` below
# has one spelling either way.
_NOT_PRECOMPILED = struct(args = [], deps = [], data = [], env = {})

# What the test's own environment says about the GPU it expects, which the
# record action has none of. A non-zero memory-manager size in particular makes
# `pytest_runner.py` import torch and poke `torch.cuda`.
_RECORD_STRIP_ENV = [
    "GPU_ENV_DO_NOT_USE",
    "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT",
    "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_ONLY",
    "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE",
]

def _wire_precompile(
        test_name,
        test_srcs,
        shards,
        device_count,
        pytest_args,
        deps,
        data,
        env,
        mojo_deps,
        imports,
        gpu_constraints,
        target_compatible_with):
    """Emits the record half of one per-file test, on CPU.

    The producer is the test's own pytest invocation as a `py_binary`, and the
    record rule runs it as a build action per shard. Nothing about the program
    differs from the test; the plugin reads the environment the rule sets and
    records instead of replaying.

    The producer is built in the exec configuration, because that is what it
    is: a tool that runs during the build. So it gets its own mojo environment
    rather than the test's, whose GPU constraints no CPU exec platform can
    satisfy, and none of the test's GPU environment, which the record action
    strips anyway.

    Args:
        test_name: The consumer test's name, which the generated targets extend.
        test_srcs: The consumer test's srcs, including the pytest runner.
        shards: How many record actions to split the compiles across.
        device_count: How many virtual accelerators to present.
        pytest_args: The arguments the consumer passes to pytest.
        deps: The consumer's deps.
        data: The consumer's data.
        env: The consumer's env.
        mojo_deps: The consumer's `mojo_deps`.
        imports: The consumer's import paths.
        gpu_constraints: The consumer's GPU requirements, which say which arch
            to record for.
        target_compatible_with: The consumer's own constraints.

    Returns:
        A struct of what to add to the consumer test so it replays.
    """

    # The test's own `.mojo_deps` is gated to its GPU lane, so the producer
    # collects the same libraries without that gate. The kernel packages are
    # what MAX compiles *with*; the arch it compiles *for* arrives as an
    # environment variable from the record rule.
    producer_mojo_deps = test_name + ".precompile.mojo_deps"
    collect_transitive_mojoinfo(
        name = producer_mojo_deps,
        deps_to_scan = deps,
        target_compatible_with = target_compatible_with,
        testonly = True,
    )
    producer_env = test_name + ".precompile.mojo_test_env"
    mojo_test_environment(
        name = producer_env,
        data = mojo_deps + [producer_mojo_deps],
        testonly = True,
    )

    py_binary(
        name = test_name + ".precompile",
        srcs = test_srcs,
        main = "pytest_runner.py",
        data = data + RUNTIME_SANITIZER_DATA + [producer_env],
        deps = deps + [
            requirement("pytest"),
            "@rules_python//python/runfiles",
            "//bazel/internal:pytest-shard",
            _PLUGIN,
        ],
        # Deliberately not the test's env: `GPU_TEST_ENV` and the
        # memory-manager settings describe a GPU worker this never runs on, and
        # the record action strips them. The mojo vars point at the producer's
        # own environment above.
        env = env_for_available_tools() | runtime_sanitizer_env() | {
            "PYTHONUNBUFFERED": "set",
            "MODULAR_MOJO_MAX_COMPILERRT_PATH": "$(COMPILER_RT_PATH)",
            "MODULAR_MOJO_MAX_DRIVER_PATH": "$(MOJO_BINARY_PATH)",
            "MODULAR_MOJO_MAX_IMPORT_PATH": "$(COMPUTED_IMPORT_PATH)",
            "MODULAR_MOJO_MAX_LINKER_DRIVER": "$(MOJO_LINKER_DRIVER)",
            "MODULAR_MOJO_MAX_LLD_PATH": "$(LLD_PATH)",
            "MODULAR_MOJO_MAX_SHARED_LIBS": "$(COMPUTED_LIBS)",
            "MODULAR_MOJO_MAX_SYSTEM_LIBS": "$(MOJO_LINKER_SYSTEM_LIBS)",
        } | env,
        imports = imports,
        toolchains = [producer_env],
        # Built only by the record rule below, and mypy already sees these
        # sources through the test.
        tags = ["manual", "no-mypy"],
        testonly = True,
        # No `gpu_constraints`: this runs during the build, on whatever CPU
        # platform the exec configuration picks. What it compiles *for* comes
        # from the record rule's own toolchain, and that rule carries the
        # lane's constraints.
        target_compatible_with = target_compatible_with,
        visibility = ["//visibility:private"],
    )

    pytest_record(
        name = test_name + ".mefs",
        producer = ":" + test_name + ".precompile",
        pytest_args = pytest_args + [
            "-p",
            "pytest-shard",
            "-p",
            _PLUGIN_MODULE,
            # A build action's output tree is not a place to leave a cache.
            "-p",
            "no:cacheprovider",
        ],
        shards = shards,
        env = {"PRECOMPILE_MEFS_DEVICE_COUNT": str(device_count)},
        strip_env = _RECORD_STRIP_ENV,
        test_label = "//{}:{}".format(native.package_name(), test_name),
        expansion_targets = data,
        exec_properties = get_resources_exec_properties(test_name, test = False),
        tags = ["manual"],
        testonly = True,
        target_compatible_with = gpu_constraints + target_compatible_with + _NO_RECORD,
        visibility = ["//visibility:private"],
    )

    return struct(
        args = ["-p", _PLUGIN_MODULE],
        deps = [_PLUGIN],
        # The lane `_NO_RECORD` marks incompatible above: there is nothing to
        # depend on or replay from there.
        data = select({
            "//:apple_gpu": [],
            "//conditions:default": [":" + test_name + ".mefs"],
        }),
        env = select({
            "//:apple_gpu": {},
            "//conditions:default": {
                "PRECOMPILE_MEFS_MODE": "replay",
                "PRECOMPILE_MEFS_REPLAY_RLOCATIONS": "$(rlocationpaths :{}.mefs)".format(test_name),
            },
        }),
    )

def modular_py_test(
        name,
        srcs,
        deps = [],
        env = {},
        args = [],
        data = [],
        ignore_extra_deps = [],
        ignore_unresolved_imports = [],
        mojo_deps = [],
        tags = [],
        exec_properties = {},
        target_compatible_with = [],
        gpu_constraints = [],
        main = None,
        imports = [],
        per_test_tags = {},
        test_name_prefix = "",
        shard_count = None,
        per_test_shard_count = {},
        precompile_mefs = False,
        precompile_shards = None,
        precompile_device_count = 1,
        **kwargs):
    """Creates a pytest based python test target.

    Args:
        name: The name of the test target
        srcs: The test source files
        deps: py_library deps of the test target
        env: Any environment variables that should be set during the test runtime
        args: Arguments passed to the test execution
        data: Runtime deps of the test target
        ignore_extra_deps: Forwarded to pydeps_test
        ignore_unresolved_imports: Forwarded to pydeps_test
        mojo_deps: mojo_library targets the test depends on at runtime
        tags: Tags added to the py_test target
        exec_properties: https://bazel.build/reference/be/common-definitions#common-attributes
        target_compatible_with: https://bazel.build/extending/platforms#skipping-incompatible-targets
        gpu_constraints: GPU requirements for the tests
        main: If provided, this is the main entry point for the test. If not provided, pytest is used.
        imports: Additional python import paths
        per_test_tags: A mapping of source files to extra tags to apply to that test file.
        test_name_prefix: Prefix added to per-src py_test target names (multi-source only).
        shard_count: Forwarded to the underlying test target.
        per_test_shard_count: Forwarded to the underlying test target for the specified source(s).
        precompile_mefs: Compile this test's graphs in a CPU build action and
            initialize the artifacts on the GPU, rather than compiling there.
            See `precompile_mefs_plugin.py` and
            `docs/internal/CompileOnCpuRunOnGpu.md`.
        precompile_shards: How many build actions to split the compiles across
            (an int, or a `{src: int}` mapping when there are several tests).
            Defaults to the test's own shard count, which is unrelated to it:
            nothing correlates a record shard with a test shard.
        precompile_device_count: How many virtual accelerators to present while
            recording. Defaults to 1.
        **kwargs: Extra arguments passed through to py_test
    """

    if len(imports) > 1:
        fail("modular_py_test only supports a single import path.")

    if len(set(per_test_tags.keys()) - set(srcs)) != 0:
        fail("keys specified in per_test_tags that are not source files: {}".format(set(per_test_tags.keys()) - set(srcs)))

    if len(set(per_test_shard_count.keys()) - set(srcs)) != 0:
        fail("keys specified in per_test_shard_count that are not source files: {}".format(set(per_test_shard_count.keys()) - set(srcs)))

    if "gpu" in tags and "enable-sanitizers" in tags:
        fail("gpu + sanitizers are able to be run manually, but not in CI. remove `enable-sanitizers`.")

    if precompile_mefs:
        if main:
            fail("precompile_mefs runs the test's own pytest invocation, so it cannot be used with a custom `main`")
        if "gpu" not in tags:
            fail("precompile_mefs moves compiles off a GPU worker, so it only makes sense on a test tagged `gpu`")
        if not gpu_constraints:
            fail("precompile_mefs compiles for the accelerator the lane's toolchain names, so the test has to declare which lanes it runs on with `gpu_constraints`")

    validate_gpu_tags(tags, target_compatible_with + gpu_constraints)
    toolchains = [
        "//bazel/internal:current_gpu_toolchain",
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
        target_compatible_with = gpu_constraints + target_compatible_with,
        deps = deps + [
            requirement("pytest"),
        ],
    )

    py_repl(
        name = name + ".debug",
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
        },
        srcs = srcs + ["//bazel/internal:test_debug_shim.py"],
        toolchains = toolchains,
        target_compatible_with = gpu_constraints + target_compatible_with,
    )

    if main:
        final_args = args
        final_main = main
    else:
        final_args = [native.package_name(), "-svv", "--color=yes", "--durations=3"] + args
        final_main = "pytest_runner.py"

    manual_srcs = _get_manual_srcs(tags, per_test_tags, srcs)
    if manual_srcs:
        # Non-test srcs are sibling helper modules the manual tests import, so
        # mypy needs them here too (they're already in manual_srcs when the
        # whole target is manual).
        mypy_srcs = manual_srcs + [src for src in non_test_srcs if src not in manual_srcs]

        # TODO: Remove once we run mypy-style lints in a separate test target.
        # Raw py_library, not modular_py_library: the latter loads
        # modular_py_test, so depending back on it would cycle.
        py_library(
            name = name + ".mypy_library",
            tags = [ALLOW_UNUSED_TAG, "no-pydeps"],
            deps = deps + [
                requirement("pytest"),
                "@rules_python//python/runfiles",
            ],
            testonly = True,
            srcs = mypy_srcs + ["//bazel/internal:pytest_runner"],
            visibility = ["//visibility:private"],
            imports = compute_py_imports(native.package_name(), imports),
        )

    if len(test_srcs) > 1:
        if shard_count:
            fail("do not use shard_count when there are multiple tests, use per_test_shard_count")

        test_names = []
        for src in test_srcs:
            n_shards = per_test_shard_count.get(src)

            # If a custom main is used, it is responsible for sharding via
            # TEST_SHARD_INDEX and TEST_TOTAL_SHARDS env vars.
            use_shard_plugin = n_shards and not main
            shard_args = ["-p", "pytest-shard"] if use_shard_plugin else []
            test_name = test_name_prefix + src.replace(".py", "")
            test_names.append(test_name)
            test_srcs_with_runner = [src] + non_test_srcs + ["//bazel/internal:pytest_runner"]
            replay = _NOT_PRECOMPILED
            if precompile_mefs:
                requested = precompile_shards.get(src) if type(precompile_shards) == "dict" else precompile_shards
                replay = _wire_precompile(
                    test_name = test_name,
                    test_srcs = test_srcs_with_runner,
                    shards = requested or n_shards or 1,
                    device_count = precompile_device_count,
                    pytest_args = final_args,
                    deps = deps,
                    data = data,
                    env = env,
                    mojo_deps = mojo_deps,
                    imports = imports,
                    gpu_constraints = gpu_constraints,
                    target_compatible_with = target_compatible_with,
                )
            py_test(
                name = test_name,
                data = data + extra_data + replay.data,
                main = final_main,
                args = final_args + shard_args + replay.args,
                toolchains = toolchains,
                env = env_for_available_tools() | extra_env | env | replay.env,
                deps = deps + [
                    requirement("pytest"),
                    "@rules_python//python/runfiles",
                ] + (["//bazel/internal:pytest-shard"] if use_shard_plugin else []) + replay.deps,
                shard_count = n_shards,
                srcs = test_srcs_with_runner,
                exec_properties = default_exec_properties | get_resources_exec_properties(test_name, test = True) | exec_properties,
                target_compatible_with = gpu_constraints + target_compatible_with,
                tags = tags + get_resources_tags(test_name) + per_test_tags.get(src, []),
                imports = imports,
                **kwargs
            )

        native.test_suite(
            name = name,
            tests = test_names,
            tags = ["manual"],
        )
    else:
        if per_test_tags:
            fail("Don't use `per_test_tags` if only one source file is specified, use `tags` directly.")
        if per_test_shard_count:
            fail("do not use per_test_shard_count with only one test, use shard_count")

        # If a custom main is used, it is responsible for sharding via
        # TEST_SHARD_INDEX and TEST_TOTAL_SHARDS env vars.
        use_shard_plugin = shard_count and not main
        shard_args = ["-p", "pytest-shard"] if use_shard_plugin else []

        replay = _NOT_PRECOMPILED
        if precompile_mefs:
            if type(precompile_shards) == "dict":
                fail("precompile_shards takes an int when there is only one test, not a mapping")
            replay = _wire_precompile(
                test_name = name,
                test_srcs = srcs + ["//bazel/internal:pytest_runner"],
                shards = precompile_shards or shard_count or 1,
                device_count = precompile_device_count,
                pytest_args = final_args,
                deps = deps,
                data = data,
                env = env,
                mojo_deps = mojo_deps,
                imports = imports,
                gpu_constraints = gpu_constraints,
                target_compatible_with = target_compatible_with,
            )

        # test_name_prefix intentionally doesn't apply here: single-source
        # collisions happen at the `name` arg, which callers pick themselves.
        py_test(
            name = name,
            data = data + extra_data + replay.data,
            toolchains = toolchains,
            env = env_for_available_tools() | extra_env | env | replay.env,
            main = final_main,
            args = final_args + shard_args + replay.args,
            deps = deps + [
                requirement("pytest"),
                "@rules_python//python/runfiles",
            ] + (["//bazel/internal:pytest-shard"] if use_shard_plugin else []) + replay.deps,
            shard_count = shard_count,
            srcs = srcs + ["//bazel/internal:pytest_runner"],
            exec_properties = default_exec_properties | get_resources_exec_properties(name, test = True) | exec_properties,
            target_compatible_with = gpu_constraints + target_compatible_with,
            tags = tags + get_resources_tags(name),
            imports = imports,
            **kwargs
        )

    if "no-pydeps" not in tags:
        pydeps_test(
            name = name + ".pydeps_test",
            data = data,
            srcs = srcs,
            # We provide these as a convenience, okay if not used.
            # The plugin is loaded with `-p`, never imported, so pydeps
            # must not flag it as unused.
            ignore_extra_deps = ignore_extra_deps + [
                requirement("pytest"),
                "@rules_python//python/runfiles",
            ] + ([_PLUGIN] if precompile_mefs else []),
            ignore_unresolved_imports = ignore_unresolved_imports,
            target_compatible_with = select({
                # No point in running these, causes "error replanting symlinks" failures
                "//:asan": ["@platforms//:incompatible"],
                "//:ubsan": ["@platforms//:incompatible"],
                "//conditions:default": [],
            }),
            imports = imports,
            deps = deps + [
                requirement("pytest"),
                "@rules_python//python/runfiles",
            ],
            tags = ["pydeps"],
        )
