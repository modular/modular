"""Targets for a `bench_sweep` benchmark.

A sweep runs every config in one process, so its targets differ from a plain
`mojo_test` in two ways: the process needs the device memory its fixture
plan allocates, and its comptime arms can each be pinned into a separate
executable with `-D`, when a sweep too heavy for one compile is split across
machines. `mojo_sweep_benchmark` creates both.
"""

load("@bazel_skylib//rules:build_test.bzl", _build_test = "build_test")
load("//bazel:api.bzl", "mojo_test")

def _arm_points(arm_axes):
    """Returns the `(-D` defines, tag`)` pair of every comptime arm.

    The tag names a point of the product of `arm_axes`, with each axis
    labeled by its define minus the `sweep_` prefix that `arm_enabled`
    conventions carry, so `{"sweep_h": [8, 64], "sweep_seq": [6, 8192]}`
    yields tags such as `h8_seq6` and `h64_seq8192`.
    """
    if not arm_axes:
        return []
    points = [([], "")]
    for define, values in arm_axes.items():
        if not values:
            fail("arm axis {} has no values".format(define))
        label = define[len("sweep_"):] if define.startswith("sweep_") else define
        points = [
            (
                defines + ["{}={}".format(define, value)],
                tag + ("_" if tag else "") + label + str(value),
            )
            for value in values
            for defines, tag in points
        ]
    tags = [tag for _, tag in points]
    if len(tags) != len(set(tags)):
        fail("arm_axes {} produce duplicate arm tags".format(arm_axes))
    return points

def mojo_sweep_benchmark(
        name,
        srcs,
        deps,
        gpu_memory,
        arm_axes = {},
        args = [],
        build_test = True,
        copts = [],
        enable_assertions = False,
        exec_properties = {},
        manual = True,
        size = "enormous",
        tags = [],
        target_compatible_with = [],
        **kwargs):
    """Creates the test targets of a sweep benchmark.

    The combined target `<name>.run` is the one to time: one process, one
    compile, every config. `arm_axes` adds one target per comptime arm, the
    way a comptime-parameterized kbench sweep builds them, each pinning the
    axis with `-D` and tagging its `SWEEPTIME` and JSON lines. The
    `<name>.test` build test keeps every executable compiled under
    `bazel test //...` while `manual` keeps the runs opt-in.

    Args:
        name: Conventionally the benchmark's source file, `foo.mojo`. The
            combined target is `<name>.run`, an arm target is
            `<foo>_<arm tag>.mojo.run`, and the build test is `<name>.test`.
        srcs: The benchmark's `.mojo` sources.
        deps: The benchmark's Mojo dependencies, `bench_sweep` included.
        gpu_memory: Device memory to request, in GiB. Taken from the sweep's
            fixture plan (`SWEEPINFO ... planned_bytes`, rounded up) plus the
            0.6 GiB the memory manager reserves; becomes the
            `test.resources:gpu-memory` execution property.
        arm_axes: The comptime axes to build one target per arm for, as a
            `define -> values` dict matching the benchmark's `arm_enabled`
            calls. A define keeps its `sweep_` prefix in `-D` and loses it in
            the arm tag.
        args: Test arguments; `--tag=<arm tag>` is appended to each arm
            target so its `SWEEPTIME` and JSON lines are told apart.
        build_test: Whether to create `<name>.test`, a build test covering
            the combined target and every arm.
        copts: Extra compile options; the arm's `-D` pins are appended.
        enable_assertions: Off by default: assertions slow the measured loop,
            and not every kernel runs with them on.
        exec_properties: Extra execution properties, merged under the GPU
            memory request.
        manual: Whether the `.run` targets stay out of `bazel test //...`.
            Compile coverage stays, through the build test.
        size: Test size; a sweep runs for minutes, hence `enormous`.
        tags: Extra test tags; `gpu` is always added.
        target_compatible_with: The GPUs the sweep runs on.
        **kwargs: Passed through to every `mojo_test`, such as `data` and
            `env`.
    """
    tags = ["gpu"] + [tag for tag in tags if tag != "gpu"]
    run_tags = tags + (["manual"] if manual else [])
    exec_properties = dict(exec_properties)
    if "test.resources:gpu-memory" in exec_properties:
        fail("gpu_memory and exec_properties both set the GPU memory request")
    exec_properties["test.resources:gpu-memory"] = str(gpu_memory)

    targets = []
    mojo_test(
        name = name + ".run",
        srcs = srcs,
        args = args,
        copts = copts,
        deps = deps,
        enable_assertions = enable_assertions,
        exec_properties = exec_properties,
        size = size,
        tags = run_tags,
        target_compatible_with = target_compatible_with,
        **kwargs
    )
    targets.append(name + ".run")

    stem = name[:-len(".mojo")] if name.endswith(".mojo") else name
    for defines, tag in _arm_points(arm_axes):
        arm_copts = list(copts)
        for define in defines:
            arm_copts += ["-D", define]
        arm_name = "{}_{}.mojo.run".format(stem, tag)
        mojo_test(
            name = arm_name,
            srcs = srcs,
            args = args + ["--tag=" + tag],
            copts = arm_copts,
            deps = deps,
            enable_assertions = enable_assertions,
            exec_properties = exec_properties,
            size = size,
            tags = run_tags,
            target_compatible_with = target_compatible_with,
            **kwargs
        )
        targets.append(arm_name)

    if build_test:
        _build_test(
            name = name + ".test",
            tags = tags,
            targets = targets,
        )
