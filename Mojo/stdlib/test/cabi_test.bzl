"""Builds C ABI cross-check suites for Mojo mirrors of C declarations."""

load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("//bazel:api.bzl", "modular_cc_library", "mojo_binary", "mojo_library", "mojo_test")

def _only_on(platforms):
    constraints = {"@platforms//os:" + os: [] for os in platforms}
    constraints["//conditions:default"] = ["@platforms//:incompatible"]
    return select(constraints)

def mojo_cabi_test(name, platforms):
    """Cross-checks a manifest's Mojo mirrors against the C headers.

    `name` is a directory holding the manifest package. Its `__init__.mojo`
    defines `CABI_INCLUDES`, `CABI_STRUCTS`, `CABI_TYPEDEFS`, and
    `CABI_CONSTANTS` (see `test_utils.cabi_check`). The macro generates a
    C reference from the manifest, compiles it against the platform
    headers, and adds the test `test_<name>.mojo.test`, which compares the
    mirrors against it.

    The generator runs on the build machine, whose OS can differ from the
    test target, so a manifest lists concrete per-platform mirror types and
    never selects them with a compile-time platform check.

    Args:
        name: The manifest package's directory, which also names the suite.
        platforms: The operating systems the suite runs on, such as
            `["linux", "macos"]`. Every other platform skips it.
    """
    compatible_with = _only_on(platforms)

    mojo_library(
        name = name,
        testonly = True,
        srcs = [name + "/__init__.mojo"],
        deps = [
            "@mojo//:std",
            "@mojo//:test_utils",
        ],
    )

    write_file(
        name = name + "_gen_src",
        testonly = True,
        out = name + "_gen.mojo",
        content = [
            "from {} import (".format(name),
            "    CABI_CONSTANTS,",
            "    CABI_INCLUDES,",
            "    CABI_STRUCTS,",
            "    CABI_TYPEDEFS,",
            ")",
            "from test_utils.cabi_check import emit_cabi_checks_for",
            "",
            "",
            "def main():",
            "    print(",
            "        emit_cabi_checks_for(",
            "            includes=materialize[CABI_INCLUDES](),",
            "            structs=CABI_STRUCTS,",
            "            typedefs=CABI_TYPEDEFS,",
            "            constants=materialize[CABI_CONSTANTS](),",
            "        )",
            "    )",
        ],
    )

    # Built for the build machine as a genrule tool, so it carries no
    # platform constraint of its own.
    mojo_binary(
        name = name + "_gen",
        testonly = True,
        srcs = [name + "_gen.mojo"],
        deps = [
            ":" + name,
            "@mojo//:std",
            "@mojo//:test_utils",
        ],
    )

    native.genrule(
        name = name + "_reference_src",
        testonly = True,
        outs = [name + "_reference.c"],
        cmd = "$(location :{}_gen) > $@".format(name),
        tools = [":" + name + "_gen"],
        target_compatible_with = compatible_with,
    )

    modular_cc_library(
        name = name + "_reference",
        testonly = True,
        srcs = [name + "_reference.c"],
        # Generated source; nothing for clang-tidy to check.
        tags = ["no-clang-tidy"],
        target_compatible_with = compatible_with,
    )

    write_file(
        name = "test_" + name + "_src",
        testonly = True,
        out = "test_" + name + ".mojo",
        content = [
            "from {} import CABI_CONSTANTS, CABI_STRUCTS, CABI_TYPEDEFS".format(
                name,
            ),
            "from std.testing import TestSuite",
            "from test_utils.cabi_check import assert_cabi_checks_for",
            "",
            "",
            "def test_cabi_checks() raises:",
            "    assert_cabi_checks_for[",
            "        structs=CABI_STRUCTS,",
            "        typedefs=CABI_TYPEDEFS,",
            "        constants=CABI_CONSTANTS,",
            "    ]()",
            "",
            "",
            "def main() raises:",
            "    TestSuite.discover_tests[__functions_in_module()]().run()",
        ],
    )

    mojo_test(
        name = "test_" + name + ".mojo.test",
        srcs = ["test_" + name + ".mojo"],
        copts = [
            "--debug-level",
            "full",
        ],
        tags = ["cabi"],
        target_compatible_with = compatible_with,
        deps = [
            ":" + name,
            ":" + name + "_reference",
            "@mojo//:std",
            "@mojo//:test_utils",
        ],
    )
