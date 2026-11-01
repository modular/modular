"""Wrapper for the upstream pyo3_extension rule to apply modular specific defaults."""

load("@rules_rust//extensions/pyo3:defs.bzl", "pyo3_extension")

def modular_pyo3_extension(
        name,
        srcs,
        crate_name,
        module_name,
        rustc_flags = [],
        target_compatible_with = [],
        deps = [],
        **kwargs):
    """A Rust Python extension.

    The counterpart of `modular_nanobind_extension` for pyo3. Both the internal
    and the OSS build compile it from source (SDLC-4549), so its sources and
    deps must be exported.

    Args:
        name: The name of the extension target.
        srcs: The Rust sources of the extension crate.
        crate_name: The Rust crate name, since `name` need not be a valid one.
        module_name: The module name, which must match the `#[pymodule]` function.
        rustc_flags: Extra flags passed to rustc.
        target_compatible_with: Platform constraints, added to the sanitizer gating below.
        deps: Rust crates the extension links. pyo3 itself is injected by the
             toolchain, so it must not be listed.
        **kwargs: Forwarded to `pyo3_extension`.
    """
    pyo3_extension(
        name = name,
        srcs = srcs,
        crate_name = crate_name,
        module_name = module_name,
        rustc_flags = rustc_flags + select({
            # Python C-API symbols are resolved at import time; cancel the
            # toolchain's default -Wl,-z,defs so the undefined refs pass.
            # Shipped stripped, like the nanobind extensions.
            "@platforms//os:linux": [
                "-C",
                "link-arg=-Wl,-z,undefs",
                "-C",
                "strip=symbols",
            ],
            # The macOS link runs on the Linux remote executor, which has
            # neither `dsymutil` nor `rust-objcopy`.
            "@platforms//os:macos": [
                "-C",
                "split-debuginfo=off",
                "-C",
                "strip=none",
            ],
            "//conditions:default": [],
        }),
        stubs = False,
        # Same gating as modular_nanobind_extension: these sanitizer runtimes
        # aren't available when Python loads the module.
        target_compatible_with = target_compatible_with + select({
            "//:asan_macos": ["@platforms//:incompatible"],
            "//:tsan": ["@platforms//:incompatible"],
            "//:ubsan": ["@platforms//:incompatible"],
            "//conditions:default": [],
        }),
        deps = deps,
        **kwargs
    )
