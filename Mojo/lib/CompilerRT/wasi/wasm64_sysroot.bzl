"""Builds the wasm64 wasi-libc sysroot as a Bazel tree artifact.

The output directory is a complete `--wasi-sysroot` for
`mojo build --target-triple wasm64-unknown-wasip1`: wasi-libc headers and
libc.a/crt1-command.o under lib/wasm64-wasip1, plus two archives installed
into that lib dir by the same action:

  - libclang_rt.builtins-wasm64.a   (compiler-rt builtins from @llvm-raw)
  - libkgen_compilerrt_wasi.a       (KGEN_CompilerRT_* glue, auto-linked
                                     by `mojo build`)
"""

def _wasm64_sysroot_impl(ctx):
    out = ctx.actions.declare_directory(ctx.label.name)

    clang_bin = None
    for f in ctx.files.clang_tools:
        if f.basename == "clang-22":
            clang_bin = f.dirname
    if clang_bin == None:
        fail("clang_tools must contain clang-22 (with llvm-ar/llvm-nm alongside)")

    inputs = depset(
        ctx.files.clang_tools +
        ctx.files.resource_dir +
        ctx.files.shim_srcs +
        ctx.files.wasi_libc_srcs +
        ctx.files.builtins_srcs +
        [ctx.file.wasi_libc_root, ctx.file.builtins_root, ctx.file.builtins_list],
    )

    ctx.actions.run(
        executable = ctx.executable.build_script,
        arguments = [
            clang_bin,
            ctx.file.wasi_libc_root.dirname,
            ctx.file.builtins_root.dirname,
            ctx.file.builtins_list.path,
            ctx.files.shim_srcs[0].dirname,
            out.path,
        ],
        inputs = inputs,
        outputs = [out],
        # clang_tools/resource_dir are selected for the *target* platform's
        # OS, so the action must execute on a host with that OS: a remote
        # Linux worker would receive macOS binaries (and vice versa). Remote
        # caching stays allowed.
        execution_requirements = {"no-remote-exec": "1"},
        mnemonic = "Wasm64Sysroot",
        progress_message = "Building wasm64 wasi-libc sysroot (%s)" % ctx.label.name,
    )
    return [DefaultInfo(files = depset([out]))]

wasm64_sysroot = rule(
    implementation = _wasm64_sysroot_impl,
    attrs = {
        "build_script": attr.label(
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
        "builtins_list": attr.label(allow_single_file = True),
        "builtins_root": attr.label(
            allow_single_file = True,
            doc = "Any file in compiler-rt/lib/builtins; its dirname locates the sources.",
        ),
        "builtins_srcs": attr.label_list(allow_files = True),
        "clang_tools": attr.label_list(
            allow_files = True,
            doc = "clang-22, llvm-ar, llvm-nm from the pinned host toolchain.",
        ),
        "resource_dir": attr.label_list(
            doc = "clang's lib/clang/22 resource directory; every compile reads its freestanding headers.",
        ),
        "shim_srcs": attr.label_list(allow_files = True),
        "wasi_libc_root": attr.label(
            allow_single_file = True,
            doc = "wasi-libc Makefile; its dirname locates the source tree.",
        ),
        "wasi_libc_srcs": attr.label_list(allow_files = True),
    },
)
