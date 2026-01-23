"""Generate MEF files from python Graphs."""

load("//bazel:api.bzl", "modular_py_binary")

# All transitive mojo dependencies of //max:MOGGKernelAPI
MOJO_DEPS = [
    "//max:MOGGKernelAPI",
    "//max:MOGGPrimitives",
    "//max:_cublas",
    "//max:_cudnn",
    "//max:_cufft",
    "//max:_rocblas",
    "//max:buffer",
    "//max:comm",
    "//max:compiler_internal",
    "//max:internal_utils",
    "//max:kv_cache",
    "//max:layout",
    "//max:linalg",
    "//max:nn",
    "//max:quantization",
    "//max:register",
    "//max:shmem",
    "//max:tensor",
    "//max:weights_registry",
    "@mojo//:std",
]

def mef(name, src, args = [], target_compatible_with = [], **kwargs):
    """Generate mef file from a python executable.

    The generated file will be {name}.mef.

    Args:
        name: The target name for the generated file.
        src: .py file generating the MEF file.
        args: Args added to the python file's execution
        target_compatible_with: Constraints for platform execution
        **kwargs: forwarded to the `py_binary` target
    """
    py_binary_name = name + ".py_binary"
    mef_name = name + ".mef"

    modular_py_binary(
        name = py_binary_name,
        srcs = [src],
        main = src,
        target_compatible_with = target_compatible_with,
        mojo_deps = kwargs.get("mojo_deps", []) + MOJO_DEPS,
        **kwargs
    )

    native.genrule(
        name = name,
        outs = [mef_name],
        exec_compatible_with = target_compatible_with,
        target_compatible_with = target_compatible_with,
        cmd = " ".join([
            "MODULAR_HOME=.",
            "MODULAR_MOJO_MAX_IMPORT_PATH=" + ",".join([
                "$$(dirname $(location {}))".format(dep)
                for dep in MOJO_DEPS
            ]),
            "$(location :" + py_binary_name + ")",
            "$(location :" + mef_name + ")",
        ] + args),
        tools = [":" + py_binary_name] + MOJO_DEPS,
    )
