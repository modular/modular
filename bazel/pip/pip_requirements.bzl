"""Create a wrapper repo around pip requirements to automatically handle differing requirements for GPUs."""

load("@module_versions//:config.bzl", "TORCH_DEFAULT_VERSION")

def _get_requirements(rctx, path):
    lines = rctx.read(path).splitlines()

    deps = set()
    direct_dep = False
    requirement = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(("#", "-")):
            if line.endswith(".in"):
                direct_dep = True
        else:
            if direct_dep:
                deps.add(requirement)

            direct_dep = False

            # requirement @ URL
            if " @ " in line:
                requirement, _ = line.split(" @ ", 1)
            else:
                requirement, _ = line.split("=", 1)

            # lm-eval[api] -> lm-eval
            requirement = requirement.split("[", 1)[0]

    return deps

def _impl(rctx):
    amd_deps = set()
    for src in rctx.attr.amd_gpu_requirements:
        amd_deps |= _get_requirements(rctx, src)
    nvidia_deps = set()
    for src in rctx.attr.nvidia_gpu_requirements:
        nvidia_deps |= _get_requirements(rctx, src)
    cpu_deps = set()
    for src in rctx.attr.requirements:
        cpu_deps |= _get_requirements(rctx, src)

    shared_deps = []
    gpu_only_deps = []
    amd_only_deps = []
    nvidia_only_deps = []

    shared_deps = cpu_deps & (amd_deps | nvidia_deps)
    gpu_only_deps = (amd_deps & nvidia_deps) - cpu_deps
    amd_only_deps = amd_deps - nvidia_deps
    nvidia_only_deps = nvidia_deps - amd_deps

    is_x86_64 = rctx.os.arch in ("amd64", "x86_64")
    lines = ['load("@pip_torch-{}_deps//:requirements.bzl", cpu_requirement = "requirement")'.format(TORCH_DEFAULT_VERSION)]
    if is_x86_64:
        lines.append('load("@modular_amd_gpu_pip_deps//:requirements.bzl", amd_gpu_requirement = "requirement")')
        lines.append('load("@modular_nvidia_gpu_pip_deps//:requirements.bzl", nvidia_gpu_requirement = "requirement")')

    lines.append('package(default_visibility = ["//visibility:public"])')
    lines.append('filegroup(name = "incompatible", srcs = [], target_compatible_with = ["@platforms//:incompatible"])')

    for dep in sorted(amd_only_deps):
        if is_x86_64:
            lines.append("""
alias(
    name = "{name}",
    actual = select({{
        "@//:amd_gpu": amd_gpu_requirement("{name}"),
    }})
)
""".format(name = dep))
        else:
            lines.append('alias(name = "{name}", actual = ":incompatible")'.format(name = dep))

    for dep in sorted(nvidia_only_deps):
        if is_x86_64:
            lines.append("""
alias(
    name = "{name}",
    actual = select({{
        "@//:nvidia_gpu": nvidia_gpu_requirement("{name}"),
    }})
)
""".format(name = dep))
        else:
            lines.append('alias(name = "{name}", actual = ":incompatible")'.format(name = dep))

    for dep in sorted(gpu_only_deps):
        if is_x86_64:
            lines.append("""
alias(
    name = "{name}",
    actual = select({{
        "@//:amd_gpu": amd_gpu_requirement("{name}"),
        "@//:nvidia_gpu": nvidia_gpu_requirement("{name}"),
    }})
)
""".format(name = dep))
        else:
            lines.append('alias(name = "{name}", actual = ":incompatible")'.format(name = dep))

    for dep in sorted(shared_deps):
        if is_x86_64:
            lines.append("""
alias(
    name = "{name}",
    actual = select({{
        "@//:amd_gpu": amd_gpu_requirement("{name}"),
        "@//:nvidia_gpu": nvidia_gpu_requirement("{name}"),
        "//conditions:default": cpu_requirement("{name}")
    }})
)
""".format(name = dep))
        else:
            lines.append('alias(name = "{name}", actual = cpu_requirement("{name}"))'.format(name = dep))

    rctx.file("BUILD.bazel", content = "\n".join(lines))

pip_requirements = repository_rule(
    implementation = _impl,
    attrs = {
        "amd_gpu_requirements": attr.label_list(allow_files = [".txt"]),
        "nvidia_gpu_requirements": attr.label_list(allow_files = [".txt"]),
        "requirements": attr.label_list(allow_files = [".txt"]),
    },
)
