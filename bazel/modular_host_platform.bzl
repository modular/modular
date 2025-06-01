"""Setup a host platform that takes into account current GPU hardware"""

def _verbose_log(rctx, msg):
    if rctx.getenv("MODULAR_VERBOSE_GPU_DETECT"):
        # buildifier: disable=print
        print(msg)

def _log_result(rctx, binary, result):
    _verbose_log(
        rctx,
        "\n------ {}:\nexit status: {}\nstdout: {}\nstderr: {}\n------ end gpu-query info"
            .format(binary, result.return_code, result.stdout, result.stderr),
    )

def _get_amd_constraint(blob):
    for value in blob.values():
        series = value["Card Series"]
        if "MI300X" in series:
            return "@//:mi300x_gpu"
        if "AMD Radeon Graphics" in series:
            return "@//:radeon_gpu"

    fail("Unrecognized rocm-smi output, please report: {}".format(blob))

def _get_nvidia_constraint(lines):
    line = lines[0]
    if " A10G" in line:
        return "@//:a10_gpu"
    if " A100-" in line:
        return "@//:a100_gpu"
    if " H100 " in line:
        return "@//:h100_gpu"
    if " H200" in line:
        return "@//:h200_gpu"
    if " L4" in line or " Ada " in line:
        return "@//:l4_gpu"
    if " A3000 " in line:
        return "@//:a3000_gpu"

    # The Blackwell architecture (B100 and B200) is an SM100 GPU.
    if "B100" in line:
        return "@//:b100_gpu"
    if "B200" in line:
        return "@//:b200_gpu"

    # The RTX 5090 is an SM120 GPU.
    if " RTX 5090" in line:
        return "@//:rtx5090_gpu"

    if "Laptop GPU" in line:
        return None

    if "RTX 4070 Ti" in line:
        return None

    if "RTX 4080 SUPER" in line:
        return None

    if "NVIDIA GeForce RTX 3090" in line:
        return None

    fail("Unrecognized nvidia-smi output, please report: {}".format(lines))

def _impl(rctx):
    constraints = []
    if rctx.os.name == "linux" and rctx.os.arch == "amd64":
        nvidia_smi = rctx.which("nvidia-smi")
        rocm_smi = rctx.which("rocm-smi")
        _verbose_log(rctx, "nvidia-smi path: {}, rocm-smi path: {}".format(nvidia_smi, rocm_smi))

        if nvidia_smi:
            result = rctx.execute([nvidia_smi, "--query-gpu=gpu_name", "--format=csv,noheader"])
            _log_result(rctx, nvidia_smi, result)
            if result.return_code == 0:
                lines = result.stdout.splitlines()
                if len(lines) == 0:
                    fail("nvidia-smi succeeded but had no GPUs, please report this issue")

                constraint = _get_nvidia_constraint(lines)
                if constraint:
                    constraints.extend([
                        "@//:nvidia_gpu",
                        "@//:has_gpu",
                        constraint,
                    ])

                if len(lines) > 1:
                    constraints.append("@//:has_multi_gpu")
                if len(lines) >= 4:
                    constraints.append("@//:has_4_gpus")

        elif rocm_smi:
            result = rctx.execute([rocm_smi, "--json", "--showproductname"])
            _log_result(rctx, rocm_smi, result)

            if result.return_code == 0:
                constraints.extend([
                    "@//:amd_gpu",
                    "@//:has_gpu",
                ])

                blob = json.decode(result.stdout)
                if len(blob.keys()) == 0:
                    fail("rocm-smi succeeded but didn't actually have any GPUs, please report this issue")

                constraints.append(_get_amd_constraint(blob))
                if len(blob.keys()) > 1:
                    constraints.append("@//:has_multi_gpu")
                if len(blob.keys()) >= 4:
                    constraints.append("@//:has_4_gpus")

    rctx.file("WORKSPACE.bazel", "workspace(name = {})".format(rctx.attr.name))
    rctx.file("BUILD.bazel", """
platform(
    name = "modular_host_platform",
    parents = ["@local_config_platform//:host"],
    visibility = ["//visibility:public"],
    constraint_values = [{constraints}],
    exec_properties = {{
        "no-remote-exec": "1",
    }},
)
""".format(constraints = ", ".join(['"{}"'.format(x) for x in constraints])))

modular_host_platform = repository_rule(
    implementation = _impl,
    configure = True,
    environ = [
        "MODULAR_VERBOSE_GPU_DETECT",
    ],
)
