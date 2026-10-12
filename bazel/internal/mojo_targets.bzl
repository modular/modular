"""Reads the targets the graph compiler needs off the mojo toolchain.

A rule that compiles a graph without the accelerator attached has to say which
accelerator, and which host CPU, to compile for. The toolchain already
describes the lane being built for, and the artifacts have to agree with it, so
a call site names neither.

Reading the toolchain is all this does. Whether a lane that names no
accelerator is a misuse is the rule's own policy, and the rules differ: one is
always gated to a GPU lane, while another has call sites that a CPU lane
analyzes and only a GPU lane ever runs.
"""

def mojo_targets_from_toolchain(ctx):
    """Returns the accelerator and host-CPU targets this lane compiles for.

    Args:
        ctx: The rule context. The rule must declare
            ``toolchains = ["@rules_mojo//:toolchain_type"]``.

    Returns:
        A struct with ``accelerator`` (``"api:arch"``, e.g. ``"cuda:sm_100a"``,
        or ``""`` on a lane whose toolchain names none) and ``cpu_target`` (the
        host-CPU codegen descriptor), as MAX's virtual-device knobs spell them.
    """
    copts = ctx.toolchains["@rules_mojo//:toolchain_type"].mojo_toolchain_info.copts

    accelerator = ""
    cpu = None
    triple = None
    for copt in copts:
        if copt.startswith("--target-accelerator="):
            accelerator = copt.removeprefix("--target-accelerator=")
        elif copt.startswith("--target-cpu="):
            cpu = copt.removeprefix("--target-cpu=")
        elif copt.startswith("-target-triple=") or copt.startswith("--target-triple="):
            triple = copt.split("=", 1)[1]

    # Mojo and the graph compiler spell the vendor differently.
    if accelerator:
        accelerator = accelerator.replace("nvidia", "cuda")
        accelerator = accelerator.replace("amdgpu", "hip")

    # The triple matters as much as the CPU: it is part of the compile target,
    # and the artifact's host code has to match the machine that runs it.
    if triple and cpu:
        cpu_target = "triple={};cpu={}".format(triple, cpu)
    else:
        cpu_target = cpu or ""
    return struct(accelerator = accelerator, cpu_target = cpu_target)
