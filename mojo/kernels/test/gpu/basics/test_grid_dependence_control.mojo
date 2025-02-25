# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s


from gpu.host._compile import _compile_code_asm, _get_gpu_target
from gpu.grid_controls import wait_on_dependent_grids, launch_dependent_grids


fn control_dep_grids_kernel():
    launch_dependent_grids()
    wait_on_dependent_grids()


# CHECK: griddepcontrol.launch_dependents
# CHECK: griddepcontrol.wait
# CHECK: griddepcontrol.launch_dependents
# CHECK: griddepcontrol.wait


def test_grid_control_primitives():
    print(
        _compile_code_asm[
            control_dep_grids_kernel,
            emission_kind="asm",
            target = _get_gpu_target["sm_90"](),
        ]()
    )
    print(
        _compile_code_asm[
            control_dep_grids_kernel,
            emission_kind="asm",
            target = _get_gpu_target["sm_90a"](),
        ]()
    )


def main():
    test_grid_control_primitives()
