# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug --target-accelerator=nvidia:90 %s | FileCheck %s


from gpu.grid_controls import _ENABLE_PDL_LAUNCH as ENABLE_PDL_LAUNCH
from gpu.grid_controls import (
    PDL,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from gpu.host._compile import _compile_code_asm, _get_gpu_target
from testing import assert_true


fn control_dep_grids_kernel():
    wait_on_dependent_grids()
    launch_dependent_grids()


# CHECK-LABEL: test_grid_control_primitives
# CHECK: griddepcontrol.wait
# CHECK: griddepcontrol.launch_dependents
# CHECK: griddepcontrol.wait
# CHECK: griddepcontrol.launch_dependents
def test_grid_control_primitives():
    print("== test_grid_control_primitives")
    assert_true(ENABLE_PDL_LAUNCH)
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


fn control_dep_grids_kernel_context():
    with PDL():
        pass


# CHECK-LABEL: test_grid_control_primitives_context
# CHECK: griddepcontrol.wait
# CHECK: griddepcontrol.launch_dependents
def test_grid_control_primitives_context():
    print("== test_grid_control_primitives_context")
    print(
        _compile_code_asm[
            control_dep_grids_kernel_context,
            emission_kind="asm",
            target = _get_gpu_target["sm_90a"](),
        ]()
    )


def main():
    test_grid_control_primitives()
    test_grid_control_primitives_context()
