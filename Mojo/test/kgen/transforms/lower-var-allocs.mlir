// RUN: kgen-opt -lower-var-allocs %s | FileCheck %s --check-prefix=HOST
// RUN: kgen-opt -lower-var-allocs="shared-address-space=3 stack-address-space=5" %s | FileCheck %s --check-prefix=GPU

// Without shared memory, every variable gets a stack slot in its address
// space. With shared memory, a variable in the shared memory address space is
// a global, and one in the stack address space is a default stack slot cast to
// it. A variable in the default address space is a default stack slot on every
// target.
// HOST-LABEL: kgen.func @vars
// GPU-LABEL: kgen.func @vars
kgen.func @vars() {
  // HOST-NEXT: pop.stack_allocation 1 x index address_space 3 align 16
  // GPU-NEXT: pop.global_alloc "tile" 1 x index address_space 3 align 16 <memoryType = gpu_shared>
  %0 = pop.var_alloc "tile" index address_space 3 align 16

  // HOST-NEXT: pop.stack_allocation 1 x index address_space 5
  // GPU-NEXT: %[[SLOT:.*]] = pop.stack_allocation 1 x index
  // GPU-NOT: address_space
  // GPU-NEXT: pop.pointer.bitcast %[[SLOT]] : !kgen.pointer<index> to !kgen.pointer<index, 5>
  %1 = pop.var_alloc "scratch" index address_space 5

  // HOST-NEXT: pop.stack_allocation 1 x index align 8
  // GPU-NEXT: pop.stack_allocation 1 x index align 8
  %2 = pop.var_alloc "local" index address_space 0 align 8

  // HOST-NOT: pop.var_alloc
  // GPU-NOT: pop.var_alloc
  hlcf.return
}
