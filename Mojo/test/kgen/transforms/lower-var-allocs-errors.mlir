// RUN: kgen-opt -lower-var-allocs="shared-address-space=3 stack-address-space=5" -verify-diagnostics %s

// With shared memory, only the stack and shared memory address spaces can hold
// a variable.
kgen.func @constant() {
  // expected-error @below {{variable 'table' in address space 4 is not supported on this target}}
  %0 = pop.var_alloc "table" index address_space 4
  hlcf.return
}
