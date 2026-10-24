// RUN: kgen-opt -allow-unregistered-dialect -lower-arg-conventions -verify-diagnostics %s

// expected-error @below {{cannot unpack argument 1 that has fnArgAttrs}}
kgen.func @lower_args_with_arg_metadata(
 %arg0: !kgen.pointer<i8> imm_mem,
 %arg1: !kgen.pointer<!kgen.struct<() isParamPack>> imm_mem,
 %arg2: !kgen.pointer<i32> imm_mem
) -> !kgen.pointer<i8> attributes {
  fnArgAttrs = [{nvvm.grid_constant = unit}, {nvvm.grid_constant = unit}, {}]
} {
    hlcf.return %arg0 : !kgen.pointer<i8>
}
