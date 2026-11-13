// RUN: kgen-opt %s -lower-kgen-to-llvm | kgen-translate -mlir-to-llvmir | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// ===-------------------------------------------------------------------=== //
// Unit-valued LLVMFuncOp attributes.
// ===-------------------------------------------------------------------=== //

// CHECK-LABEL: define{{.*}} void @fn_always_inline()
// CHECK-SAME: #[[AI:[0-9]+]]
kgen.func export @fn_always_inline() { hlcf.return } <fnAttrs = {llvm.always_inline = unit}>

// CHECK-LABEL: define{{.*}} void @fn_no_inline()
// CHECK-SAME: #[[NI:[0-9]+]]
kgen.func export @fn_no_inline() { hlcf.return } <fnAttrs = {llvm.no_inline = unit}>

// `optimize_none` must coexist with `no_inline`.
// CHECK-LABEL: define{{.*}} void @fn_optimize_none()
// CHECK-SAME: #[[ON:[0-9]+]]
kgen.func export @fn_optimize_none() { hlcf.return } <fnAttrs = {llvm.optimize_none = unit, llvm.no_inline = unit}>

// CHECK-LABEL: define{{.*}} void @fn_no_unwind()
// CHECK-SAME: #[[NU:[0-9]+]]
kgen.func export @fn_no_unwind() { hlcf.return } <fnAttrs = {llvm.no_unwind = unit}>

// CHECK-LABEL: define{{.*}} void @fn_will_return()
// CHECK-SAME: #[[WR:[0-9]+]]
kgen.func export @fn_will_return() { hlcf.return } <fnAttrs = {llvm.will_return = unit}>

// CHECK-LABEL: define{{.*}} void @fn_convergent()
// CHECK-SAME: #[[CV:[0-9]+]]
kgen.func export @fn_convergent() { hlcf.return } <fnAttrs = {llvm.convergent = unit}>

// CHECK-LABEL: define{{.*}} void @fn_cold()
// CHECK-SAME: #[[CO:[0-9]+]]
kgen.func export @fn_cold() { hlcf.return } <fnAttrs = {llvm.cold = unit}>

// CHECK-LABEL: define{{.*}} void @fn_hot()
// CHECK-SAME: #[[HT:[0-9]+]]
kgen.func export @fn_hot() { hlcf.return } <fnAttrs = {llvm.hot = unit}>

// CHECK-LABEL: define{{.*}} void @fn_inline_hint()
// CHECK-SAME: #[[IH:[0-9]+]]
kgen.func export @fn_inline_hint() { hlcf.return } <fnAttrs = {llvm.inline_hint = unit}>

// OptionalAttr<UnitAttr> properties on LLVMFuncOp.
// CHECK-LABEL: define{{.*}} void @fn_optsize()
// CHECK-SAME: #[[OS:[0-9]+]]
kgen.func export @fn_optsize() { hlcf.return } <fnAttrs = {llvm.optsize = unit}>

// CHECK-LABEL: define{{.*}} void @fn_minsize()
// CHECK-SAME: #[[MS:[0-9]+]]
kgen.func export @fn_minsize() { hlcf.return } <fnAttrs = {llvm.minsize = unit}>

// CHECK-LABEL: define{{.*}} void @fn_save_reg_params()
// CHECK-SAME: #[[SRP:[0-9]+]]
kgen.func export @fn_save_reg_params() { hlcf.return } <fnAttrs = {llvm.save_reg_params = unit}>

// ===-------------------------------------------------------------------=== //
// String-valued KV-passthrough attributes.
// ===-------------------------------------------------------------------=== //

// CHECK-LABEL: define{{.*}} void @fn_denorm_fp()
// CHECK-SAME: #[[DFP:[0-9]+]]
kgen.func export @fn_denorm_fp() { hlcf.return } <fnAttrs = {llvm.denormal_fp_math = "preserve-sign,preserve-sign"}>

// CHECK-LABEL: define{{.*}} void @fn_denorm_fp32()
// CHECK-SAME: #[[DFP32:[0-9]+]]
kgen.func export @fn_denorm_fp32() { hlcf.return } <fnAttrs = {llvm.denormal_fp_math_f32 = "ieee,ieee"}>

// CHECK-LABEL: define{{.*}} void @fn_fp_contract()
// CHECK-SAME: #[[FPC:[0-9]+]]
kgen.func export @fn_fp_contract() { hlcf.return } <fnAttrs = {llvm.fp_contract = "fast"}>

// String/bool/integer-valued attrs whose LLVM IR names use dashes.
// CHECK-LABEL: define{{.*}} void @fn_probe_stack()
// CHECK-SAME: #[[PS:[0-9]+]]
kgen.func export @fn_probe_stack() { hlcf.return } <fnAttrs = {llvm.probe_stack = "__chkstk"}>

// CHECK-LABEL: define{{.*}} void @fn_no_inline_line_tables()
// CHECK-SAME: #[[NILT:[0-9]+]]
kgen.func export @fn_no_inline_line_tables() { hlcf.return } <fnAttrs = {llvm.no_inline_line_tables = true}>

// CHECK-LABEL: define{{.*}} void @fn_no_stack_arg_probe()
// CHECK-SAME: #[[NSAP:[0-9]+]]
kgen.func export @fn_no_stack_arg_probe() { hlcf.return } <fnAttrs = {llvm.no_stack_arg_probe = true}>

// Integer values are stringified as decimal.
// CHECK-LABEL: define{{.*}} void @fn_stack_probe_size()
// CHECK-SAME: #[[SPS:[0-9]+]]
kgen.func export @fn_stack_probe_size() { hlcf.return } <fnAttrs = {llvm.stack_probe_size = 4096 : i64}>

// CHECK-LABEL: define{{.*}} void @fn_warn_stack_size()
// CHECK-SAME: #[[WSS:[0-9]+]]
kgen.func export @fn_warn_stack_size() { hlcf.return } <fnAttrs = {llvm.warn_stack_size = 1024 : i64}>

// ===-------------------------------------------------------------------=== //
// Boolean-valued fast-math KV-passthrough attributes (all five together).
// ===-------------------------------------------------------------------=== //

// CHECK-LABEL: define{{.*}} void @fn_fast_math()
// CHECK-SAME: #[[FM:[0-9]+]]
kgen.func export @fn_fast_math() { hlcf.return } <fnAttrs = {
    llvm.unsafe_fp_math = true,
    llvm.no_infs_fp_math = true,
    llvm.no_nans_fp_math = true,
    llvm.approx_func_fp_math = true,
    llvm.no_signed_zeros_fp_math = true
  }>

// ===-------------------------------------------------------------------=== //
// Composite typed attributes.
// ===-------------------------------------------------------------------=== //

// CHECK-LABEL: define{{.*}} void @fn_frame_pointer()
// CHECK-SAME: #[[FP:[0-9]+]]
kgen.func export @fn_frame_pointer() { hlcf.return } <fnAttrs = {llvm.frame_pointer = "all"}>

// CHECK-LABEL: define{{.*}} void @fn_vscale_range()
// CHECK-SAME: #[[VS:[0-9]+]]
kgen.func export @fn_vscale_range() { hlcf.return } <fnAttrs = {llvm.vscale_range = #pop.array<1, 16> : !pop.array<2, i32>}>

// ===-------------------------------------------------------------------=== //
// Integer-valued typed attributes.
// ===-------------------------------------------------------------------=== //

// `alignment` becomes `align N` after the attribute id.
// CHECK-LABEL: define{{.*}} void @fn_alignment()
// CHECK-SAME: align 32
kgen.func export @fn_alignment() { hlcf.return } <fnAttrs = {llvm.alignment = 32 : index}>

// `function_entry_count` lands as a !prof annotation, not in `attributes #N`.
// CHECK-LABEL: define{{.*}} void @fn_entry_count()
// CHECK-SAME: !prof
kgen.func export @fn_entry_count() { hlcf.return } <fnAttrs = {llvm.function_entry_count = 1234 : index}>

// ===-------------------------------------------------------------------=== //
// Unknown `llvm.*` names fall through to generic passthrough.
// ===-------------------------------------------------------------------=== //

// CHECK-LABEL: define{{.*}} void @fn_passthrough()
// CHECK-SAME: #[[PT:[0-9]+]]
kgen.func export @fn_passthrough() { hlcf.return } <fnAttrs = {llvm.noredzone = unit}>

// LLVM auto-promotes known parametric attribute names from KV passthrough.
// CHECK-LABEL: define{{.*}} void @fn_alignstack()
// CHECK-SAME: #[[AS:[0-9]+]]
kgen.func export @fn_alignstack() { hlcf.return } <fnAttrs = {llvm.alignstack = 16 : i64}>

// CHECK-LABEL: define{{.*}} void @fn_memory()
// CHECK-SAME: #[[MEM:[0-9]+]]
kgen.func export @fn_memory() { hlcf.return } <fnAttrs = {llvm.memory = 1 : i64}>


// CHECK-DAG: attributes #[[AI]] = { alwaysinline {{.*}} }
// CHECK-DAG: attributes #[[NI]] = { noinline {{.*}} }
// CHECK-DAG: attributes #[[ON]] = { noinline optnone {{.*}} }
// CHECK-DAG: attributes #[[NU]] = { nounwind {{.*}} }
// CHECK-DAG: attributes #[[WR]] = { willreturn {{.*}} }
// CHECK-DAG: attributes #[[CV]] = { convergent {{.*}} }
// CHECK-DAG: attributes #[[CO]] = { cold {{.*}} }
// CHECK-DAG: attributes #[[HT]] = { hot {{.*}} }
// CHECK-DAG: attributes #[[IH]] = { inlinehint {{.*}} }
// CHECK-DAG: attributes #[[OS]] = { optsize {{.*}} }
// CHECK-DAG: attributes #[[MS]] = { minsize {{.*}} }
// CHECK-DAG: attributes #[[SRP]] = { {{.*}}"save-reg-params"{{.*}} }
// CHECK-DAG: attributes #[[DFP]] = { {{.*}}"denormal-fp-math"="preserve-sign,preserve-sign"{{.*}} }
// CHECK-DAG: attributes #[[DFP32]] = { {{.*}}"denormal-fp-math-f32"="ieee,ieee"{{.*}} }
// CHECK-DAG: attributes #[[FPC]] = { {{.*}}"fp-contract"="fast"{{.*}} }
// CHECK-DAG: attributes #[[PS]] = { {{.*}}"probe-stack"="__chkstk"{{.*}} }
// CHECK-DAG: attributes #[[NILT]] = { {{.*}}"no-inline-line-tables"="true"{{.*}} }
// CHECK-DAG: attributes #[[NSAP]] = { {{.*}}"no-stack-arg-probe"="true"{{.*}} }
// CHECK-DAG: attributes #[[SPS]] = { {{.*}}"stack-probe-size"="4096"{{.*}} }
// CHECK-DAG: attributes #[[WSS]] = { {{.*}}"warn-stack-size"="1024"{{.*}} }
// CHECK-DAG: attributes #[[FM]] = { {{.*}}"approx-func-fp-math"="true"{{.*}}"no-infs-fp-math"="true"{{.*}}"no-nans-fp-math"="true"{{.*}}"no-signed-zeros-fp-math"="true"{{.*}}"unsafe-fp-math"="true"{{.*}} }
// CHECK-DAG: attributes #[[FP]] = { {{.*}}"frame-pointer"="all"{{.*}} }
// CHECK-DAG: attributes #[[VS]] = { {{.*}}vscale_range(1,16){{.*}} }
// CHECK-DAG: attributes #[[PT]] = { noredzone {{.*}} }
// CHECK-DAG: attributes #[[AS]] = { alignstack=16 {{.*}} }
// CHECK-DAG: attributes #[[MEM]] = { {{.*}}memory(argmem: read){{.*}} }

}
