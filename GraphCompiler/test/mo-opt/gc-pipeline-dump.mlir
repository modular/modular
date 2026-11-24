// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_MAX_TEMPS_DIR=%t mo-opt %s --gc-pipeline-pre-jit -o /dev/null
// RUN: FileCheck %s --check-prefix=PREMAP --input-file=%t/add+mul.mo-pre-map.mlir
// RUN: FileCheck %s --check-prefix=MGP --input-file=%t/add+mul.mgp.mlir
// RUN: FileCheck %s --check-prefix=MOJO --input-file=%t/add+mul.mojo

// With MODULAR_MAX_TEMPS_DIR set, --gc-pipeline-pre-jit dumps each phase
// boundary, including an intermediate one the final -o output cannot show.
// The two top-level graphs share one stem joined by "+" (add+mul), matching the
// <names>.mo.mlir filename pipeline_to_mo_graph.py dumps, so a graph's stages
// land beside its source instead of clobbering.
// Lowering correctness is covered by MODialect/gc-pipeline-pre-jit.mlir.
// Related to MXTOOLS-226.
mo.graph @add(%a: !mo.tensor<[4], f32>, %b: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) {
  %r = mo.add(%a, %b) : !mo.tensor<[4], f32>
  mo.output %r : !mo.tensor<[4], f32>
}

mo.graph @mul(%a: !mo.tensor<[4], f32>, %b: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>) {
  %r = mo.mul(%a, %b) : !mo.tensor<[4], f32>
  mo.output %r : !mo.tensor<[4], f32>
}

// The mo-pre-map boundary is still in the mo dialect (the mgp dump has
// mgp.model instead). Both graphs are present in the shared dump.
// PREMAP-DAG: mo.graph @add
// PREMAP-DAG: mo.graph @mul

// MGP-DAG: mgp.model @add
// MGP-DAG: mgp.model @mul

// emit-mojo lowered the add to its elementwise kernel call.
// MOJO: builtin_kernels.elementwise.Add
