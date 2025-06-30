// RUN: not kgen-opt -outline-closures-new -split-input-file %s 2>&1 | FileCheck %s

// CHECK: error: value is a capture of closure fn but is not in the capture list
kgen.generator @closure_types(%arg0 : index, %arg1 : index) {
  // CHECK: error: value is a capture of closure fn but is not in the capture list
  %0 = index.add %arg0, %arg0
  %3 = kgen.closure.init(%arg1)(%arg2: index) -> index {
    %1 = index.add %0, %arg0
    kgen.return %1 : index
  } : (index), !kgen.pointer<!kgen.closure<@closure_types, "fn" nonescaping>>
  kgen.return
}

// -----

// COM: Ensure that unhandled symbols and types fail the pass.

// CHECK: error: no lifted closure struct found for closure type '!kgen.closure<@invalid, "UNKNOWN" nonescaping>'
// CHECK: error: no lifted closure method found for closure symbol #kgen.closure.symbol<@invalid, "UNKNOWN", #kgen.closure_method<call>>
#type_value = #kgen.type<!kgen.closure<@invalid, "UNKNOWN" nonescaping>,
              {"__call__" :
              (!kgen.pointer<!kgen.closure<@invalid, "UNKNOWN" nonescaping>> read_mem, index) -> index =
               #kgen.closure.symbol<@invalid, "UNKNOWN", #kgen.closure_method<call>>}> : !kgen.type

kgen.generator @consume<x: type>(%arg0: !kgen.pointer<x> read_mem) -> index {
    %0 = kgen.call_param[(!kgen.pointer<x> read_mem) -> index: get_vtable_entry(x, "__call__")](%arg0)
    kgen.return %0 : index
}


kgen.generator @invalid() {
  %3 = kgen.closure.init()(%arg2: index) -> index {
    kgen.return %arg2 : index
  } : (), !kgen.pointer<!kgen.closure<@invalid, "fn" nonescaping>>
  %2 = kgen.call @consume<:type #type_value>(%3) : (!kgen.pointer<!kgen.closure<@invalid, "fn" nonescaping>> read_mem) -> index
  kgen.return
}
