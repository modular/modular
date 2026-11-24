// RUN: mo-opt %s --mo-topological-sort | FileCheck %s


mo.graph @sort_no_nested_region<D0, D1 -> out>(
  %input: !mo.tensor<[D0], f32>,
  %dummy: !mo.tensor<[D1], f32> // this is used solely to define `D1`
) -> !mo.tensor<[out, 768], f32> {

  // `%unused` makes a regression test that ensures we don't move `mo.output`.
  %unused = mo.relu(%res) : !mo.tensor<[D3, 768], f32>

  %res = mo.static.broadcast_to(%input_rs) : !mo.tensor<[D2, 768], f32> -> !mo.tensor<[D3, 768], f32>

  kgen.param.declare D3: scalar<si64> = <add(D2, D1)>

  %input_rs = mo.static.reshape(%input) : !mo.tensor<[D0], f32> -> !mo.tensor<[D2, 768], f32>

  kgen.param.declare D2: scalar<si64> = <max(D1, 1)>

  mo.output<D3> %res : !mo.tensor<[D3, 768], f32>
}

// CHECK-LABEL: mo.graph @sort_no_nested_region
// CHECK-SAME:   <[[D0:.*]], [[D1:.*]] -> [[OUT:.*]]>(
// CHECK-SAME:    %[[INPUT:.*]]: !mo.tensor<[[[D0]]], f32>,
// CHECK-SAME:    %[[DUMMY:.*]]: !mo.tensor<[[[D1]]], f32>
// CHECK-SAME:  ) -> !mo.tensor<[[[OUT]], 768], f32> {
//
// NOTE these are all reordered
// CHECK-NEXT:    kgen.param.declare [[D2:.*]]: scalar<si64> = <max([[D1]], 1)>
// CHECK-NEXT:    kgen.param.declare [[D3:.*]]: scalar<si64> = <add([[D1]], [[D2]])>
// CHECK-NEXT:    %[[INPUT_RS:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[[[D0]]], f32> -> !mo.tensor<[[[D2]], 768], f32>
// CHECK-NEXT:    %[[RES:.*]] = mo.static.broadcast_to(%[[INPUT_RS]]) : !mo.tensor<[[[D2]], 768], f32> -> !mo.tensor<[[[D3]], 768], f32>
// CHECK-NEXT:    %[[UNUSED:.*]] = mo.relu(%[[RES]]) : !mo.tensor<[[[D3]], 768], f32>
//
// CHECK-NEXT:    mo.output<[[D3]]> %[[RES]] : !mo.tensor<[[[D3]], 768], f32>
// CHECK-NEXT: }


mo.graph @sort_with_nested_region<D0, D1>(
  %cond: !mo.tensor<[], bool>,
  %scalar: !mo.tensor<[], f32>,
  %dummy: !mo.tensor<[D0, D1], f32> // this is used solely to define `D0` and `D1`
) -> !mo.tensor<[D0], f32> {

  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[D0], f32> {
    %1 = mo.static.broadcast_to(%scalar) : !mo.tensor<[], f32> -> !mo.tensor<[D3], f32>
    kgen.param.declare D5: scalar<si64> = <mul(D1, D4)>
    %2 = mo.static.broadcast_to(%1) : !mo.tensor<[D3], f32> -> !mo.tensor<[D0], f32>
    kgen.param.declare D4: scalar<si64> = <min(D1, 1)>
    mo.yield %2 : !mo.tensor<[D0], f32>

  } else {
    %3 = mo.static.broadcast_to(%scalar) : !mo.tensor<[], f32> -> !mo.tensor<[D0], f32>
    %4 = mo.add(%3, %3) : !mo.tensor<[D0], f32>
    mo.yield %4 : !mo.tensor<[D0], f32>
  }

  kgen.param.declare D3: scalar<si64> = <div(D2, D1)>
  kgen.param.declare D2: scalar<si64> = <add(D1, 1)>

  mo.output %0 : !mo.tensor<[D0], f32>
}

// CHECK-LABEL: mo.graph @sort_with_nested_region
// CHECK-SAME:   <[[D0:.*]], [[D1:.*]]>(
// CHECK-SAME:    %[[COND:.*]]: !mo.tensor<[], bool>,
// CHECK-SAME:    %[[SCALAR:.*]]: !mo.tensor<[], f32>,
// CHECK-SAME:    %[[DUMMY:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>
// CHECK-SAME:  ) -> !mo.tensor<[[[D0]]], f32> {
//
// NOTE these are reordered
// CHECK-NEXT:  kgen.param.declare [[D2:.*]]: scalar<si64> = <add([[D1]], 1)>
// CHECK-NEXT:  kgen.param.declare [[D3:.*]]: scalar<si64> = <div([[D2]], [[D1]])>
//
// CHECK-NEXT:  %[[RES:.*]] = mo.if(%[[COND]] : !mo.tensor<[], bool>) -> !mo.tensor<[[[D0]]], f32> {
// CHECK-NEXT:    %[[V1:.*]] = mo.static.broadcast_to(%[[SCALAR]]) : !mo.tensor<[], f32> -> !mo.tensor<[[[D3]]], f32>
// CHECK-NEXT:    %[[V2:.*]] = mo.static.broadcast_to(%1) : !mo.tensor<[[[D3]]], f32> -> !mo.tensor<[[[D0]]], f32>
// NOTE these are reordered
// CHECK-NEXT:    kgen.param.declare [[D4:.*]]: scalar<si64> = <min([[D1]], 1)>
// CHECK-NEXT:    kgen.param.declare [[D5:.*]]: scalar<si64> = <mul([[D1]], [[D4]])>
//
// CHECK-NEXT:    mo.yield %[[V2]] : !mo.tensor<[[[D0]]], f32>
//
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %[[V3:.*]] = mo.static.broadcast_to(%[[SCALAR]]) : !mo.tensor<[], f32> -> !mo.tensor<[[[D0]]], f32>
// CHECK-NEXT:    %[[V4:.*]] = mo.add(%[[V3]], %[[V3]]) : !mo.tensor<[[[D0]]], f32>
// CHECK-NEXT:    mo.yield %[[V4]] : !mo.tensor<[[[D0]]], f32>
// CHECK-NEXT:  }
//
// CHECK-NEXT:  mo.output %[[RES]] : !mo.tensor<[[[D0]]], f32>
// CHECK-NEXT: }

// CHECK-LABEL: mo.graph @sort_with_nested_non_graph_region
mo.graph @sort_with_nested_non_graph_region<D0, D1>(
  %cond: !mo.tensor<[], bool>,
  %scalar: !mo.tensor<[], f32>,
  %dummy: !mo.tensor<[D0, D1], f32> // this is used solely to define `D0` and `D1`
) -> !mo.tensor<[D0], f32> {

  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[D0], f32> {
    %1 = mo.static.broadcast_to(%scalar) : !mo.tensor<[], f32> -> !mo.tensor<[D3], f32>
    kgen.param.declare D5: scalar<si64> = <mul(D1, D4)>
    %2 = mo.static.broadcast_to(%1) : !mo.tensor<[D3], f32> -> !mo.tensor<[D0], f32>
    kgen.param.declare D4: scalar<si64> = <min(D1, 1)>
    mo.yield %2 : !mo.tensor<[D0], f32>

  } else {
    %3 = mo.static.broadcast_to(%scalar) : !mo.tensor<[], f32> -> !mo.tensor<[D0], f32>
    %4 = mo.add(%3, %3) : !mo.tensor<[D0], f32>
    mo.yield %4 : !mo.tensor<[D0], f32>
  }

  kgen.param.declare D3: scalar<si64> = <div(D2, D1)>
  kgen.param.declare D2: scalar<si64> = <add(D1, 1)>

  mo.output %0 : !mo.tensor<[D0], f32>
}
