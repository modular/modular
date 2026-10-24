// RUN: mo-opt --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

!tt  = !mo.tensor<[5, 5], f32>
!stt = !mo.tensor<[d0, d1], f32>
!bt  = !mo.buffer<[5, 5], f32>
!st = !mo.tensor<[2], si64>
!c   = !mo.chain

// CHECK-LABEL: mo.graph @slice_non_staticizable
// CHECK-SAME: <[[d0:.*]], [[d1:.*]]>([[buf:%.*]]: !mo.buffer<{{.*}}>, [[slice:%.*]]: !mo.tensor<[[[d0]], [[d1]]], f32>,
// CHECK-SAME: [[step:%.*]]: !mo.tensor<[2], si64>
// CHECK-DAG: [[start:%.*]] = mo.constant{{.*}}#M.dense_array<0, 0>
// CHECK-DAG: [[stop:%.*]] = mo.constant{{.*}}#M.dense_array<5, 5>
// CHECK-DAG: [[ch0:%.*]] = mo.chain.create([[buf]]
// CHECK: %{{.*}}, [[ch1:%.*]] = mo.mutable.load({{.*}}, [[ch0]]
// CHECK: [[ch2:%.*]] = mo.mutable.store({{.*}}, [[ch1]]
// CHECK: mo.mutable.store.slice([[buf]], [[slice]], [[start]], [[stop]], [[step]], [[ch2]])
mo.graph @slice_non_staticizable<d0, d1>(%buf: !mo.buffer<[5, 5], f32>, %slice: !mo.tensor<[d0, d1], f32>, %step: !mo.tensor<[2], si64>) -> !bt
{
  %ch0 = mo.chain.create(%buf) : !bt

  %start = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop  = mo.constant <#M.dense_array<5, 5> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %vbuf, %ch1 = mo.mutable.load(%buf, %ch0) : (!bt, !c) -> (!tt, !c)
  %neg = mo.negative(%vbuf) : !tt
  %ch2 = mo.mutable.store(%buf, %neg, %ch1) : (!bt, !tt, !c) -> !c

  rmo.mo.mutable.store.slice(%buf, %slice, %start, %stop, %step, %ch2) : (!bt, !stt, !st, !st, !st, !c) -> !c

  mo.output %buf : !bt
}

// CHECK-LABEL: mo.graph @slice_staticizable
// CHECK-SAME: <[[d0:.*]], [[d1:.*]]>([[buf:%.*]]: !mo.buffer<{{.*}}>, [[slice:%.*]]: !mo.tensor<[[[d0]], [[d1]]], f32>
// CHECK-DAG: [[start:%.*]] = mo.constant{{.*}}#M.dense_array<0, 0>
// CHECK-DAG: [[stop:%.*]] = mo.constant{{.*}}#M.dense_array<5, 5>
// CHECK-DAG: [[step:%.*]] = mo.constant{{.*}}#M.dense_array<2, 2>
// CHECK-DAG: [[ch0:%.*]] = mo.chain.create([[buf]]
// CHECK: %{{.*}}, [[ch1:%.*]] = mo.mutable.load({{.*}}, [[ch0]]
// CHECK: [[ch2:%.*]] = mo.mutable.store({{.*}}, [[ch1]]
// CHECK: mo.mutable.store.slice([[buf]], [[slice]], [[start]], [[stop]], [[step]], [[ch2]])
mo.graph @slice_staticizable<d0, d1>(%buf: !mo.buffer<[5, 5], f32>, %slice: !mo.tensor<[d0, d1], f32>) -> !bt
{
  %ch0 = mo.chain.create(%buf) : !bt

  %start = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop  = mo.constant <#M.dense_array<5, 5> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %step  = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %vbuf, %ch1 = mo.mutable.load(%buf, %ch0) : (!bt, !c) -> (!tt, !c)
  %neg = mo.negative(%vbuf) : !tt
  %ch2 = mo.mutable.store(%buf, %neg, %ch1) : (!bt, !tt, !c) -> !c

  rmo.mo.mutable.store.slice(%buf, %slice, %start, %stop, %step, %ch2) : (!bt, !stt, !st, !st, !st, !c) -> !c

  mo.output %buf : !bt
}
