// RUN: mo-opt --mo-flatten-elementwise --cse --canonicalize %s | FileCheck %s --check-prefix=CHECK-MO

!tt = !mo.tensor<[1,2,3,4,5], f32>
!tta = !mo.tensor<[6,4,5], f32>
!ttb = !mo.tensor<[2,12,5,1,1], f32>
!ttc = !mo.tensor<[120], f32>
!ttd = !mo.tensor<[6,20], f32>
!tte = !mo.tensor<[1,2,12,5], f32>

// CHECK-MO-LABEL: mo.graph @binary_elemwise_with_parameter
// CHECK-MO: mo.relu({{.*}}) : !mo.tensor<[mul_no_wrap(P1, 20)], f32>
// CHECK-MO: mo.add({{.*}}) : !mo.tensor<[mul_no_wrap(P1, 20)], f32>
mo.graph @binary_elemwise_with_parameter<P1>(%arg0 : !mo.tensor<[P1,4,5], f32>, %arg1 : !mo.tensor<[2,2,P1,5], f32>) -> !mo.tensor<[P1,4,5], f32> {
    %0 = mo.relu(%arg0) : !mo.tensor<[P1,4,5], f32>
    %1 = mo.static.reshape(%0) : !mo.tensor<[P1,4,5], f32> -> !mo.tensor<[2,2,P1,5], f32>
    %2 = mo.add(%1, %arg1) : !mo.tensor<[2,2,P1,5], f32>
    %3 = mo.static.reshape(%2) : !mo.tensor<[2,2,P1,5], f32> -> !mo.tensor<[P1,4,5], f32>
    mo.output %3 : !mo.tensor<[P1,4,5], f32>
}

// CHECK-MO-LABEL: mo.graph @with_non_elemwise
// CHECK-MO-NOT: mo.relu({{.*}}) : !mo.tensor<[{{.*}},{.*}}], f32>
// CHECK-MO-NOT: mo.add({{.*}}) : !mo.tensor<[{{.*}},{.*}}], f32>
mo.graph @with_non_elemwise(%arg0 : !tta, %arg1 : !ttb) -> !tta {
    %0 = mo.relu(%arg0) : !tta
    %1 = mo.static.reshape(%0) : !tta -> !ttb
    %2 = mo.add(%1, %arg1) : !ttb
    %3 = mo.static.reshape(%2) : !ttb -> !tta

    %4 = mo.reduce.softmax(%3) [axis = 2] : (!tta) -> !tta

    %5 = mo.relu(%4) : !tta
    %6 = mo.static.reshape(%5) : !tta -> !ttb
    %7 = mo.add(%6, %arg1) : !ttb
    %8 = mo.static.reshape(%7) : !ttb -> !tta

    mo.output %8 : !tta
}

// CHECK-MO-LABEL: mo.graph @many_reshapes
// CHECK-MO-NOT: mo.relu({{.*}}) : !mo.tensor<[{{.*}},{.*}}], f32>
// CHECK-MO-NOT: mo.add({{.*}}) : !mo.tensor<[{{.*}},{.*}}], f32>
mo.graph @many_reshapes(%arg1 : !tt, %arg2 : !tt) -> !tt {
    %a1 = mo.static.reshape(%arg1) : !tt -> !tta
    %a2 = mo.static.reshape(%arg2) : !tt -> !tta

    %b1 = mo.static.reshape(%arg1) : !tt -> !ttb
    %b2 = mo.static.reshape(%arg2) : !tt -> !ttb

    %c1 = mo.static.reshape(%arg1) : !tt -> !ttc
    %c2 = mo.static.reshape(%arg2) : !tt -> !ttc

    %a_add = mo.add(%a1, %a2) : !tta
    %b_add = mo.add(%b1, %b2) : !ttb
    %c_add = mo.add(%c1, %c2) : !ttc

    %d1 = mo.static.reshape(%a_add) : !tta -> !ttd
    %d2 = mo.static.reshape(%b_add) : !ttb -> !ttd
    %d3 = mo.static.reshape(%c_add) : !ttc -> !ttd

    %t0 = mo.static.reshape(%d3) : !ttd -> !tt

    %d_add_1 = mo.add(%d1, %d2) : !ttd
    %d_add_2 = mo.add(%d_add_1, %d3) : !ttd

    %e1 = mo.static.reshape(%d_add_1) : !ttd -> !tte
    %e2 = mo.static.reshape(%d_add_2) : !ttd -> !tte
    %e3 = mo.static.reshape(%b2) : !ttb -> !tte
    %e4 = mo.static.reshape(%t0) : !tt -> !tte
    %e5 = mo.static.reshape(%c2) : !ttc -> !tte

    %e_relu_1 = mo.relu(%e1) : !tte
    %e_add_1 = mo.add(%e1, %e2) : !tte
    %e_add_2 = mo.add(%e3, %e4) : !tte
    %e_add_3 = mo.add(%e2, %e5) : !tte

    %e_add_4 = mo.add(%e_relu_1, %e_add_1) : !tte
    %e_add_5 = mo.add(%e_add_4, %e_add_1) : !tte
    %e_add_6 = mo.add(%e_add_5, %e_add_2) : !tte
    %e_add_7 = mo.add(%e_add_6, %e_add_3) : !tte

    %out = mo.static.reshape(%e_add_7) : !tte -> !tt

    mo.output %out : !tt
}

// CHECK-MO-LABEL: mo.graph @with_cast
// CHECK-MO: mo.cast({{.*}}) : (!mo.tensor<[24], f32>) -> !mo.tensor<[24], f16>
// CHECK-MO: mo.relu({{.*}}) : !mo.tensor<[24], f16>
// CHECK-MO: mo.cast({{.*}}) : (!mo.tensor<[24], f16>) -> !mo.tensor<[24], f64>
mo.graph @with_cast(%arg0 : !mo.tensor<[1,2,3,4], f32>) -> !mo.tensor<[2,3,4], f64> {
    %0 = mo.cast(%arg0) : (!mo.tensor<[1,2,3,4], f32>) -> !mo.tensor<[1,2,3,4], f16>
    %1 = mo.relu(%0) : !mo.tensor<[1,2,3,4], f16>
    %2 = mo.static.reshape(%1) : !mo.tensor<[1,2,3,4], f16> -> !mo.tensor<[2,3,4], f16>
    %3 = mo.cast(%2) : (!mo.tensor<[2,3,4], f16>) -> !mo.tensor<[2,3,4], f64>
    mo.output %3 : !mo.tensor<[2,3,4], f64>
}

// CHECK-MO-LABEL: mo.graph @with_pow
// CHECK-MO: mo.pow({{.*}} : !mo.tensor<[6], f32>, {{.*}} : !mo.tensor<[6], si32>) : !mo.tensor<[6], f32>
// CHECK-MO: mo.relu({{.*}}) : !mo.tensor<[6], f32>
mo.graph @with_pow(%arg0 : !mo.tensor<[3,2], f32>, %arg1 : !mo.tensor<[3,2], si32>) -> !mo.tensor<[3,2], f32> {
    %0 = mo.pow(%arg0 : !mo.tensor<[3,2], f32>, %arg1 : !mo.tensor<[3,2], si32>) : !mo.tensor<[3,2], f32>
    %1 = mo.static.reshape(%0) : !mo.tensor<[3,2], f32> -> !mo.tensor<[3,2,1], f32>
    %2 = mo.relu(%1) : !mo.tensor<[3,2,1], f32>
    %3 = mo.static.reshape(%2) : !mo.tensor<[3,2,1], f32> -> !mo.tensor<[3,2], f32>
    mo.output %3 : !mo.tensor<[3,2], f32>
}
