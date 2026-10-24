// RUN: mo-opt --mo-symbolic-optimize %s | FileCheck %s --enable-var-scope

// If a reshape feeding into a broadcast is simply padding dimensions to the LHS of the shape, the reshape is extraneous
// CHECK-LABEL: @simple_broadcasts
// CHECK: mo.static.broadcast
// CHECK-SAME: !mo.tensor<[A, B], f32> -> !mo.tensor<[1, 2, 3, A, B], f32>
mo.graph @simple_broadcasts<A, B>(%input: !mo.tensor<[A, B], f32>) -> !mo.tensor<[1, 2, 3, A, B], f32>{
    %reshape = mo.static.reshape(%input) : !mo.tensor<[A, B], f32> -> !mo.tensor<[1, 1, 1, A, B], f32>
    %out = mo.static.broadcast_to(%reshape) : !mo.tensor<[1, 1, 1, A, B], f32> -> !mo.tensor<[1, 2, 3, A, B], f32>
    mo.output %out: !mo.tensor<[1, 2, 3, A, B], f32>
}

// Here the broadcast also pads some more rank, but the optimization should still fire
// CHECK-LABEL: @simple_broadcast_more_padding
// CHECK: mo.static.broadcast
// CHECK-SAME: !mo.tensor<[A, B], f32> -> !mo.tensor<[1, 2, 3, A, B], f32>
mo.graph @simple_broadcast_more_padding<A, B>(%input: !mo.tensor<[A, B], f32>) -> !mo.tensor<[1, 2, 3, A, B], f32>{
    // Only pads 1 dim, bcast pads 2 more
    %reshape = mo.static.reshape(%input) : !mo.tensor<[A, B], f32> -> !mo.tensor<[1, A, B], f32>
    %out = mo.static.broadcast_to(%reshape) : !mo.tensor<[ 1, A, B], f32> -> !mo.tensor<[1, 2, 3, A, B], f32>
    mo.output %out: !mo.tensor<[1, 2, 3, A, B], f32>
}

// CHECK-LABEL: @fails_reshape_not_expanding_rank
// CHECK: mo.static.reshape
// CHECK: mo.static.broadcast
mo.graph @fails_reshape_not_expanding_rank(%input: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[1, 2, 3, 5, 20], f32>{
    // Note rank is the same after reshape
    %reshape = mo.static.reshape(%input) : !mo.tensor<[10, 10], f32> -> !mo.tensor<[5, 20], f32>
    %out = mo.static.broadcast_to(%reshape) : !mo.tensor<[5, 20], f32> -> !mo.tensor<[1, 2, 3, 5, 20], f32>
    mo.output %out: !mo.tensor<[1, 2, 3, 5, 20], f32>
}

// CHECK-LABEL: @fails_reshape_not_just_padding
// CHECK: mo.static.reshape
// CHECK: mo.static.broadcast
mo.graph @fails_reshape_not_just_padding(%input: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[1, 2, 3, 5, 20], f32>{
    // Note padding 1's, but also transforming the indexing in the tensor
    %reshape = mo.static.reshape(%input) : !mo.tensor<[10, 10], f32> -> !mo.tensor<[1, 1, 1, 5, 20], f32>
    %out = mo.static.broadcast_to(%reshape) : !mo.tensor<[1, 1, 1, 5, 20], f32> -> !mo.tensor<[1, 2, 3, 5, 20], f32>
    mo.output %out: !mo.tensor<[1, 2, 3, 5, 20], f32>
}

// CHECK-LABEL: @fails_reshape_not_padding
// CHECK: mo.static.reshape
// CHECK: mo.static.broadcast
mo.graph @fails_reshape_not_padding(%input: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[1, 2, 3, 5, 10], f32>{
    // Note not all padded dimensions are 1's
    %reshape = mo.static.reshape(%input) : !mo.tensor<[10, 10], f32> -> !mo.tensor<[1, 2, 1, 5, 10], f32>
    %out = mo.static.broadcast_to(%reshape) :  !mo.tensor<[1, 2, 1, 5, 10], f32> -> !mo.tensor<[1, 2, 3, 5, 10], f32>
    mo.output %out: !mo.tensor<[1, 2, 3, 5, 10], f32>
}
