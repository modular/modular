// RUN: mo-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect


// -----

//
// mgp.tensor_spec.create{.dynamic}
//

func.func @spec_create_mismatched_dims(%dim : index) -> !mgp.tensor_spec<?x64x?xf32> {
  // expected-error@+1 {{'mgp.tensor_spec.create' op expected 2 dimension operands to match '!mgp.tensor_spec<?x64x?xf32>', but given 1.}}
  %0 = mgp.tensor_spec.create %dim : !mgp.tensor_spec<?x64x?xf32>
  mef.output %0 : !mgp.tensor_spec<?x64x?xf32>
}

// -----

func.func @spec_create_bogus_type(%dim : index) -> !mgp.tensor_spec<?x64x?xf32> {
  // expected-error@+1 {{custom op 'mgp.tensor_spec.create' expected a !mgp.tensor_spec type.}}
  %0 = mgp.tensor_spec.create %dim : !mgp.tensor<?x64xf32>
  mef.output %0 : !mgp.tensor_spec<?x64x?xf32>
}

// -----

func.func @spec_create_unknown_elem_type(%dim : index) -> !mgp.tensor_spec<?x64xunk> {
  // expected-error@+1 {{custom op 'mgp.tensor_spec.create' expected !mgp.tensor_spec type to have a known element type.}}
  %0 = mgp.tensor_spec.create %dim : !mgp.tensor_spec<?x64xunk>
  mef.output %0 : !mgp.tensor_spec<?x64xunk>
}

// -----

//
// mgp.tensor_spec.get_dim
//

func.func @spec_get_dim_oob(%spec : !mgp.tensor_spec<?x64x?xf32>) -> index {
  // expected-error@+1 {{'mgp.tensor_spec.get_dim' op expected axis 4 to be less than the rank of the tensor spec 3}}
  %dim0 = mgp.tensor_spec.get_dim[4] %spec : !mgp.tensor_spec<?x64x?xf32>
  mef.output %dim0 : index
}

// -----

//
// !mgp.tensor_spec
//

// expected-error@+1 {{tensor spec must have a rank between 0 and 8.}}
func.func @spec_too_large(%x : !mgp.tensor_spec<1x2x3x4x5x6x7x8x9xf32>) {}

// -----

// expected-error@+1 {{unrecognized dtype 'ui33'.}}
func.func @dtype_unsupported(%x : !mgp.tensor_spec<5x10xui33>) {}

// -----

// expected-error@+1 {{invalid dtype.}}
func.func @dtype_invalid(%x : !mgp.tensor_spec<5x10xinvalid>) {}

// -----

//
// mgp.model
//

// expected-error@+1 {{'mgp.model' op init block arguments must be !mgp.opaque or !mgp.device_context, but got '!mef.chain'}}
"mgp.model"() ({
  ^bb0(%weights: !mgp.opaque<"weights_registry">, %ch: !mef.chain):
    %ctx = mgp.context.create 1
    mef.output %ch, %ctx : !mef.chain, !mgp.context
}, {
  ^bb0(%ch: !mef.chain, %ctx: !mgp.context, %x: tensor<3xf32>):
    mef.output %ch, %x : !mef.chain, tensor<3xf32>
}) {frameworkName = "tf", devices = [#M.device_ref<"cpu", 0>], inputOutputTypes = (tensor<3xf32>) -> (tensor<3xf32>), sym_name = "badSetupArgs"} : () -> ()

// -----

"mgp.model"() ({
  ^bb0(%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>):
    %ch = grt.chain.create ()
    %ctx = mgp.context.create 1
    // expected-error@+1 {{'mef.output' op expected 4 operands, but given 1}}
    mef.output %ch : !mef.chain
}, {
  ^bb0(%ch: !mef.chain, %ctx: !mgp.context, %x: tensor<3xf32>):
    mef.output %ch, %x : !mef.chain, tensor<3xf32>
}) {frameworkName = "tf", devices = [#M.device_ref<"cpu", 0>], inputOutputTypes = (tensor<3xf32>) -> (tensor<3xf32>), sym_name = "badSetupResults"} : () -> ()

// -----

// expected-error@+1 {{'mgp.model' op expected the arguments of the 'execute' body to be '!mef.chain', '!mgp.context', '!mgp.opaque<"weights_registry">', '!mgp.device_context<<"cpu", 0>>', 'tensor<3xf32>', but got '!mef.chain', 'tensor<3xf32>', '!mgp.context'.}}
"mgp.model"() ({
  ^bb0(%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>):
    %ch = grt.chain.create ()
    %ctx = mgp.context.create 1
    mef.output %ch, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
}, {
  ^bb0(%ch: !mef.chain, %x: tensor<3xf32>, %ctx: !mgp.context):
    mef.output %ch, %x : !mef.chain, tensor<3xf32>
}) {frameworkName = "tf", devices = [#M.device_ref<"cpu", 0>], inputOutputTypes = (tensor<3xf32>) -> (tensor<3xf32>), sym_name = "badExecuteArgs"} : () -> ()

// -----

// expected-error@+1 {{'mgp.model' op expected the arguments of the 'execute' body to be '!mef.chain', '!mgp.context', '!mgp.opaque<"weights_registry">', '!mgp.device_context<<"cpu", 0>>', 'tensor<4xf32>', but got '!mef.chain', '!mgp.context', '!mgp.device_context<<"cpu", 0>>', 'tensor<3xf32>'.}}
"mgp.model"() ({
  ^bb0(%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>):
    %ch = grt.chain.create ()
    %ctx = mgp.context.create 1
    mef.output %ch, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
}, {
  ^bb0(%ch: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.device_context<<"cpu", 0>>, %x: tensor<3xf32>):
    mef.output %ch, %x : !mef.chain, tensor<3xf32>
}) {frameworkName = "tf", devices = [#M.device_ref<"cpu", 0>], inputOutputTypes = (tensor<4xf32>) -> (tensor<3xf32>), sym_name = "badExecuteArgs2"} : () -> ()

// -----

"mgp.model"() ({
  ^bb0(%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>):
    %ch = grt.chain.create ()
    %ctx = mgp.context.create 1
    mef.output %ch, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
}, {
  ^bb0(%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %x: tensor<3xf32>):
    // expected-error@+1 {{'mef.output' op expected 1 operands, but given 2}}
    mef.output %ch, %x : !mef.chain, tensor<3xf32>
}) {frameworkName = "tf", devices = [#M.device_ref<"cpu", 0>], inputOutputTypes = (tensor<3xf32>) -> (), sym_name = "badExecuteResults"} : () -> ()

// -----

mgp.model @mismatchedContextType : tf devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch = grt.chain.create ()
  %unk = mef.opaque "bogus" [] () : ()->!mef.unknown
  %ctx = mgp.context.create 1
  // expected-error@+1 {{'mef.output' op operand #1 has type '!mef.unknown' but expected '!mgp.context'}}
  mef.output %ch, %unk, %weights, %cpu : !mef.chain, !mef.unknown, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %x: tensor<2xf32>) -> (!mef.chain, tensor<2xf32>) {
  mef.output %ch, %x : !mef.chain, tensor<2xf32>
}


// -----

// expected-error@+1 {{'mgp.model' op attribute 'frameworkName' failed to satisfy constraint: string attribute must be a recognized framework name.}}
mgp.model @invalidExecuteArgType : bogus devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %x: index) -> (!mef.chain, index) {
  mef.output %ch, %x : !mef.chain, index
}

// -----

//
// mgp.guard
//

mgp.model @guardOpWrongNResults : bogus devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %tensor: !mgp.tensor<2xsi64>) -> (!mef.chain, !mgp.tensor<2xsi64>) {

  // expected-error@+1 {{operation defines 1 results but was provided 3 to bind}}
  %results:3 = mgp.guard %ch, %tensor : !mgp.tensor<2xsi64>

  mef.output %ch, %tensor : !mef.chain, !mgp.tensor<2xsi64>
}

// -----

//
// mgp.buffer.constant.external
//
func.func @buffer_constant_external(%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">) -> !mgp.buffer {
  %name = mgp.string.create "foo" : !mgp.opaque<"mojo_string">
  // expected-error@+1 {{invalid alignment 10}}
  %buf = mgp.buffer.constant.external %ch, %weights[%name, 16384, align 10] : !mgp.opaque<"weights_registry">, !mgp.opaque<"mojo_string"> -> !mgp.buffer
  mef.output %buf : !mgp.buffer
}

// -----

//
// mgp.tensor.slice
//
func.func @tensor_slice_device_context_mismatch(%input: !mgp.tensor<5xf32, <"gpu", 0>>, %spec: !mgp.tensor_spec<3xf32>, %start: !mgp.tensor<1xsi64>, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mgp.tensor<3xf32, <"gpu", 0>> {
  // expected-error@+1 {{'mgp.tensor.slice' op device context must be on the same device as the input}}
  %sliced = mgp.tensor.slice %input, %spec, %start, %cpu : !mgp.tensor<5xf32, <"gpu", 0>>, !mgp.tensor_spec<3xf32>, !mgp.tensor<1xsi64>, !mgp.device_context<<"cpu", 0>>
  mef.output %sliced : !mgp.tensor<3xf32, <"gpu", 0>>
}

// -----

//
// mgp.buffer.slice
//
func.func @buffer_slice_device_context_mismatch(%buf: !mgp.buffer<<"gpu", 0>>, %offset: index, %size: index, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mgp.buffer<<"gpu", 0>> {
  // expected-error@+1 {{'mgp.buffer.slice' op device context must be on the same device as the input buffer}}
  %sliced = mgp.buffer.slice %buf, %offset, %size, %cpu : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"cpu", 0>>
  mef.output %sliced : !mgp.buffer<<"gpu", 0>>
}

// -----

//
// mgp.buffer.bulk_slice
//
func.func @bulk_slice_device_context_mismatch(%base: !mgp.buffer<<"gpu", 0>>, %offset: index, %size: index, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mgp.buffer<<"gpu", 0>> {
  // expected-error@+1 {{'mgp.buffer.bulk_slice' op device context must be on the same device as the base buffer}}
  %sliced = mgp.buffer.bulk_slice %base, %cpu offsets(%offset) sizes(%size) : !mgp.buffer<<"gpu", 0>>, !mgp.device_context<<"cpu", 0>>
  mef.output %sliced : !mgp.buffer<<"gpu", 0>>
}

// -----

//
// mgp.call
//

// subgraph model for call tests
mgp.model @call_target_model subgraph : tf devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %in_tensor: !mgp.tensor<10xf32>, %idx: index) -> (!mef.chain, !mgp.tensor<10xf32>) {
  mef.output %ch, %in_tensor : !mef.chain, !mgp.tensor<10xf32>
}

func.func @call_operand_type_mismatch(%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %input : !mgp.tensor<20xf64>) -> (!mef.chain, !mgp.tensor<10xf32>) {
                                                                                                                                                                   // ^^^^^^^^^^^^^^^^^^^ Type mismatch
  // expected-error@+1 {{'mgp.call' op operand types '!mef.chain', '!mgp.context', '!mgp.opaque<"weights_registry">', '!mgp.device_context<<"cpu", 0>>', '!mgp.tensor<20xf64>' do not match expected argument types '!mef.chain', '!mgp.context', '!mgp.opaque<"weights_registry">', '!mgp.device_context<<"cpu", 0>>', '!mgp.tensor<10xf32>', 'index' of @call_target_model's execute block.}}
  %ch_res, %res = mgp.call @call_target_model(%ch, %ctx, %weights, %cpu, %input) : (!mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.tensor<20xf64>) -> (!mef.chain, !mgp.tensor<10xf32>)
  mef.output %ch_res, %res : !mef.chain, !mgp.tensor<10xf32>
}

// -----

mgp.model @call_target_model subgraph : tf devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %in_tensor: !mgp.tensor<10xf32>, %idx: index) -> (!mef.chain, !mgp.tensor<10xf32>) {
  mef.output %ch, %in_tensor : !mef.chain, !mgp.tensor<10xf32>
}

func.func @call_result_type_mismatch(%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %input : !mgp.tensor<10xf32>) -> (!mef.chain, !mgp.tensor<5xsi32>) {
  // expected-error@+1 {{'mgp.call' op result types '!mef.chain', '!mgp.tensor<5xsi32>' do not match expected result types '!mef.chain', '!mgp.tensor<10xf32>' of @call_target_model's execute block.}}
  %ch_res, %res = mgp.call @call_target_model(%ch, %ctx, %weights, %cpu, %input) : (!mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.tensor<10xf32>) -> (!mef.chain, !mgp.tensor<5xsi32>)
                                                                                                                                                                                                                  // ^^^^^^^^^^^^^^^^^^^ Type mismatch
  mef.output %ch_res, %res : !mef.chain, !mgp.tensor<5xsi32>
}

// -----

func.func @call_symbol_not_found(%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %input : !mgp.tensor<10xf32>) -> (!mef.chain, !mgp.tensor<10xf32>) {
  // expected-error@+1 {{'mgp.call' op failed to find symbol @non_existent_model}}
  %ch_res, %res = mgp.call @non_existent_model(%ch, %ctx, %weights, %cpu, %input) : (!mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.tensor<10xf32>) -> (!mef.chain, !mgp.tensor<10xf32>)
  mef.output %ch_res, %res : !mef.chain, !mgp.tensor<10xf32>
}


// -----

// expected-error@+1 {{expected exactly one mgp.context_create op in the 'init' block, but found 0}}
mgp.model @zero_context_create_ops : tf devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch = grt.chain.create ()
  // Missing mgp.context.create
  mef.output %ch, %weights, %cpu : !mef.chain, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %x: tensor<2xf32>) -> (!mef.chain, tensor<2xf32>) {
  mef.output %ch, %x : !mef.chain, tensor<2xf32>
}

// -----

// expected-error@+1 {{expected exactly one mgp.context_create op in the 'init' block, but found 2}}
mgp.model @multiple_context_create_ops : tf devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch = grt.chain.create ()
  %ctx1 = mgp.context.create 1
  %ctx2 = mgp.context.create 1 // Second context create
  mef.output %ch, %ctx1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %x: tensor<2xf32>) -> (!mef.chain, tensor<2xf32>) {
  mef.output %ch, %x : !mef.chain, tensor<2xf32>
}

// -----

//
// mgp.tensor.create with mismatched `mut`
//

func.func @tensor_create_missing_mut_attr(%ch: !mef.chain, %buf: !mgp.buffer<<"gpu", 0>>, %spec: !mgp.tensor_spec<3x5xf32>) -> !mgp.tensor<3x5xf32, <"gpu", 0>, mut> {
  // expected-error@+1 {{expected output tensor type does not match actual output tensor type}}
  %0 = mgp.tensor.create %ch, %buf, %spec : !mgp.tensor<3x5xf32, <"gpu", 0>, mut>
  mef.output %0 : !mgp.tensor<3x5xf32, <"gpu", 0>, mut>
}

// -----

func.func @tensor_create_spurious_mut_attr(%ch: !mef.chain, %buf: !mgp.buffer<<"gpu", 0>>, %spec: !mgp.tensor_spec<3x5xf32>) -> !mgp.tensor<3x5xf32, <"gpu", 0>> {
  // expected-error@+1 {{expected output tensor type does not match actual output tensor type}}
  %0 = mgp.tensor.create %ch, %buf, %spec mut : !mgp.tensor<3x5xf32, <"gpu", 0>>
  mef.output %0 : !mgp.tensor<3x5xf32, <"gpu", 0>>
}
