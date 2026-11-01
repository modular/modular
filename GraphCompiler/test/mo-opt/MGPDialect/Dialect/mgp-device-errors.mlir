// RUN: mo-opt %s --verify-diagnostics --allow-unregistered-dialect

//
// mgp.buffer.constant
//

func.func @buffer_constant_invalid_elem_type(%ch: !mef.chain, %ctx : !mgp.context) {
  // expected-error@+1 {{'mgp.buffer.constant' op only supports attributes of uint8_t elements (have 'i16').}}
  %0 = "mgp.buffer.constant" () { value = #M.dense_array<1, 2, 3, 4> : !M.array<4xi16>, rawAlign = 32 : ui64 }
         : () -> !mgp.buffer
  mef.output %0 : !mgp.buffer
}
