// RUN: mo-opt %s --allow-unregistered-dialect

mo.graph @opaque_type() {
  %0 = "my_op"() : () -> !mo.opaque<"q4_0">
  mo.output
}

// -----

mo.graph @opaque_type_parameter() {
  %0 = "my_op"() : () -> !mo.opaque<"q4_0", {some_data = 8 : index}>
  mo.output
}

// -----

mo.graph @opaque_in_out(%arg0: !mo.opaque<"MyInt">) -> !mo.opaque<"MyInt"> {
  %0 = mo.custom [symbol = "my_int_square", device = #M.device_ref<"cpu", 0>](%arg0) : (!mo.opaque<"MyInt">) -> !mo.opaque<"MyInt">
  mo.output %0 : !mo.opaque<"MyInt">
}
