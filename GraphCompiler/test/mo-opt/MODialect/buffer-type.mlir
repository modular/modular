// RUN: mo-opt %s --split-input-file

mo.graph @buffer_type_simple(%arg0: !mo.buffer<[1, 2, 3], f32>) {
  mo.output
}

// -----

mo.graph @buffer_type_no_device(%arg0: !mo.buffer<[1, 2, 3], f32, { foo = "bar" }>) {
  mo.output
}

// -----

mo.graph @buffer_type_no_metadata(%arg0: !mo.buffer<[1, 2, 3], f32>) {
  mo.output
}

// -----

mo.graph @buffer_type_full(%arg0: !mo.buffer<[1, 2, 3], f32, { foo = "bar" }>) {
  mo.output
}
