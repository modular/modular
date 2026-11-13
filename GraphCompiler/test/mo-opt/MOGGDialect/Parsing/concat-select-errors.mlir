// RUN: mo-opt -split-input-file -verify-diagnostics %s

// mogg.index.concat.select yields exactly one value.

func.func @noResult(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %y: !mogg.tensor) {
  mogg.call.foreach @builtin_primitives.primitives._foreach shape(%shape) context(%ctx) (%i : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %cy = %y : !mogg.tensor} device = "cpu" attributes {max_simd_width = 4 : i64} {
    // expected-error @below{{expected '->'}}
    mogg.index.concat.select(%i) <axis = 1, sizes = #mosh<ape[4, 4]> : !mosh.ape> : !map.index<2> {
    ^bb0(%a: !map.index<2>):
      mogg.yield
    }, {
    ^bb0(%b: !map.index<2>):
      mogg.yield
    }
  }
  return
}

