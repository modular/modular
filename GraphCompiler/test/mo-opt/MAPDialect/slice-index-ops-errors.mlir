// RUN: mo-opt %s --split-input-file --verify-diagnostics

// SliceSpecAttr's signature must have the same rank on both sides -- slice
// never changes rank (unlike a future view such as reshape, which is why
// this check lives here and not on the generic ViewSignatureAttr).
// expected-error @below{{in/out shapes must have the same rank, got 2 and 1}}
#spec = #map.index.slice.spec<[10, 64] -> [3], starts = [1, 32], stops = [4, 64], steps = [1, 1]>

// -----

// starts/stops/steps must each have one entry per dimension.
// expected-error @below{{starts/stops/steps must each have 2 entries (one per dimension), got 1/2/2}}
#spec = #map.index.slice.spec<[10, 64] -> [3, 32], starts = [1], stops = [4, 64], steps = [1, 1]>

// -----

// stop must not exceed the in-shape dim.
// expected-error @below{{dim 0: start/stop out of bounds for in-shape dim 10, got start=1, stop=11}}
#spec = #map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [11, 64], steps = [1, 1]>

// -----

// start must not exceed stop.
// expected-error @below{{dim 0: start/stop out of bounds for in-shape dim 10, got start=5, stop=4}}
#spec = #map.index.slice.spec<[10, 64] -> [3, 32], starts = [5, 32], stops = [4, 64], steps = [1, 1]>

// -----

// step must be consistent with (stop - start) and the out-shape dim.
// expected-error @below{{dim 0: start/stop/step do not produce out-shape dim 3, got start=1, stop=4, step=2}}
#spec = #map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [2, 1]>

// -----

// step must be non-zero -- a zero step walks nowhere in either direction.
// expected-error @below{{dim 0: step must be non-zero}}
#spec = #map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [0, 1]>

// -----

// A reverse slice runs start down to stop, so the two swap roles.
// expected-error @below{{dim 0: start/stop out of bounds for in-shape dim 10, got start=3, stop=9}}
#spec = #map.index.slice.spec<[10, 64] -> [3, 32], starts = [3, 32], stops = [9, 64], steps = [-2, 1]>

// -----

// A reverse walk must stay in bounds: 9 - 2*5 is -1, off the front of the
// axis. `stops` is not load-bearing here (see the verifier), so the out-shape
// dim is what the walk is checked against.
// expected-error @below{{dim 0: start/step walk out of bounds for in-shape dim 10, got start=9, step=-5, out-shape dim 3}}
#spec = #map.index.slice.spec<[10, 64] -> [3, 32], starts = [9, 32], stops = [-1, 64], steps = [-5, 1]>
