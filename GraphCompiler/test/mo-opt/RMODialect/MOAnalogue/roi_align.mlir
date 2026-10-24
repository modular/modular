// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @roi_align_ok<N, C, H, W, R -> O0, O1, O2, O3>(%input: !mo.tensor<[N, C, H, W], f32>, %rois: !mo.tensor<[R, 5], f32>, %output_height: !mo.tensor<[], si64>, %output_width: !mo.tensor<[], si32>, %spatial_scale: !mo.tensor<[], f32>, %sampling_ratio: !mo.tensor<[], f32>) -> !mo.tensor<[O0, O1, O2, O3], f32> {
  %0 = rmo.mo.roi_align<() -> P0, P1, P2, P3>(%input, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio) [aligned = false, mode = "AVG"] : (!mo.tensor<[N, C, H, W], f32>, !mo.tensor<[R, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[P0, P1, P2, P3], f32>
  mo.output<P0, P1, P2, P3> %0 : !mo.tensor<[P0, P1, P2, P3], f32>
}

// -----

mo.graph @roi_align_bad_height<N, C, H, W, R>(%input: !mo.tensor<[N, C, H, W], f32>, %rois: !mo.tensor<[R, 5], f32>, %output_height: !mo.tensor<[], f32>, %output_width: !mo.tensor<[], si32>, %spatial_scale: !mo.tensor<[], f32>, %sampling_ratio: !mo.tensor<[], f32>) -> !mo.tensor<[N, C, H, W], f32> {
  // expected-error @+1 {{'rmo.mo.roi_align' op operand #2 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<[], f32>'}}
  %0 = rmo.mo.roi_align(%input, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio) [aligned = false, mode = "AVG"] : (!mo.tensor<[N, C, H, W], f32>, !mo.tensor<[R, 5], f32>, !mo.tensor<[], f32>, !mo.tensor<[], si32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[N, C, H, W], f32>
  mo.output %0 : !mo.tensor<[N, C, H, W], f32>
}

// -----

mo.graph @roi_align_bad_scale<N, C, H, W, R>(%input: !mo.tensor<[N, C, H, W], f32>, %rois: !mo.tensor<[R, 5], f32>, %output_height: !mo.tensor<[], si64>, %output_width: !mo.tensor<[], si32>, %spatial_scale: !mo.tensor<[], si32>, %sampling_ratio: !mo.tensor<[], f32>) -> !mo.tensor<[N, C, H, W], f32> {
  // expected-error @+1 {{'rmo.mo.roi_align' op operand #4 must be rank 0 or unranked with float elements, but got '!mo.tensor<[], si32>'}}
  %0 = rmo.mo.roi_align(%input, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio) [aligned = false, mode = "AVG"] : (!mo.tensor<[N, C, H, W], f32>, !mo.tensor<[R, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], f32>) -> !mo.tensor<[N, C, H, W], f32>
  mo.output %0 : !mo.tensor<[N, C, H, W], f32>
}
