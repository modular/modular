//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPH_COMPILER_MOGGSUPPORT_MOGGTENSORACCESSOR_H
#define GENERICML_GRAPH_COMPILER_MOGGSUPPORT_MOGGTENSORACCESSOR_H

#include "KGEN/KGENDialect/KGENAttrs.h"

namespace M::MOGG {

// Mirror of the tensor attributes as they exist in Mojo. This allows us to
// manipulate parameters on calls as we can understand which parameter
// corresponds to which in the tensor when passing them into a call.
struct MOGGTensorParamAccessor {
  MOGGTensorParamAccessor() { params.resize(NUM_PARAMS); }

  explicit MOGGTensorParamAccessor(KGEN::DeclRefType decl) {
    params.resize(NUM_PARAMS);
    params[DTYPE_IDX] = decl.getParamValues()[DTYPE_IDX];
    params[SHAPE_IDX] = decl.getParamValues()[SHAPE_IDX];
    params[STRIDE_IDX] = decl.getParamValues()[STRIDE_IDX];
    params[INPUT_LAMBDA_IDX] = decl.getParamValues()[INPUT_LAMBDA_IDX];
    params[OUTPUT_LAMBDA_IDX] = decl.getParamValues()[OUTPUT_LAMBDA_IDX];
    params[OWNED_MEMORY_IDX] = decl.getParamValues()[OWNED_MEMORY_IDX];
  }

  SmallVector<TypedAttr> params;

  void assignParam(TypedAttr param, size_t index) { params[index] = param; }

  bool isParamDefaulted(size_t index) const {
    return !isa<KGEN::ParamIndexRefAttr>(params[index]);
  }

  KGEN::ParamIndexRefAttr paramAsRef(size_t index) const {
    return dyn_cast<KGEN::ParamIndexRefAttr>(params[index]);
  }

  KGEN::ParamIndexRefAttr dtypeAsRef() const { return paramAsRef(DTYPE_IDX); }

  KGEN::ParamIndexRefAttr shapeAsRef() const { return paramAsRef(SHAPE_IDX); }
  TypedAttr shape() const { return params[SHAPE_IDX]; }

  KGEN::ParamIndexRefAttr strideAsRef() const { return paramAsRef(STRIDE_IDX); }
  TypedAttr strides() const { return params[STRIDE_IDX]; }

  KGEN::ParamIndexRefAttr inputLambdaAsRef() const {
    return paramAsRef(INPUT_LAMBDA_IDX);
  }

  KGEN::ParamIndexRefAttr outputLambdaAsRef() const {
    return paramAsRef(OUTPUT_LAMBDA_IDX);
  }

  KGEN::ParamIndexRefAttr ownedMemoryAsRef() const {
    return paramAsRef(OWNED_MEMORY_IDX);
  }

  // The indices of each parameter as they appear on the struct.
  static constexpr size_t DTYPE_IDX = 0;
  static constexpr size_t SHAPE_IDX = 1;
  static constexpr size_t STRIDE_IDX = 2;
  static constexpr size_t INPUT_LAMBDA_IDX = 3;
  static constexpr size_t OUTPUT_LAMBDA_IDX = 4;
  static constexpr size_t OWNED_MEMORY_IDX = 5;
  static constexpr size_t NUM_PARAMS = 6;
};

} // namespace M::MOGG

#endif // GENERICML_GRAPH_COMPILER_MOGGSUPPORT_MOGGTENSORACCESSOR_H
