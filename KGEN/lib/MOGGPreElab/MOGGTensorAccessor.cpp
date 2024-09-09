//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MOGGPreElab/MOGGTensorAccessor.h"
#include "KGEN/KGENDialect/KGENOps.h"

using namespace M;
using namespace MOGG;
using namespace KGEN;

std::optional<size_t> MOGG::getIndexOfParam(Operation *gen, TypedAttr attr) {
  if (auto ref = dyn_cast_or_null<KGEN::ParamIndexRefAttr>(attr)) {
    return ref.getIndex();
  }

  if (auto ref = dyn_cast_or_null<KGEN::ParamDeclRefAttr>(attr)) {
    for (const auto &[idx, param] :
         llvm::enumerate(cast<GeneratorOp>(gen).getInputParams())) {
      if (ref.getName() == param.getName())
        return idx;
    }
  }
  return {};
}
