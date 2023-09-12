//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements common utilities shared by the parser implementation.
//
//===----------------------------------------------------------------------===//

#include "Utils.h"

#include "KGEN/POPDialect/POPTypes.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

POP::PackType LIT::getIfPackType(SignatureType sig, size_t index) {
  return sig.isPackVararg(index)
             ? ::cast<POP::PackType>(sig.getValueInputs()[index])
             : nullptr;
}
