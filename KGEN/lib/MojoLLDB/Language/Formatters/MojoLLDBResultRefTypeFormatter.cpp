//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoLLDBResultRefTypeFormatter.h"
#include "MojoWrappingTypeSyntheticFrontEnd.h"

using namespace lldb;
using namespace lldb_private;
using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::Mojo;

SyntheticChildrenFrontEnd *
M::KGEN::Mojo::MojoREPLResultRefTypeSyntheticFrontEndCreator(
    CXXSyntheticChildren *, const ValueObjectSP &valobjSP) {
  if (!valobjSP)
    return nullptr;
  return new MojoWrappingTypeSyntheticFrontEnd(*valobjSP, {0, 0});
}
