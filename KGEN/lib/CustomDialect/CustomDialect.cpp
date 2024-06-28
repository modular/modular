//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CustomDialect/CustomDialect.h"

using namespace M;
using namespace KGEN;
using namespace Custom;

//===----------------------------------------------------------------------===//
// CustomDialect
//===----------------------------------------------------------------------===//

void CustomDialect::initialize() { allowUnknownOperations(); }

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/CustomDialect/CustomDialect.cpp.inc"
