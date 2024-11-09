//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines the logic for automatically generating Python bindings for
// Mojo functions and types.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_PYTHONBINDINGSGEN_H
#define KGEN_MOJOPARSER_PYTHONBINDINGSGEN_H

#include "Support/LogicalResult.h"

namespace M::KGEN::LIT {
class ASTDecl;
class SharedState;

LogicalResult generatePythonBindings(ASTDecl &moduleDecl);
} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PYTHONBINDINGSGEN_H
