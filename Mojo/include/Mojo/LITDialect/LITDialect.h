//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file declares the LIT MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_H
#define KGEN_LITDIALECT_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Dialect.h"

// Pull in the dialect definition.
#pragma GCC visibility push(default)
#include "Mojo/LITDialect/LITDialect.h.inc"
#pragma GCC visibility pop

#endif // KGEN_LITDIALECT_H
