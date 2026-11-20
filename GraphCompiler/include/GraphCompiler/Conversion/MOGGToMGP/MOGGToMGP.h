//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOGGTOMGP_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOGGTOMGP_H

#include "Support/AlignedAlloc.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

// Need to include the dialects we use, which includes many KGEN ones. These
// KGEN dialects are needed so we can load the LIT kernels to dynamically create
// the list of available MOGG operations.
#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/LITDialect/LITDialect.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/ToolCommon/InitAllDialects.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

namespace M::MGP {

/// Generate the code for declaring the pass.
#define GEN_PASS_DECL_MOGGTOMGP
#define GEN_PASS_DECL_RAISEBACKINDEXCONSTANT
#define GEN_PASS_DECL_LOWERPRIMITIVES
#include "GraphCompiler/Conversion/MOGGToMGP/MOGGToMGP.h.inc"

/// Generate the code for registering the pass.
#define GEN_PASS_REGISTRATION
#include "GraphCompiler/Conversion/MOGGToMGP/MOGGToMGP.h.inc"

} // namespace M::MGP

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOGGTOMGP_H
