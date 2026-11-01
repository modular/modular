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

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_H

#include "GraphCompiler/MODialect/Transforms/ConstantFoldPass.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Pass/Pass.h"

// Must include dependent dialects
#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/KGENDialect/KGENDialect.h"
#include "Mojo/LITDialect/LITDialect.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace M::MO {

#define GEN_PASS_DECL
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

#define GEN_PASS_REGISTRATION
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

/// Creates the NanCheckPass, which checks one of every `stride` float kernel
/// outputs (`stride == 1` checks all, prohibitively slow on large models).
/// The default matches the `max-debug.nan-check-stride` knob.
std::unique_ptr<Pass> createNanCheckPass(uint64_t stride = 20);

/// Creates the DumpStagePass, which writes the IR to
/// `<baseName>.<suffix>.mlir`. `baseName` is shared across one pipeline run:
/// the first dump derives it from the non-subgraph GraphOp names, which
/// lowering to MGP later removes.
std::unique_ptr<Pass>
createDumpStagePass(std::string suffix, std::shared_ptr<std::string> baseName);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOTRANSFORMS_H
