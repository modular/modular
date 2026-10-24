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

#ifndef GENERICML_MO_GRAPHCOMPILER_H
#define GENERICML_MO_GRAPHCOMPILER_H

#include "AsyncRT/ForwardDecls.h"
#include "Support/Buffer.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/DialectRegistry.h"

namespace M::MO {

/// Load dialects in Graph Compiler to the given registry.
void loadDialects(mlir::DialectRegistry &registry);

/// Register Graph Compiler passes. This function is not thread-safe.
void registerPasses();

struct GCPipelineOptions;

/// Compile a module at the MO dialect level down to the MGP level.
LogicalResult runGcPipeline(ModuleOp module,
                            const GCPipelineOptions &gcPipelineOptions);

/// Convert a module at the MGP level to a buffer in MEF format. Destructive:
/// frees the dense resources of the module's MGP constants.
ErrorOr<BufferRef> convertMGPToMEF(ModuleOp module, bool enableMEFMetadata);

constexpr StringLiteral BugReportMessage =
    "=================================================================\n"
    "    !!! MAX AI Engine has encountered an internal error !!!\n"
    "=================================================================\n"
    "Please submit a bug report to https://github.com/modular/modular/issues "
    "and include the crash backtrace along with your command line "
    "invocation.\n";

} // namespace M::MO

#endif // GENERICML_MO_GRAPHCOMPILER_H
