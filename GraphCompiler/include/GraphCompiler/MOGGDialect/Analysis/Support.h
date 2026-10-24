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

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_SUPPORT_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_SUPPORT_H

#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"

namespace M::MOGG {

class MojoRegistry;

/// Holds the Mojo registration name of an MLIR operation.
struct MojoOperationInfo {
  StringRef operationName;
};

/// Returns the Mojo registration name for `op`.
MojoOperationInfo getMojoOperationInfo(Operation *op);

/// Construct the kernel adaptor for the given operation.
ErrorOr<KernelDeclAdaptor> getMojoKernelAdaptor(const MojoRegistry &registry,
                                                Operation *op);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_ANALYSIS_SUPPORT_H
