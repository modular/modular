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

// TODO: Move elsewhere, this logic is not driver-specific.

#ifndef MOF_DRIVER_DRIVERIMPL_H
#define MOF_DRIVER_DRIVERIMPL_H

#include "Support/Buffer.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/ML/DebugPrint.h"

namespace M::MOF {

/// Parses the buffer containing MLIR representation of a model in the MO graph
/// dialect.
OwningOpRef<ModuleOp> parseMODialect(M::BufferRef buffer, MLIRContext *ctx);

} // namespace M::MOF

#endif // MOF_DRIVER_DRIVERIMPL_H
