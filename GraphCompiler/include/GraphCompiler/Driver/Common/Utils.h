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

#ifndef GENERICML_COMMONDRIVER_UTILS_H
#define GENERICML_COMMONDRIVER_UTILS_H

#include "AsyncRT/ForwardDecls.h"
#include "GraphCompiler/Driver/Common/DriverInputFormat.h"
#include "Support/Buffer.h"
#include "Support/Context.h"
#include "Support/ForwardDecls.h"
#include "Support/LLVMCompilerForwardDecls.h"

#include <functional>
#include <memory>
#include <string>

namespace M {

class CommonOptions;

// FIXME(RUNP-82): Duplicated because of #1639.
ErrorOr<std::string> getFileContentIdentity(const BufferRef &file,
                                            FileFormat format);

/// Saves an intermediate file with a dump of `module` if --saved-temps is set.
/// `baseName` is used as the output file stem when `file` has no backing path.
void writeIntermediateFileIfRequested(const std::string &identifier,
                                      ModuleOp module,
                                      const CommonOptions *options,
                                      const BufferRef &file,
                                      llvm::StringRef baseName = "temp");

/// Returns the per-stage IR dump directory, or empty if dumping is disabled.
/// Config keys, in order of preference:
///   1. `max-debug.ir-output-dir` (set via `MODULAR_DEBUG=ir-output-dir=...`,
///      `[max-debug] ir-output-dir = ...` in `modular.cfg`, or
///      `InferenceSession.debug.ir_output_dir = ...`).
///   2. `max.temps_dir` (legacy; also set by `MODULAR_MAX_TEMPS_DIR`).
///
/// All IR-dump gating should go through this so one setting drives both.
std::string getMaxIRDumpDir();

} // namespace M

#endif // GENERICML_COMMONDRIVER_UTILS_H
