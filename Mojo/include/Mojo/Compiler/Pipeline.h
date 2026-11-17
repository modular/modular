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

#ifndef KGEN_COMPILER_OPTPIPELINE_H
#define KGEN_COMPILER_OPTPIPELINE_H

namespace mlir {
class PassManager;
} // namespace mlir

namespace M::KGEN {

class CompilationOptions;

/// The optimization and simplification passes that run right after the
/// elaborator.
void buildPostElaborationPipeline(mlir::PassManager &pm,
                                  const CompilationOptions &options);

/// Convention lowering, inlining and the optimizations that run at every
/// level.
void buildFirstOptPipeline(mlir::PassManager &pm,
                           const CompilationOptions &options);

/// The `-O1` and `-O2` optimization rounds, then loop, closure and async
/// lowering.
void buildLateOptPipeline(mlir::PassManager &pm,
                          const CompilationOptions &options);

} // namespace M::KGEN

#endif // KGEN_COMPILER_OPTPIPELINE_H
