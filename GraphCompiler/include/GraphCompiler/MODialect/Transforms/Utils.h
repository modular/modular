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
//
// Utilities common to MO passes.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOOps.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/Support/raw_os_ostream.h"

#ifndef GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_UTILS_H
#define GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_UTILS_H

namespace M::MO {

inline void printValue(llvm::raw_ostream &os, Value value) {
  value.printAsOperand(os, mlir::OpPrintingFlags().printGenericOpForm());
}

inline void printOp(llvm::raw_ostream &os, Operation *op) {
  op->getName().print(os);
}

inline void printGraph(llvm::raw_ostream &os, GraphOp graphOp) {
  graphOp.print(os, mlir::OpPrintingFlags().printGenericOpForm());
  os << "\n";
}

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_UTILS_H
