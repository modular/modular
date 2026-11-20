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

#ifndef MOF_INITALLDIALECTS_H
#define MOF_INITALLDIALECTS_H

#include "GraphCompiler/GraphCompiler.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPDialect.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "MLRT/GraphRT/GRTDialect/GRTDialect.h"
#include "MLRT/GraphRT/MEFDialect/MEFDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace M::MOF {

inline void registerAllDialects(DialectRegistry &registry) {
  registry.insert<MDialect, GRT::GRTDialect, MEF::MEFDialect,
                  mlir::func::FuncDialect, M::MGP::MGPDialect>();
  M::MO::loadDialects(registry);
}
} // namespace M::MOF

#endif // MOF_INITALLDIALECTS_H
