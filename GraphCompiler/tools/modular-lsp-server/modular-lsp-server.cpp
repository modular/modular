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
// Provides an LSP server for all the modular dialects.
//
//===----------------------------------------------------------------------===//

#include "Cache/CachedTransform.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPDialect.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GraphCompiler/RMODialect/RMODialect.h"
#include "MLRT/GraphRT/GRTDialect/GRTDialect.h"
#include "MLRT/GraphRT/MEFDialect/MEFDialect.h"
#include "Mojo/ToolCommon/InitAllDialects.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;
using namespace M;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<
      // MLIR.
      arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
      index::IndexDialect, LLVM::LLVMDialect, quant::QuantDialect,
      mlir::ub::UBDialect,
      // Modular.
      MEF::MEFDialect, DebugInfo::DebugInfoDialect, GRT::GRTDialect, MDialect,
      MGP::MGPDialect, RMO::RMODialect, MOSH::MOSHDialect, MO::MODialect,
      MOGG::MOGGDialect, MAP::MAPDialect>();
  registerAllKGENDialects(registry);
  return failed(MlirLspServerMain(argc, argv, registry));
}
