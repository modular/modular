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
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "MLRT/GraphRT/GRTDialect/GRTDialect.h"
#include "MLRT/GraphRT/MEFToMLIR.h"
#include "MLRT/GraphRT/MLIRToMEF.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
using namespace M;

int main(int argc, char **argv) {
  DialectRegistry extraDialectsForMLIRToMEF;
  extraDialectsForMLIRToMEF.insert<GRT::GRTDialect>();
  extraDialectsForMLIRToMEF.insert<mlir::quant::QuantDialect>();
  registerMLIRToMEFTranslation(extraDialectsForMLIRToMEF);

  registerMEFToMLIRTranslation();

  return failed(
      mlir::mlirTranslateMain(argc, argv, "GraphRT Translation Testing Tool"));
}
