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

#ifndef GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHDIALECT_TD
#define GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHDIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// MOSHDialect
//===----------------------------------------------------------------------===//

def MOSHDialect : Dialect {
  let name = "mosh";

  let summary = "Modular high level shape dialect.";
  let description = [{
    Dialect to provide types and operations for writing shape functions and
    shape inference infrastructure.

    This dialect should not depend on the MO dialect.
  }];

  let cppNamespace = "::M::MOSH";
  let dependentDialects = [
    "MDialect",
    "KGEN::KGENDialect",
    "KGEN::POP::POPDialect",
    "mlir::index::IndexDialect",
    "mlir::arith::ArithDialect",
    "mlir::ub::UBDialect",
  ];
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
  private:
    void registerAttributes();
    void registerTypes();
    void registerOperations();
  }];

  let useDefaultTypePrinterParser = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHDIALECT_TD
