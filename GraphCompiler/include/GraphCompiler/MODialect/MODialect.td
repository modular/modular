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

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MODIALECT_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MODIALECT_TD

include "mlir/IR/DialectBase.td"
include "Mojo/KGENDialect/KGENDialect.td"
include "Support/MDialect/MDialect.td"

//===----------------------------------------------------------------------===//
// MODialect
//===----------------------------------------------------------------------===//

def MODialect : Dialect {
  let name = "mo";

  let summary = "MAX Graph operations.";
  let description = [{
    The MO Dialect represents a MAX graph and its operations. It is a frontend
    dialect that serves as input to the graph compiler.
  }];

  let cppNamespace = "::M::MO";
  let dependentDialects = [
    "M::MDialect",
    "MOSH::MOSHDialect",
  ];
  let hasCanonicalizer = 1;
  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
  private:
    void registerAttributes();
    void registerTypes();
    void registerOperations();
  }];

  let useDefaultTypePrinterParser = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MODIALECT_TD
