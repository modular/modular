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

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGDIALECT_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGDIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// MOGGDialect
//===----------------------------------------------------------------------===//

def MOGGDialect : Dialect {
  let name = "mogg";

  let summary = "Modular operations graph generators.";
  let description = [{
    MOGG specifies which kernel implements each op, which gives fusion enough
    information about the kernels' numerics to decide whether fusing is safe.
    Fusion embeds kernels in the output region of the root kernel. Selected
    kernels then lower, generally in buffer semantics, to MO for integration
    with framework kernels, memory optimization, and lowering to MEF.
  }];

  let cppNamespace = "::M::MOGG";
  let dependentDialects = ["::M::MO::MODialect", "::M::DebugInfo::DebugInfoDialect"];
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
  private:
    void registerAttributes();
    void registerOperations();
    void registerTypes();
  }];

  let useDefaultTypePrinterParser = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGDIALECT_TD
