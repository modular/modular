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
// This defines types in the MOSH dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHTYPES_TD
#define GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHTYPES_TD

include "GraphCompiler/MOSHDialect/MOSHDialect.td"
include "Mojo/KGENDialect/KGENTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

class MOSH_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MOSHDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def MOSH_ShapeType : MOSH_Type<"Shape", "ape",
  [DeclareTypeInterfaceMethods<ParameterTypeInterface>]>;

#endif // GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHTYPES_TD
