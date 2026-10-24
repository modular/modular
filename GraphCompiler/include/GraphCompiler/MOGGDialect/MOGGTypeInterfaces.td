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
// This defines type interfaces in the MOGG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGTYPEINTERFACES_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

def MOGG_ElementTypeInterface : TypeInterface<"ElementTypeInterface"> {
  let cppNamespace = "::M::MOGG";
  let description = [{
    A value a fusion body computes on, such as a SIMD vector or a tile, that
    holds elements of one scalar dtype.

    Temporary: fusion bodies will compute on tiles only. Once `!mogg.simd` is
    deleted, this interface goes with it.
  }];
  let methods = [
    InterfaceMethod<"Returns the scalar element type.", "::mlir::Type",
                    "getElementType">,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGTYPEINTERFACES_TD
