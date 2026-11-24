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

#ifndef GENERICML_GRAPHCOMPILER_INTERFACES_EMITSMOJOINTERFACE_TD
#define GENERICML_GRAPHCOMPILER_INTERFACES_EMITSMOJOINTERFACE_TD

include "mlir/IR/OpBase.td"

def EmitsMojoInterface : OpInterface<"EmitsMojoInterface"> {
  let description = [{
    An op that implements this interface will ultimately lower to Mojo text.
  }];

  let cppNamespace = "::M::MOGG";

  let methods = [
    InterfaceMethod<
      "Return the body of the kernel, which should be the code we will compile",
      "mlir::LogicalResult", "emitMojo", (ins "M::MOGG::EmitMojoState&":$state)
    >,
    InterfaceMethod<
      "Return a boolean to indicate whether the emitted code raises exceptions.",
      "bool", "isThrows", (ins), [{}], [{ return false; }]
    >,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_INTERFACES_EMITSMOJOINTERFACE_TD
