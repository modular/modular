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

#ifndef GENERICML_GRAPHCOMPILER_INTERFACES_CONVERTIBLETOMGPTYPEINTERFACE_TD
#define GENERICML_GRAPHCOMPILER_INTERFACES_CONVERTIBLETOMGPTYPEINTERFACE_TD

include "mlir/IR/OpBase.td"


def ConvertibleToMGPTypeInterface : TypeInterface<"ConvertibleToMGPTypeInterface"> {
  let description = [{
    This interface allows foreign types to be converted in the MOToMGP pass.
  }];

  let cppNamespace = "::M::MGP";

  let methods = [
    InterfaceMethod<[{
      Returns the converted type of this type.
    }],
    "::mlir::Type", "convertType">,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_INTERFACES_CONVERTIBLETOMGPTYPEINTERFACE_TD
