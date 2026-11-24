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

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MGPSUPPORT_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MGPSUPPORT_H

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"

namespace M::MGP {

inline MGP::TensorSpecType createTensorSpecType(MO::TensorType tensor) {
  auto shape = tensor.getShape().getIntDims();
  return MGP::TensorSpecType::get(shape, tensor.getElementType());
}

} // namespace M::MGP

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MGPSUPPORT_H
