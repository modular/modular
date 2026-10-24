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
// Activation helpers for frontends converting to RMO.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOSUPPORT_ACTIVATIONS_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOSUPPORT_ACTIVATIONS_H

#include "GraphCompiler/MODialect/MOTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"

namespace M::RMO {
/// Builds the operations needed to perform the PRelu operation in MO.
/// PRelu(x) = max(x, broadcast(0)) + alpha * min(x, broadcast(0))
MO::TensorValue convertPRelu(OpBuilder &builder, Location loc,
                             MO::TensorValue x, MO::TensorValue alpha);

/// Builds the operations needed to perform the relu operation in MO.
/// Relu(x) = max(x, broadcast(0))
MO::TensorValue convertRelu(OpBuilder &builder, Location loc,
                            MO::TensorValue x);

/// Builds the operations needed to perform the relu6 operation in MO.
/// ReLU6(x) = max(broadcast(0), min(x, broadcast(6)))
MO::TensorValue convertRelu6(OpBuilder &builder, Location loc,
                             MO::TensorValue x);

} // namespace M::RMO

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOSUPPORT_ACTIVATIONS_H
