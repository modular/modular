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
// Dialect-level interfaces for tensors in the MGP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPTENSORINTERFACE_H
#define GENERICML_MGPDIALECT_MGPTENSORINTERFACE_H

#include "GraphCompiler/MGPDialect/Dialect/BufferBuilder.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOpInterfaces.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "MLRT/GraphRT/GRTDialect/GRTChainInterface.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpDefinition.h"

namespace M::MGP {

//===----------------------------------------------------------------------===//
// TensorBufferDialectInterface
//===----------------------------------------------------------------------===//

/// Interface for dialects which must manipulate device-specific tensors,
/// buffers and chain values.
struct TensorBufferDialectInterface
    : public mlir::DialectInterface::Base<TensorBufferDialectInterface> {

  /// Returns all the interfaces of this type active in the given MGP::ModelOp.
  static DenseSet<TensorBufferDialectInterface *>
  getAllInterfaces(ModelOp modelOp);

  /// Adds to builderContext builder for all the devices in the given model.
  static BufferBuilderContext getAllBufferBuilders(ModelOp modelOp);

  /// Returns the buffer builder appropriate for the device which will execute
  /// op. Returns the builder for the 'host' device for ops which don't have
  /// a specific device.
  static MGPBufferBuilder *
  getBuilderForOpExecution(const BufferBuilderContext &builderContext,
                           Operation *op);

  /// Returns the buffer builder appropriate for the device which will store
  /// values of type.
  static MGPBufferBuilder *
  getBuilderForTypeStorage(const BufferBuilderContext &builderContext,
                           Type type);

  TensorBufferDialectInterface(Dialect *dialect) : Base(dialect) {}

  /// Adds to BuilderContext the type and buffer builder for all the devices
  /// used by modelOp. Returns the distinguished 'host' device.
  virtual DeviceRefAttr addBuilders(ModelOp modelOp,
                                    BufferBuilderContext &builderContext) = 0;

  /// Returns the device that will execute op.
  virtual DeviceRefAttr
  getExecutionDevice(const BufferBuilderContext &builderContext,
                     Operation *op) = 0;

  /// Returns the device that will store values of type.
  virtual DeviceRefAttr
  getStorageDevice(const BufferBuilderContext &builderContext, Type type) = 0;

  virtual void vtableAnchor();
};

//===----------------------------------------------------------------------===//
// MGPTensorBufferDialectInterface
//===----------------------------------------------------------------------===//

/// Concrete TensorBufferDialectInterface for the device-generic MGP dialect.
struct MGPTensorBufferDialectInterface : public TensorBufferDialectInterface {
  using TensorBufferDialectInterface::TensorBufferDialectInterface;

  DeviceRefAttr addBuilders(ModelOp modelOp,
                            BufferBuilderContext &builderContext) override;
  DeviceRefAttr getExecutionDevice(const BufferBuilderContext &builderContext,
                                   Operation *op) override;
  DeviceRefAttr getStorageDevice(const BufferBuilderContext &builderContext,
                                 Type type) override;
};

} // namespace M::MGP

#endif // GENERICML_MGPDIALECT_MGPTENSORINTERFACE_H
