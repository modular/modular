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
#ifndef GENERICML_GRAPHCOMPILER_SUPPORT_EMITMOJODECLARATIVECALL_H
#define GENERICML_GRAPHCOMPILER_SUPPORT_EMITMOJODECLARATIVECALL_H

#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace M::MOGG {

/// A kernel entry point and the parameters inferred for one call to it. A call
/// that is a statement emits through `MojoBuilder::call`; a call whose result
/// the caller uses builds the expression with `operator()`.
struct KernelCallee {
  // `Kernel[...].fn`, with the kernel's struct parameters applied.
  MojoValue function;
  // The entry point's own parameters.
  SmallVector<MojoValue> parameters;

  // Returns `Kernel[...].fn[...](operands...)`.
  MojoValue operator()(ArrayRef<MojoValue> operands) const {
    return function(parameters, operands);
  }
};

/// How a kernel receives an operand, which the launch passes as itself and
/// encodes as its device type.
enum class DeviceArgKind {
  // As is.
  Scalar,
  // As its `TileTensor` projection, which the kernel rebuilds the tensor or
  // buffer from.
  Tile,
  // As the access-enabled view on which its fusion runs, which the kernel uses
  // as is.
  Fused,
};

/// Holds state for emitting a call to a device-side entry point.
struct DeclarativeCall {
  // def execute_configuration(...)
  KernelCallee executeConfiguration;
  // def execute_device(...)
  KernelCallee executeDevice;
  // The IR value of each of `operands`. `DeviceKernelArgs::add` keys on it, so
  // an operand shared by several calls in a graph blob becomes one argument.
  SmallVector<Value> values;
  // List of operands for both calls.
  SmallVector<MojoValue> operands;
  // How the kernel receives each of `operands`.
  SmallVector<DeviceArgKind> kinds;
};

/// The arguments of a device kernel, and the values the launch passes for them:
/// each operand as itself, which the launch encodes as its device type. Each
/// vector holds one entry per argument, at the argument's index.
struct DeviceKernelArgs {
  SmallVector<NamedMojoDef> parameters;
  SmallVector<MojoValue> launchArgs;
  SmallVector<MojoValue> types;
  SmallVector<DeviceArgKind> kinds;
  // Maps an operand's IR value to the index of the argument that carries it, so
  // an operand shared by several calls crosses the launch once.
  llvm::DenseMap<Value, unsigned> indexOf;

  // Adds `call`'s operands as arguments and returns their indices. Emits each
  // operand's type, and records the kernel argument for it and the value the
  // launch passes. For a tensor `operand0`, a scalar `operand1` and a fused
  // tensor `operand2`:
  //
  //   comptime T0 = type_of(operand0)  # argument a0: T0.device_type,
  //                                    # passed operand0
  //   comptime T1 = type_of(operand1)  # argument a1: T1, passed operand1
  //   comptime T2 = type_of(operand2)  # argument a2: T2.device_type,
  //                                    # passed operand2
  SmallVector<unsigned> add(MojoBuilder &builder, const DeclarativeCall &call) {
    SmallVector<unsigned> indices;
    for (auto [value, operand, kind] :
         llvm::zip_equal(call.values, call.operands, call.kinds)) {
      auto [it, inserted] = indexOf.try_emplace(value, parameters.size());
      indices.push_back(it->second);
      if (!inserted)
        continue;
      MojoValue type = builder.comptime(MojoValue("type_of")({}, {operand}));
      types.push_back(type);
      kinds.push_back(kind);
      parameters.emplace_back("operand", kind == DeviceArgKind::Scalar
                                             ? type
                                             : type.get("device_type"));
      launchArgs.push_back(operand);
    }
    return indices;
  }

  // Adds a scalar argument of `type` that the launch passes `value` for, and
  // returns its index.
  unsigned addScalar(const MojoValue &value, const MojoValue &type) {
    types.push_back(type);
    kinds.push_back(DeviceArgKind::Scalar);
    parameters.emplace_back("operand", type);
    launchArgs.push_back(value);
    return parameters.size() - 1;
  }

  // Rebuilds, inside the kernel, the operands at `indices`: a tensor from its
  // device type, a scalar or a fused tensor as is. For the operands above:
  //
  //   T0(a0), a1, a2
  SmallVector<MojoValue> rebuild(const FunctionBody &body,
                                 ArrayRef<unsigned> indices) const {
    SmallVector<MojoValue> operands;
    for (unsigned i : indices)
      operands.push_back(kinds[i] == DeviceArgKind::Tile
                             ? types[i]({}, {body.arguments[i].resolvedName})
                             : body.arguments[i].resolvedName);
    return operands;
  }
};

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_SUPPORT_EMITMOJODECLARATIVECALL_H
