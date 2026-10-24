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

#ifndef GENERICML_API_C_DETAIL_TYPES_H
#define GENERICML_API_C_DETAIL_TYPES_H

#include "GraphCompiler/Support/ModelTensorSpec.h"
#include "GraphCompiler/Support/SafeTensor/SafeTensor.h"
#include "Kernels/max_license/WeightsRegistry.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/DeviceSpecs.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "max/c/types.h"
#include "max/max_license/API/c/EngineConfig.h"
#include "max/max_license/API/c/EngineContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <string>
#include <vector>

struct M_Status {
  M::ErrorOrSuccess status;
};

struct M_RuntimeConfig {
  M::Engine::Config config;
};

struct M_RuntimeContext {
  M::Engine::Context context;
};

struct M_CompileConfig {
  M::CompilationRequest config;
};

struct M_WeightsRegistry {
  M::WeightsRegistryRef weights;
};

// Holds the tensors parsed from the files passed to `M_loadSafetensors`,
// flattened into a single name-keyed view across all files.
struct M_Safetensors {
  // A materialized tensor: its device-memory view and its parsed metadata.
  struct Entry {
    M::Driver::DeviceBufferRef memory;
    M::SafeTensor::TensorInfo info;
  };

  // Tensor names in load order, for stable index-based enumeration.
  std::vector<std::string> names;
  // name -> materialized entry.
  llvm::StringMap<Entry> entries;
};

struct M_AsyncCompiledModel {
  M::AsyncValueRef<M::CompiledModels> valueOrError;
};

struct M_AsyncModel {
  M::AsyncExecutableModel modelOrError;
};

// Wraps an MLRT device context directly. `spec` carries the cpu/gpu label
// + id the C API surface exposes (M_getDeviceType, M_getDeviceLabel, ...);
// `context` is the MLRT device context backing it.
struct M_Device {
  M::DeviceSpecRef spec;
  M::Driver::DeviceContextRef context;
};

// A device-resident tensor: the MLRT buffer holding it, the spec describing
// its shape and dtype, and the device it lives on. The device is stored rather
// than derived from the buffer: the cpu/gpu label the C API reports
// (M_getTensorDevice, M_getTensorSpec) lives on the DeviceSpec.
// Shared rather than owned: every consumer but M_getTensorDevice reads only
// `ref`, and the spec itself lives as long as the session's Engine::Config.
struct M_TensorStorage {
  M::Driver::DeviceBufferRef buffer;
  M::TensorSpec spec;
  M::DeviceSpecRef device;
};

struct M_AsyncTensor {
  std::shared_ptr<M_TensorStorage> storage;
};

struct M_TensorNameArray {
  llvm::SmallVector<std::string> names;
};

struct M_TensorSpec {
  M::ModelTensorSpec spec;
  M::DeviceRef device;
};

struct M_AsyncTensorMap {
  llvm::SmallVector<std::shared_ptr<M_TensorStorage>> tensors;
  llvm::StringMap<size_t> indices;
  const M::Engine::Context *context;
};

#endif // GENERICML_API_C_DETAIL_TYPES_H
