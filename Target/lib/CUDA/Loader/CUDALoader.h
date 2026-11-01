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
// This file declares the CUDA driver interface.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CUDA_CUDALOADER_H
#define TARGET_LIB_CUDA_CUDALOADER_H

#include "Target/lib/CUDA/Error.h"

#include <cuda.h>

namespace M::Target::CUDA {

struct CUDALoader {
  void *lib = nullptr;

#define CUDA_FN(name) decltype(&::name) name = nullptr;
#include "CUDALoader.inc"
#undef CUDA_FN

  CUDALoader(const CUDALoader &) = delete;
  CUDALoader &operator=(const CUDALoader &) = delete;
  CUDALoader(CUDALoader &&) = default;
  CUDALoader &operator=(CUDALoader &&) = default;
  CUDALoader() = default;

  static DriverResult<CUDALoader> load();
  void unload();
};

} // namespace M::Target::CUDA

#endif // TARGET_LIB_CUDA_CUDALOADER_H
