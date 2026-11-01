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

#include "Common.h"
#include "HAL/M_driver_bundle.h"
#include "MetalCpp.h"
#include "llvm/ADT/StringRef.h"

#include <string>

struct M_driver_runtime_bundle {
  NS::SharedPtr<MTL::Library> library;

  [[nodiscard]] static DriverResult<M_driver_runtime_bundle_t>
  load(M_driver_context_t ctx, M_driver_static_bundle_t bundle,
       M_driver_bundle_compilation_options_t opts);

  static M_driver_status_t unload(M_driver_runtime_bundle_t bundle);
};

struct M_driver_function {
  [[nodiscard]] static DriverResult<M_driver_function_t>
  create(M_driver_context_t context, M_driver_runtime_bundle_t bundle,
         llvm::StringRef name);
  static M_driver_status_t destroy(M_driver_function_t function);

  M_driver_context_t context;
  std::string name;
  NS::SharedPtr<MTL::ComputePipelineState> pipelineState;
  size_t argumentCount;
};
