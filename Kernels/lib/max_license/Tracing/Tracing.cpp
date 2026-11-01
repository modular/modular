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

#include "Kernels/max_license/Tracing.h"
#include "llvm/ADT/StringRef.h"

using namespace M;

static constexpr int kResultSuccess = 0;

ErrorOr<MojoTrace> MojoTrace::create(StringRef message, StringRef color) {
  void *trace;
  if (create_trace(&trace, (void *)message.data(), message.size(),
                   (void *)color.data(), color.size()) == kResultSuccess)
    return MojoTrace(MojoTracePtr(trace, destroy_trace));

  return Error("unable to create Mojo trace");
}
