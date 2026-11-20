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

#include "max/max_license/API/c/EngineConfig.h"
#include "Support/Configuration.h"
#include "llvm/ADT/StringExtras.h"

using namespace M;
//===----------------------------------------------------------------------===//
// Config initialization
//===----------------------------------------------------------------------===//
// The runtime config initialization checks the modular.cfg file for a thread
// identifier to use, if specified, under the section 'triton_modular_backend'
// and key 'context_name'. If found, use it as the context name, otherwise
// use 'AsyncRT Thread ' as the prefix for threads in the thread pool.
M::Engine::Config::Config() {
  M::Config config;
  ErrorOr<M::Config> configOr = M::Config::open();
  if (!configOr.isError()) {
    config = std::move(*configOr);
  }

  // Extract a context name.
  llvm::StringRef val = config.getValue("runtime.context_name");
  if (!val.empty()) {
    contextThreadPoolName = val.str();
  }

  // Consider a default num_threads.
  val = config.getValue("runtime.num_threads");
  if (!val.empty()) {
    int parsedValue;
    if (llvm::to_integer<int>(val, parsedValue))
      setNumThreads(parsedValue);
  }

  // Extract a thread busy wait time.
  val = config.getValue("runtime.thread_busy_wait_time_us");
  if (!val.empty()) {
    size_t parsedValue;
    if (llvm::to_integer<size_t>(val, parsedValue))
      threadBusyWaitTime = std::chrono::microseconds(parsedValue);
  }
}

void M::Engine::Config::setNumThreads(size_t count) {
  numThreads = count; // May still be zero (intentional).
}

void M::Engine::Config::setMaxThreads(size_t count) {
  maxThreads = count; // May also be zero.
}
