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
// This file provides a singleton manager for the Tracy CUDA profiling context.
// The Tracy CUDA context is initialized once when the first CUDA device context
// is created, and is shared across all CUDA contexts.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROFILING_TRACYCUDA_H
#define SUPPORT_PROFILING_TRACYCUDA_H

#if TRACY_CUDA_ENABLE

namespace tracy {
struct CUDACtx;
} // namespace tracy

namespace M {

/// Singleton manager for Tracy CUDA profiling context.
/// The Tracy CUDA context is initialized once on first use and shared across
/// all CUDA device contexts. Unlike other Tracy GPU backends, the CUDA backend
/// has a background collector thread, so no periodic collect() calls are
/// needed.
class TracyCudaContext {
public:
  /// Get the singleton instance, initializing Tracy CUDA context if needed.
  /// This is thread-safe (Meyers singleton).
  static TracyCudaContext &getInstance();

  /// Check if the Tracy CUDA context was successfully initialized.
  bool isInitialized() const { return ctx_ != nullptr; }

  // Delete copy/move constructors and assignment operators.
  TracyCudaContext(const TracyCudaContext &) = delete;
  TracyCudaContext &operator=(const TracyCudaContext &) = delete;
  TracyCudaContext(TracyCudaContext &&) = delete;
  TracyCudaContext &operator=(TracyCudaContext &&) = delete;

private:
  TracyCudaContext();
  ~TracyCudaContext();

  tracy::CUDACtx *ctx_;
};

/// Initialize the Tracy CUDA context singleton and start profiling.
/// Safe to call multiple times; initialization only happens once.
/// This should be called when the first CUDA device context is created.
inline void initTracyCuda() { TracyCudaContext::getInstance(); }

} // namespace M

#else // !TRACY_CUDA_ENABLE

namespace M {

// No-op implementation when Tracy CUDA support is disabled (e.g. on macOS or
// when building without an NVIDIA GPU target).
inline void initTracyCuda() {}

} // namespace M

#endif // TRACY_CUDA_ENABLE

#endif // SUPPORT_PROFILING_TRACYCUDA_H
