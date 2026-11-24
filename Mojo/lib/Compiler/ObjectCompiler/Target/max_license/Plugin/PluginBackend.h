//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// PluginBackend is a registered singleton (like the other backends) that owns
// one TargetBackend per dynamically-loaded compiler plugin. It does not emit
// itself; it resolves a target triple to the per-plugin backend that claims it.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_TARGET_PLUGIN_PLUGINBACKEND_H
#define KGEN_COMPILER_TARGET_PLUGIN_PLUGINBACKEND_H

#include "Target/Host/HostBackend.h"

namespace M::KGEN {

/// Owner/dispatcher for the per-plugin backends. Registered with
/// `TargetBackendRegistry` like every other backend, but it never handles a
/// target itself: `matches()` returns false and `resolve()` returns one of the
/// per-plugin backends it owns. Each owned backend wraps a single plugin, so it
/// carries no mutable per-target selection and is safe to share across
/// compilations.
class PluginBackend : public HostBackend {
public:
  llvm::StringRef name() const override { return "plugin"; }

  bool matches(const llvm::Triple &) const override { return false; }

  const TargetBackend *resolve(const llvm::Triple &triple) const override;

  // This is max only target
  bool isBaseTarget() const override { return false; }
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_TARGET_PLUGIN_PLUGINBACKEND_H
