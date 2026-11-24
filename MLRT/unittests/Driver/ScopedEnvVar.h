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

#ifndef ASYNCRT_UNITTESTS_SCOPED_ENV_VAR_H
#define ASYNCRT_UNITTESTS_SCOPED_ENV_VAR_H

#include "Support/PlatformUtils.h"
#include "llvm/ADT/StringRef.h"

#include <string>

#if MODULAR_MACOS

#include <crt_externs.h>
static char **listOfEnvironmentVariables(void) {
  return *_NSGetEnviron(); // man 7 environ
}

#else

#include <unistd.h>
static char **listOfEnvironmentVariables(void) {
  return environ; // from cstdlib
}

#endif // MODULAR_MACOS

namespace {

// RAII wrapper for environment variable management.
class ScopedEnvVar {
public:
  ScopedEnvVar(llvm::StringRef name, llvm::StringRef value) : varName(name) {
    originalValue = nullptr;

    // Attempt to save the existing variable if it exists.
    char **environ = listOfEnvironmentVariables();
    for (char **it = environ; *it != nullptr; ++it) {
      auto str = llvm::StringRef(*it);
      auto [prefix, _] = str.split('=');

      if (prefix == name) {
        originalValue = *it;
        break;
      }
    }

    // Set the new value
    newVar = std::string(name) + "=" + std::string(value);
    putenv(newVar.data());
  }

  ~ScopedEnvVar() {
    // Restore the original value or unset if it didn't exist
    if (originalValue) {
      putenv(originalValue);
    } else {
      unsetenv(varName.c_str());
    }
  }

  // Disable copy and move.
  ScopedEnvVar(const ScopedEnvVar &) = delete;
  ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;
  ScopedEnvVar(ScopedEnvVar &&) = delete;
  ScopedEnvVar &operator=(ScopedEnvVar &&) = delete;

private:
  std::string varName;
  std::string newVar;
  char *originalValue;
};

} // namespace

#endif // ASYNCRT_UNITTESTS_SCOPED_ENV_VAR_H
