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
// Loads the driver's C shared library the way a consumer outside bazel does,
// and resolves every entry point the C header check names through it.
//
// The other C API tests link the implementation statically, so they would
// pass with a library that exports nothing. That is what a dropped export
// attribute on a declaration, or a lost always link on the library the shim
// wraps, produces, and this is the test that notices.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <set>
#include <string>

#include <dlfcn.h>

namespace {

/// The entry point names the C header check pins, read from its source so
/// this test keeps no list of its own. Every check there assigns an entry
/// point to a function pointer, so the names are the identifiers that start
/// with the interface's prefix and end the statement.
std::set<std::string> entryPointsFromHeaderCheck(const std::string &path) {
  std::ifstream in(path);
  const std::string source((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
  const std::string prefix = "M_Driver";
  std::set<std::string> names;
  for (size_t at = source.find(prefix); at != std::string::npos;
       at = source.find(prefix, at + 1)) {
    size_t end = at;
    while (end < source.size() &&
           (std::isalnum(static_cast<unsigned char>(source[end])) ||
            source[end] == '_'))
      ++end;
    size_t next = end;
    while (next < source.size() &&
           std::isspace(static_cast<unsigned char>(source[next])))
      ++next;
    if (next < source.size() && source[next] == ';')
      names.insert(source.substr(at, end - at));
  }
  return names;
}

} // namespace

TEST(DriverCApiShim, ExportsEveryEntryPoint) {
  const char *shim = std::getenv("M_DRIVER_CAPI_SHIM");
  const char *check = std::getenv("M_DRIVER_CAPI_HEADER_CHECK");
  ASSERT_NE(shim, nullptr) << "M_DRIVER_CAPI_SHIM names the shared library";
  ASSERT_NE(check, nullptr)
      << "M_DRIVER_CAPI_HEADER_CHECK names the C header check source";

  const std::set<std::string> names = entryPointsFromHeaderCheck(check);
  ASSERT_GE(names.size(), 50u)
      << "the header check source yielded fewer entry points than it has";

  void *library = dlopen(shim, RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(library, nullptr) << dlerror();
  for (const std::string &name : names)
    EXPECT_NE(dlsym(library, name.c_str()), nullptr)
        << name << " is not exported";

  // The library is live and not just populated, because the version and the
  // count answer through it, and neither needs a device.
  using Version = uint32_t (*)(void);
  auto version =
      reinterpret_cast<Version>(dlsym(library, "M_DriverCApiVersion"));
  ASSERT_NE(version, nullptr);
  EXPECT_EQ(version() >> 24, 1u);
  using Count = int64_t (*)(void);
  auto count =
      reinterpret_cast<Count>(dlsym(library, "M_DriverAcceleratorCount"));
  ASSERT_NE(count, nullptr);
  EXPECT_GE(count(), 0);
  dlclose(library);
}
