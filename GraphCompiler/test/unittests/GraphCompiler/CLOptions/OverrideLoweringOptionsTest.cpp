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

#include "GraphCompiler/GraphCompilerCLOptions.h"
#include "gtest/gtest.h"

using namespace M::MO;

class OverrideLoweringOptionsTest : public ::testing::Test {};

/// Regression test for #30415.
/// This tests that after overriding lowering options with command-line options,
/// `opt.hasValue()` returns true.
TEST_F(OverrideLoweringOptionsTest, HasValue) {
  GCPipelineOptions compilerOptions;
  EXPECT_FALSE(compilerOptions.cacheDir.hasValue());

  GraphCompilerOptions clOptions;
  clOptions.cacheDir = "foo";
  clOptions.overrideLoweringOptions(compilerOptions);
  EXPECT_TRUE(compilerOptions.cacheDir.hasValue());
}
