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

// Test helpers for gunit tests that make it more easy to handle
// `ErrorOr<T>` and `ErrorOrSuccess` with less boilerplate.

#include "gmock/gmock.h"

#define ASSIGN_NO_ERR(type, var, expr)                                         \
  type var;                                                                    \
  do {                                                                         \
    auto _err_or_val_ = (expr);                                                \
    ASSERT_FALSE(_err_or_val_.isError()) << _err_or_val_.getError() << "\n";   \
    var = _err_or_val_.takeValue();                                            \
  } while (0)

#define EXPECT_NO_ERR(expr)                                                    \
  do {                                                                         \
    auto err = (expr);                                                         \
    EXPECT_FALSE(err.isError()) << err.getError() << "\n";                     \
  } while (0)

#define EXPECT_ERR(err)                                                        \
  do {                                                                         \
    EXPECT_TRUE((err).isError());                                              \
  } while (0)

#define EXPECT_ERR_MSG(expr, msg)                                              \
  do {                                                                         \
    auto err = (expr);                                                         \
    EXPECT_TRUE(err.isError());                                                \
    EXPECT_THAT(err.getError(), testing::HasSubstr(msg));                      \
  } while (0)
