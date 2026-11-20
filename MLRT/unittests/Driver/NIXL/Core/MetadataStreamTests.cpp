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

#include "stream/metadata_stream.h"
#include "gtest/gtest.h"

#include <cerrno>
#include <sys/socket.h>

// Regression test for macOS compatibility
TEST(MetadataStreamTest, ListenerSocketIsNonBlocking) {
  nixlMDStreamListener listener(19876);
  listener.setupListener();

  nixl::scopedFd csock = listener.acceptClient();

  EXPECT_FALSE(csock.valid());
  EXPECT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK)
      << "Expected EAGAIN or EWOULDBLOCK, got errno=" << errno;
}
