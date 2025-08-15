# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #


# start-static-method-define
struct Logger:
    fn __init__(out self):
        pass

    @staticmethod
    fn log_info(message: String):
        print("Info: ", message)
        # end-static-method-define


def main():
    # start-static-method-invoke
    Logger.log_info("Static method called.")
    var l = Logger()
    l.log_info("Static method called from instance.")
    # end-static-method-invoke
    _ = l^
