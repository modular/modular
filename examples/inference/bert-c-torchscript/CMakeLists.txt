##===----------------------------------------------------------------------===##
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
##===----------------------------------------------------------------------===##

cmake_minimum_required(VERSION 3.24)

project(max-examples LANGUAGES C)

list(APPEND CMAKE_MODULE_PATH "$ENV{MAX_PKG_DIR}/lib/cmake")
include(AddMaxEngine)

add_executable(bert main.c)
target_link_libraries(bert PUBLIC max-engine)
