# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
from register import *
from tensor import InputTensor, OutputTensor
import compiler_internal as compiler


# This function has the same name as the parameter for the kernel registration.
fn top_k():
    pass


@compiler.register("parameter_name_overload")
struct ParameterNameOverload:
    # The top_k parameter matches the name of the top_k function defined above.
    @staticmethod
    fn execute[
        top_k: Int
    ](result: OutputTensor, input: InputTensor,) -> None:
        print("Success!")
