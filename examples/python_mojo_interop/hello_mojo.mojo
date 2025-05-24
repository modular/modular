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


from python import PythonObject, PythonModule
from python.bindings import PythonModuleBuilder
from os import abort


# An interface for this Mojo module must be exported to Python.
@export
fn PyInit_hello_mojo() -> PythonObject:
    try:
        # A Python module is constructed, matching the name of this Mojo module.
        var module = PythonModuleBuilder("hello_mojo")
        # The functions to be exported are registered within this module.
        module.def_function[passthrough]("passthrough")
        return module.finalize()
    except e:
        return abort[PythonObject](
            String("failed to create Python module: ", e)
        )


fn passthrough(value: PythonObject) raises -> PythonObject:
    """A very basic function illustrating passing values to and from Mojo."""
    return value + " world from Mojo"
