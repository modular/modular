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

# DOC: mojo/docs/manual/python/mojo-from-python.mdx

from python import PythonObject
from python.bindings import PythonModuleBuilder
from os import abort


@fieldwise_init
struct Person(Movable, Representable):
    var name: String
    var age: Int

    fn __repr__(self) -> String:
        return String("Person(", self.name, ", ", self.age, ")")

    @staticmethod
    fn py_init(
        out self: Person, args: PythonObject, kwargs: PythonObject
    ) raises:
        # Validate argument count
        if len(args) != 2:
            raise Error("Person() takes exactly 2 arguments")

        # Convert Python arguments to Mojo types
        var name = String(args[0])
        var age = Int(args[1])

        self = Self(name, age)


@export
fn PyInit_person_module() -> PythonObject:
    try:
        var mb = PythonModuleBuilder("person_module")

        _ = mb.add_type[Person]("Person").def_py_init[Person.py_init]()

        return mb.finalize()
    except e:
        return abort[PythonObject](
            String("error creating Python Mojo module:", e)
        )
