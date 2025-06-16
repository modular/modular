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

from os import abort

from python import Python, PythonObject
from python.bindings import PythonModuleBuilder


@export
fn PyInit_mojo_module() -> PythonObject:
    var b: PythonModuleBuilder
    try:
        b = PythonModuleBuilder("mojo_module")
    except e:
        return abort[PythonObject](
            String("failed to create Python module: ", e)
        )

    # def_function with return, raising
    b.def_function[takes_zero_raises_returns]("takes_zero_raises_returns")
    b.def_function[takes_one_raises_returns]("takes_one_raises_returns")
    b.def_function[takes_two_raises_returns]("takes_two_raises_returns")
    b.def_function[takes_three_raises_returns]("takes_three_raises_returns")

    # def_function with return, not raising
    b.def_function[takes_zero_returns]("takes_zero_returns")
    b.def_function[takes_one_returns]("takes_one_returns")
    b.def_function[takes_two_returns]("takes_two_returns")
    b.def_function[takes_three_returns]("takes_three_returns")

    # def_function with no return, raising
    b.def_function[takes_zero_raises]("takes_zero_raises")
    b.def_function[takes_one_raises]("takes_one_raises")
    b.def_function[takes_two_raises]("takes_two_raises")
    b.def_function[takes_three_raises]("takes_three_raises")

    # def_function with no return, not raising
    b.def_function[takes_zero]("takes_zero")
    b.def_function[takes_one]("takes_one")
    b.def_function[takes_two]("takes_two")
    b.def_function[takes_three]("takes_three")

    try:
        return b.finalize()
    except e:
        return abort[PythonObject](
            String("failed to finalize Python module: ", e)
        )


@export
fn takes_zero_raises_returns() raises -> PythonObject:
    var s = Python().evaluate("getattr(sys.modules[__name__], 's')")
    if s != "just a python string":
        raise "`s` must be 'just a python string'"

    return PythonObject("just another python string")


@export
fn takes_one_raises_returns(a: PythonObject) raises -> PythonObject:
    if a != PythonObject("foo"):
        raise "input must be 'foo'"
    return a


@export
fn takes_two_raises_returns(
    a: PythonObject, b: PythonObject
) raises -> PythonObject:
    if a != PythonObject("foo"):
        raise "first input must be 'foo'"
    return a + b


@export
fn takes_three_raises_returns(
    a: PythonObject, b: PythonObject, c: PythonObject
) raises -> PythonObject:
    if a != PythonObject("foo"):
        raise "first input must be 'foo'"
    return a + b + c


@export
fn takes_zero_returns() -> PythonObject:
    try:
        return takes_zero_raises_returns()
    except e:
        return abort[PythonObject](String("Unexpected Python error: ", e))


@export
fn takes_one_returns(a: PythonObject) -> PythonObject:
    try:
        return takes_one_raises_returns(a)
    except e:
        return abort[PythonObject](String("Unexpected Python error: ", e))


@export
fn takes_two_returns(a: PythonObject, b: PythonObject) -> PythonObject:
    try:
        return takes_two_raises_returns(a, b)
    except e:
        return abort[PythonObject](String("Unexpected Python error: ", e))


@export
fn takes_three_returns(
    a: PythonObject, b: PythonObject, c: PythonObject
) -> PythonObject:
    try:
        return takes_three_raises_returns(a, b, c)
    except e:
        return abort[PythonObject](String("Unexpected Python error: ", e))


@export
fn takes_zero_raises() raises:
    var s = Python().evaluate("getattr(sys.modules[__name__], 's')")
    if s != "just a python string":
        raise "`s` must be 'just a python string'"

    _ = Python().eval(
        "setattr(sys.modules[__name__], 's', 'Hark! A mojo function calling"
        " into Python, called from Python!')"
    )


@export
fn takes_one_raises(list_obj: PythonObject) raises:
    if len(list_obj) != 3:
        raise "list_obj must have length 3"
    list_obj[PythonObject(0)] = PythonObject("baz")


@export
fn takes_two_raises(list_obj: PythonObject, obj: PythonObject) raises:
    if len(list_obj) != 3:
        raise "list_obj must have length 3"
    list_obj[PythonObject(0)] = obj


@export
fn takes_three_raises(
    list_obj: PythonObject, obj: PythonObject, obj2: PythonObject
) raises:
    if len(list_obj) != 3:
        raise "list_obj must have length 3"
    list_obj[PythonObject(0)] = obj + obj2


@export
fn takes_zero():
    try:
        takes_zero_raises()
    except e:
        abort(String("Unexpected Python error: ", e))


@export
fn takes_one(list_obj: PythonObject):
    try:
        takes_one_raises(list_obj)
    except e:
        abort(String("Unexpected Python error: ", e))


@export
fn takes_two(list_obj: PythonObject, obj: PythonObject):
    try:
        takes_two_raises(list_obj, obj)
    except e:
        abort(String("Unexpected Python error: ", e))


@export
fn takes_three(list_obj: PythonObject, obj: PythonObject, obj2: PythonObject):
    try:
        takes_three_raises(list_obj, obj, obj2)
    except e:
        abort(String("Unexpected Python error: ", e))
