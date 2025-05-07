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
# RUN: %bare-mojo build %S/mojo_module.mojo --emit shared-lib -o mojo_module.so
# RUN: python3 %s

import sys
import unittest

# Put the current directory (containing .so) on the Python module lookup path.
sys.path.insert(0, "")

# Imports from 'mojo_module.so'
import mojo_module as def_method


class TestPythonTypeBuilderDefMethod(unittest.TestCase):
    def test_get_name(self):
        person = def_method.Person()
        self.assertEqual(person.get_name(), "John Smith")

        setattr(sys.modules[__name__], "deny_name", True)
        try:
            with self.assertRaises(Exception) as cm:
                person.get_name()
            self.assertEqual(cm.exception.args, ("name cannot be accessed",))
        finally:
            delattr(sys.modules[__name__], "deny_name")

    def test_split_name(self):
        person = def_method.Person()
        self.assertEqual(person.split_name(" "), ["John", "Smith"])

        with self.assertRaises(Exception) as cm:
            person.split_name("")
        self.assertEqual(cm.exception.args, ("Separator cannot be empty.",))

    def test_with(self):
        person = def_method.Person()
        same_person = person._with("Jane Doe", 25)
        self.assertEqual(person.get_name(), "Jane Doe")
        self.assertEqual(person.get_age(), 25)
        self.assertEqual(same_person.get_name(), person.get_name())
        self.assertEqual(same_person.get_age(), person.get_age())

    def test_get_age(self):
        self.assertEqual(def_method.Person().get_age(), 123)

    def test_get_birth_year(self):
        self.assertEqual(def_method.Person()._get_birth_year(2025), 1902)

    def test_with_first_last_name(self):
        person = def_method.Person()
        self.assertEqual(person._with_first_last_name("Jane", "Doe"), person)
        self.assertEqual(person.get_name(), "Jane Doe")


if __name__ == "__main__":
    unittest.main()
