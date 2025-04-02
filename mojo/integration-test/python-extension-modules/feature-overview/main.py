# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %bare-mojo build %S/mojo_module.mojo --emit shared-lib -o mojo_module.so
# RUN: python3 %s

import sys
import unittest

# Put the current directory (containing .so) on the Python module lookup path.
sys.path.insert(0, "")

# Imports from 'mojo_module.so'
import mojo_module as feature_overview


class TestMojoPythonInterop(unittest.TestCase):
    def test_case_return_arg_tuple(self):
        result = feature_overview.case_return_arg_tuple(
            1, 2, "three", ["four", "four.B"]
        )

        self.assertEqual(result, (1, 2, "three", ["four", "four.B"]))

    def test_case_raise_empty_error(self):
        with self.assertRaises(ValueError) as cm:
            feature_overview.case_raise_empty_error()

        self.assertEqual(cm.exception.args, ())

    def test_case_raise_string_error(self):
        with self.assertRaises(ValueError) as cm:
            feature_overview.case_raise_string_error()

        self.assertEqual(cm.exception.args, ("sample value error",))

    def test_case_mojo_raise(self):
        with self.assertRaises(Exception) as cm:
            feature_overview.case_mojo_raise()

        self.assertEqual(cm.exception.args, ("Mojo error",))

    def test_case_create_mojo_type_instance(self):
        person = feature_overview.Person()

        self.assertEqual(type(person).__name__, "Person")

        self.assertEqual(person.name(), "John Smith")

        self.assertEqual(repr(person), "Person('John Smith', 123)")

        # Test that an error is raised if passing any arguments to the initalizer

        with self.assertRaises(ValueError) as cm:
            person = feature_overview.Person("John")

        self.assertEqual(
            cm.exception.args,
            (
                (
                    "unexpected arguments passed to default initializer"
                    " function of wrapped Mojo type"
                ),
            ),
        )

    def test_case_create_mojo_object_in_mojo(self):
        # Returns a new Mojo 'String' object, not derived from
        # any of the arguments. This requires creating a PythonObject from
        # within Mojo code.
        string = feature_overview.create_string()

        self.assertEqual(repr(string), "'Hello'")

    def test_case_mutate_wrapped_object(self):
        mojo_int = feature_overview.Int()
        self.assertEqual(repr(mojo_int), "0")

        feature_overview.incr_int(mojo_int)
        self.assertEqual(repr(mojo_int), "1")

        feature_overview.incr_int(mojo_int)
        self.assertEqual(repr(mojo_int), "2")

        # --------------------------------
        # Test passing the wrong arguments
        # --------------------------------

        #
        # Too few arguments
        #

        with self.assertRaises(Exception) as cm:
            feature_overview.incr_int()

        self.assertEqual(
            cm.exception.args,
            ("TypeError: incr_int() missing 1 required positional argument",),
        )

        #
        # Too many arguments
        #

        with self.assertRaises(Exception) as cm:
            feature_overview.incr_int(1, 2, 3)

        self.assertEqual(
            cm.exception.args,
            (
                (
                    "TypeError: incr_int() takes 1 positional argument but 3"
                    " were given"
                ),
            ),
        )

        #
        # Wrong type of argument
        #

        with self.assertRaises(Exception) as cm:
            feature_overview.incr_int("string")

        self.assertEqual(
            cm.exception.args,
            (
                (
                    "TypeError: incr_int() expected Mojo 'Int' type argument,"
                    " got 'str'"
                ),
            ),
        )

    def test_case_mojo_value_convert_from_python(self):
        mojo_int = feature_overview.Int()
        self.assertEqual(repr(mojo_int), "0")

        feature_overview.add_to_int(mojo_int, 5)
        self.assertEqual(repr(mojo_int), "5")

        feature_overview.add_to_int(mojo_int, 3)
        self.assertEqual(repr(mojo_int), "8")

        #
        # Wrong type of argument
        #

        with self.assertRaises(Exception) as cm:
            feature_overview.add_to_int(mojo_int, "foo")

        self.assertEqual(
            cm.exception.args,
            (
                (
                    "TypeError: add_to_int() expected argument at position"
                    " 1 to be instance of (or convertible to) Mojo 'Int';"
                    " got 'str'."
                    " (Note: attempted conversion failed due to: an integer"
                    " is required)"
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
