# Tests functionality that depends on the python version
# RUN: %mojo %s

from python.bindings import _get_type_name
from python import PythonObject
from testing import assert_equal
def test_get_type_name() -> None:
    """Test the _get_type_name function."""
    var str_obj = PythonObject("hello")
    assert_equal(_get_type_name(str_obj).__str__(), "str")
    var int_obj = PythonObject(32)
    assert_equal(_get_type_name(int_obj).__str__(), "int")

def main():
    test_get_type_name()
