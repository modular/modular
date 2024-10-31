//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/CPython/PythonGIL.h"
#include "Support/CPython/PythonObject.h"
#include "llvm/ADT/StringRef.h"
#include <Python.h>
#include <optional>
#include <string>

namespace M::CPython {

std::optional<std::string> asString(PyObject *pystr) {
  if (!pystr || !PyUnicode_Check(pystr))
    return std::nullopt;

  PythonObjectWrapper utf8{PyUnicode_AsEncodedString(pystr, "utf-8", "strict")};
  if (utf8)
    return PyBytes_AsString(utf8.ptr);

  // error case
  PyErr_Clear();
  return std::nullopt;
}

PythonObjectWrapper stringToPythonObject(llvm::StringRef str) {
  // https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_FromStringAndSize
  return {PyUnicode_FromStringAndSize(str.data(), str.size()), false};
}

/// Get value from a Python dict
/// returns nullptr if dict is not a dict or key not found
PythonObjectWrapper getDictValue(PyObject *dict, llvm::StringRef keyStr) {
  if (!dict || !PyDict_Check(dict))
    return {};

  auto key = stringToPythonObject(keyStr);
  if (!key)
    return {};

  PyObject *value = PyDict_GetItem(dict, key.ptr);
  return {value};
}

std::optional<bool> getDictBool(PyObject *dict, llvm::StringRef keyStr) {
  PythonGIL gil;
  auto val = getDictValue(dict, keyStr);
  if (val && PyBool_Check(val.ptr))
    return val.ptr == Py_True;
  return std::nullopt;
}

bool setDictKeyValueString(PyObject *dict, llvm::StringRef key,
                           llvm::StringRef val) {
  PythonGIL gil;

  if (!dict || !PyDict_Check(dict))
    return false;

  auto pyKey = stringToPythonObject(key);
  if (!pyKey)
    return false;

  auto pyVal = stringToPythonObject(val);
  if (!pyVal)
    return false;

  if (PyDict_SetItem(dict, pyKey.ptr, pyVal.ptr) < 0)
    return false;

  return true;
}

bool setDictKeyValueBool(PyObject *dict, llvm::StringRef key, bool val) {
  PythonGIL gil;

  if (!dict || !PyDict_Check(dict))
    return false;

  // Convert the key to a Python string object
  PythonObjectWrapper pyKey = stringToPythonObject(key);
  if (!pyKey)
    return false;

  // Convert the bool value to a Python boolean object
  PyObject *pyVal = val ? Py_True : Py_False;

  // Set the key-value pair in the dictionary
  if (PyDict_SetItem(dict, pyKey.ptr, pyVal) < 0)
    return false;

  return true;
}

bool setDictKeyValueLong(PyObject *dict, llvm::StringRef key, long value) {
  PythonGIL gil;

  if (!dict || !PyDict_Check(dict))
    return false;

  // Convert the key to a Python string object
  PythonObjectWrapper pyKey = stringToPythonObject(key);
  if (!pyKey)
    return false;

  // Convert the int value to a Python integer object
  PythonObjectWrapper pyVal{PyLong_FromLong(value), false};
  if (!pyVal)
    return false;

  // Set the key-value pair in the dictionary
  if (PyDict_SetItem(dict, pyKey.ptr, pyVal.ptr) < 0)
    return false;

  return true;
}
} // namespace M::CPython
