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
  if (!PyDict_Check(dict))
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

void setDictKeyValue(PyObject *dict, llvm::StringRef key, llvm::StringRef val) {
  auto pyKey = stringToPythonObject(key);
  auto pyVal = stringToPythonObject(val);
  // FIXME: Error checking
  PyDict_SetItem(dict, pyKey.ptr, pyVal.ptr);
}
} // namespace M::CPython
