//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CPYTHON_UTIL_H
#define SUPPORT_CPYTHON_UTIL_H

#include "Support/CPython/PythonObject.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

/// Utility functions to provide Python GIL and refcount safe functions.
namespace M::CPython {

/// Convert a PyUnicode object to a UTF-8 string
std::optional<std::string> asString(PyObject *pystr);

/// Create a wrapped PyUnicode object from a UTF-8 encoded string
PythonObjectWrapper stringToPythonObject(llvm::StringRef str);

/// Get wrapped Python value from key:string in a Python dict
/// returns empty wrapper if dict is not a dict or key not found
PythonObjectWrapper getDictValue(PyObject *dict, llvm::StringRef keyStr);

/// Get a PyBool value from a key::string
std::optional<bool> getDictBool(PyObject *dict, llvm::StringRef keyStr);

/// Get dict[key::string,...] value by type
/// returns nullopt if key not found
/// returns nullopt if not able to convert to type
template <typename T>
std::optional<T> getDictValueAs(PyObject *dict, llvm::StringRef keyStr);

/// Get dict[key::string, string] value as a std::string
template <>
std::optional<std::string> getDictValueAs<std::string>(PyObject *dict,
                                                       llvm::StringRef key) {
  if (auto wrapper = getDictValue(dict, key))
    return asString(wrapper.ptr);
  return std::nullopt;
}

/// Get dict[key::string, bool] value as a bool
/// returns nullopt if key not found or value is not a PyBool
template <>
std::optional<bool> getDictValueAs<bool>(PyObject *dict, llvm::StringRef key) {
  return getDictBool(dict, key);
}

/// Set a string value in a dict[key:string, key:string]
bool setDictKeyValueString(PyObject *dict, llvm::StringRef key,
                           llvm::StringRef val);

/// Set a boolean value in a dict[key:string, value: bool]
bool setDictKeyValueBool(PyObject *dict, llvm::StringRef key, bool val);

/// Set an integer value in a dict[key:string, value:int]
bool setDictKeyValueLong(PyObject *dict, llvm::StringRef key, long val);

} // namespace M::CPython

#endif // SUPPORT_CPYTHON_UTIL_H
