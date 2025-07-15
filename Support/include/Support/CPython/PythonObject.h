//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_ENGINEAPI_PYTHON_PYTHONOBJECT_H
#define SDK_ENGINEAPI_PYTHON_PYTHONOBJECT_H

#include <Python.h>

extern "C" {
struct _object;
using PyObject = struct _object;
}

namespace M::CPython {

struct PythonObjectWrapper {
  PythonObjectWrapper() = default;
  PythonObjectWrapper(PyObject *ptr, bool takeOwnership = true);
  ~PythonObjectWrapper();
  PythonObjectWrapper(PythonObjectWrapper &&other) noexcept = default;
  PythonObjectWrapper &operator=(PythonObjectWrapper &&) noexcept = default;
  PythonObjectWrapper(const PythonObjectWrapper &) = delete;
  PythonObjectWrapper &operator=(const PythonObjectWrapper &) = delete;

  operator bool() const { return ptr != nullptr; }

  PyObject *ptr;
};

void freePythonObjectWrapper(void *ptr);

} // namespace M::CPython

#endif // SDK_ENGINEAPI_PYTHON_PYTHONOBJECT_H
