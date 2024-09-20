//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_ENGINEAPI_PYTHON_PYTHONOBJECT_H
#define SDK_ENGINEAPI_PYTHON_PYTHONOBJECT_H

#include <Python.h>

namespace M::core {

struct PythonObjectWrapper {
  PythonObjectWrapper(PyObject *ptr);
  ~PythonObjectWrapper();
  PythonObjectWrapper(PythonObjectWrapper &&other) noexcept = default;
  PythonObjectWrapper &operator=(const PythonObjectWrapper &other) = delete;
  PythonObjectWrapper &
  operator=(PythonObjectWrapper &&other) noexcept = default;

  PyObject *ptr;
};

void freePythonObjectWrapper(void *ptr);

} // namespace M::core

#endif // SDK_ENGINEAPI_PYTHON_PYTHONOBJECT_H
