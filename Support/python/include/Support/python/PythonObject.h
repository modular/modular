//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PYTHON_CPYTHON_PYTHONOBJECT_H
#define SUPPORT_PYTHON_CPYTHON_PYTHONOBJECT_H

#include <Python.h>

namespace M {

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

} // namespace M

#endif // SDK_ENGINEAPI_PYTHON_PYTHONOBJECT_H
