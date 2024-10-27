//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CPYTHON_PYTHONOBJECT_H
#define SUPPORT_CPYTHON_PYTHONOBJECT_H

#include <Python.h>

namespace M::CPython {

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

} // namespace M::CPython

#endif // SUPPORT_CPYTHON_PYTHONOBJECT_H
