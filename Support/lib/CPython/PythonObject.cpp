//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "Support/CPython/PythonObject.h"
#include "Support/CPython/PythonGIL.h"
#include <Python.h>

namespace M::CPython {

PythonObjectWrapper::PythonObjectWrapper(PyObject *ptr, bool takeOwnership)
    : ptr(ptr) {
  if (ptr && takeOwnership) {
    PythonGIL lock;
    Py_INCREF(ptr);
  }
}

PythonObjectWrapper::~PythonObjectWrapper() {
  if (ptr) {
    PythonGIL lock;
    Py_DECREF(ptr);
  }
}

} // namespace M::CPython
