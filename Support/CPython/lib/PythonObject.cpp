//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "Support/CPython/PythonObject.h"

M::CPython::PythonObjectWrapper::PythonObjectWrapper(PyObject *ptr) : ptr(ptr) {
  if (ptr) {
    PyGILState_STATE state = PyGILState_Ensure();
    Py_INCREF(ptr);
    PyGILState_Release(state);
  }
}

M::CPython::PythonObjectWrapper::~PythonObjectWrapper() {
  if (ptr) {
    PyGILState_STATE state = PyGILState_Ensure();
    Py_DECREF(ptr);
    PyGILState_Release(state);
  }
}

void M::CPython::freePythonObjectWrapper(void *ptr) {
  PythonObjectWrapper *wrapper = static_cast<PythonObjectWrapper *>(ptr);
  delete wrapper;
};
