//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "Support/python/PythonObject.h"

M::PythonObjectWrapper::PythonObjectWrapper(PyObject *ptr) : ptr(ptr) {
  if (ptr) {
    PyGILState_STATE state = PyGILState_Ensure();
    Py_INCREF(ptr);
    PyGILState_Release(state);
  }
}

M::PythonObjectWrapper::~PythonObjectWrapper() {
  if (ptr) {
    PyGILState_STATE state = PyGILState_Ensure();
    Py_DECREF(ptr);
    PyGILState_Release(state);
  }
}

void M::freePythonObjectWrapper(void *ptr) {
  PythonObjectWrapper *wrapper = static_cast<PythonObjectWrapper *>(ptr);
  delete wrapper;
};
