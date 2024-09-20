//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "SDK/EngineAPI/python/PythonObject.h"
#include "llvm/Support/raw_ostream.h"

M::core::PythonObjectWrapper::PythonObjectWrapper(PyObject *ptr) : ptr(ptr) {
  if (ptr) {
    PyGILState_STATE state = PyGILState_Ensure();
    Py_INCREF(ptr);
    PyGILState_Release(state);
  }
}

M::core::PythonObjectWrapper::~PythonObjectWrapper() {
  if (ptr) {
    PyGILState_STATE state = PyGILState_Ensure();
    Py_DECREF(ptr);
    PyGILState_Release(state);
  }
}

void M::core::freePythonObjectWrapper(void *ptr) {
  PythonObjectWrapper *wrapper = static_cast<PythonObjectWrapper *>(ptr);
  delete wrapper;
};
