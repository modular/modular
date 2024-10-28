//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include <Python.h>

namespace M::CPython {
/// Provides simple C++ RAII scoped Python GIL management
struct PythonGIL {
  /// RAII Constructor to recursively acquire the Python GIL
  /// acquire argument allows run-time disabling
  PythonGIL(bool acquire = true) noexcept;

  /// RAII destructor automatically releases the Python GIL
  ~PythonGIL() noexcept;

  /// allow move
  PythonGIL(PythonGIL &&other) noexcept = default;
  PythonGIL &operator=(PythonGIL &&other) noexcept = default;

  /// disallow copy
  PythonGIL(PythonGIL &copy) = delete;
  PythonGIL &operator=(PythonGIL &copy) = delete;

private:
  // PyGilState_STATE enum only has 2 states (LOCKED and UNLOCKED),
  // but we need three to store if no lock was acquired at all
  enum class State : int {
    LOCKED,
    UNLOCKED,
    NOT_ACQUIRED,
  };
  State state = State::NOT_ACQUIRED;
};

// inline header implementations to allow for compiler to inline

inline PythonGIL::PythonGIL(bool acquire) noexcept {
  // state is already initialized to State::NOT_ACQUIRED;
  if (acquire) {
    PyGILState_STATE pyState = PyGILState_Ensure();

    switch (pyState) {
    case PyGILState_LOCKED:
      state = State::LOCKED;
      break;
    case PyGILState_UNLOCKED:
      state = State::UNLOCKED;
      break;
    }
  }
}

inline PythonGIL::~PythonGIL() noexcept {
  switch (state) {
  case State::LOCKED:
    PyGILState_Release(PyGILState_LOCKED);
    break;
  case State::UNLOCKED:
    PyGILState_Release(PyGILState_UNLOCKED);
    break;
  case State::NOT_ACQUIRED:
    // do nothing
    break;
  }
}

} // namespace M::CPython
