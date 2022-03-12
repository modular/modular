//===- LLCL/Support/RCRef.h -----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_SUPPORT_RCREF_H
#define LLCL_SUPPORT_RCREF_H

#include <atomic>

namespace LLCL {

/// This is a smart pointer that keeps the specified reference counted value
/// around.  It is move-only to avoid accidental copies, but it can be copied
/// explicitly.
template <typename T>
class RCRef {
public:
  RCRef() : pointer(nullptr) {}
  RCRef(RCRef &&other) : pointer(other.pointer) { other.pointer = nullptr; }

  /// This constructor forms a reference to the specified pointer, increasing
  /// the underlying reference count by 1.
  static RCRef copy(T *pointer) {
    if (pointer)
      pointer->addRef();
    return take(pointer);
  }

  /// This constructor forms a reference to the specified pointer, taking
  /// ownership it, and thus not increasing the reference count.
  static RCRef take(T *pointer) {
    RCRef<T> ref;
    ref.pointer = pointer;
    return ref;
  }

  /// Create an instance of T with the specified constructor arguments and
  /// return it as an RCRef.
  template <typename... Args>
  static RCRef create(Args &&...args) {
    return take(new T(std::forward<Args>(args)...));
  }

  /// Support implicit conversion from RCRef<Derived> to RCRef<Base>.
  template <typename U,
            typename = std::enable_if_t<std::is_base_of<T, U>::value>>
  RCRef(RCRef<U> &&u) : pointer(u.release()) {}

  ~RCRef() {
    if (pointer)
      pointer->dropRef();
  }

  RCRef &operator=(RCRef &&other) {
    if (pointer)
      pointer->dropRef();
    pointer = other.pointer;
    other.pointer = nullptr;
    return *this;
  }

  /// Manually drop the reference in this RCRef, setting it to null.
  void reset() {
    if (pointer)
      pointer->dropRef();
    pointer = nullptr;
  }

  /// Take ownership of the underlying pointer away from the RCRef and reset it
  /// to null.
  T *release() {
    T *tmp = pointer;
    pointer = nullptr;
    return tmp;
  }

  T &operator*() const {
    assert(pointer && "null RCRef");
    return *pointer;
  }

  T *operator->() const {
    assert(pointer && "null RCRef");
    return pointer;
  }

  /// Return a raw pointer.
  T *getPointer() const { return pointer; }

  /// Make an explicit copy of this RCRef, increasing the refcount by one.
  RCRef<T> copy() const { return RCRef<T>::copy(pointer); }

  /// Test for null.
  explicit operator bool() const { return pointer != nullptr; }

  void swap(RCRef &other) {
    using std::swap;
    swap(pointer, other.pointer);
  }

private:
  // Not implicity copyable, use the copy() method for an explicit copy of
  // this reference.
  RCRef(const RCRef &) = delete;
  RCRef &operator=(const RCRef &) = delete;

  T *pointer;
};

// These global functions help make type inference work better.

/// Forms a reference to the specified pointer, increasing the underlying
/// reference count by 1.
template <typename T>
inline RCRef<T> copyRCRef(T *ptr) {
  return RCRef<T>::copy(ptr);
}

/// Form a reference to the specified pointer, taking ownership it, and thus not
/// increasing the reference count.
template <typename T>
inline RCRef<T> takeRCRef(T *ptr) {
  return RCRef<T>::take(ptr);
}

// For ADL style swap.
template <typename T>
inline void swap(RCRef<T> &a, RCRef<T> &b) {
  a.swap(b);
}

} // namespace LLCL

#endif // LLCL_SUPPORT_RCREF_H
