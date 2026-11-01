//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef MLRT_CORE_MOJOVALUE_H
#define MLRT_CORE_MOJOVALUE_H

#include <cstddef>

namespace M {

/// `MojoValue` represents a runtime mojo object.
/// It owns the object (and release it when the MojoValue is
/// destroyed). This class is used to move mojo objects between the C++ /
/// Mojo world. We can't do much with the object in C++ (besides moving it to
/// mojo land or releasing it).
///
/// Warning: The MojoValue has a pointer to the mojo destructor function, and
/// the user must ensure that the function pointer is still valid. (for example
/// if the mojo library is dynamically loaded).
class MojoValue {
public:
  using DestructorFn = void (*)(void *);

  /// Create an empty MojoValue.
  MojoValue() : MojoValue(nullptr, nullptr) {}

  /// Create a new mojo object.
  /// MojoValue owns the object and call `destructor` when it get destroyed.
  /// The underlying object isn't initialized.
  /// It's the user responsibility to initialize it properly.
  /// eg: `initialize_pointee_move(mojoValue.getData(), value^)`.
  MojoValue(size_t size, size_t align, DestructorFn destructor);

  /// Create a new mojo object.
  /// `data` points to the mojo struct.
  /// MojoValue owns the object and call `destructor` when it get destroyed.
  /// `data` is also owned and released by MojoValue (it must be allocated with
  /// malloc).
  MojoValue(void *data, DestructorFn destructor)
      : data(data), destructor(destructor), owning(true), shouldFree(true) {}

  /// Create a reference to an existing Mojo object. The object must outlive the
  /// lifetime of this MojoValue.
  static MojoValue createRef(void *data) {
    return MojoValue(data, nullptr, /*owning=*/false, /*shouldFree=*/false);
  }

  /// Allows for values to be allocated within mojo
  /// and destroyed at end of life without forcing an allocation for a
  /// T to be done from C++.
  /// `data` is, itself, a `*T`/`UnsafePointer[T]` with a backing allocation
  /// originating in Mojo. It is expected that `destructor` free the backing
  /// allocation.
  static MojoValue createOwningRef(void *data, DestructorFn destructor) {
    return MojoValue(data, destructor, /*owning=*/true, /*shouldFree=*/false);
  }

  ~MojoValue();

  MojoValue(MojoValue &&rhs);
  MojoValue &operator=(MojoValue &&rhs);

  /// Allocate the heap buffer needed for the mojo value.
  /// This is exposed to be able to call it from the mojo API.
  static void *allocateBuffer(size_t size, size_t align);

  /// Free the heap buffer needed for the mojo value.
  /// This is exposed to be able to call it from the mojo API.
  static void freeBuffer(void *ptr);

  /// Returns a pointer to the mojo object.
  void *getData() const { return data; }

  bool isOwned() const { return owning; }

  /// Swap `*this` and `rhs` objects.
  /// Required for the AsyncValue API.
  void swap(MojoValue &rhs);

private:
  MojoValue(void *data, DestructorFn destructor, bool owning, bool shouldFree)
      : data(data), destructor(destructor), owning(owning),
        shouldFree(shouldFree) {}

  void *data = nullptr;
  DestructorFn destructor = nullptr;
  /// Does the mojo value own this value. Data not freed and destructor not run
  /// if not owned.
  bool owning = false;
  /// Should the mojovalue free its pointer.
  bool shouldFree = false;
};

} // namespace M

#endif // MLRT_CORE_MOJOVALUE_H
