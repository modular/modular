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

#ifndef MLRT_GRAPHRT_MEF_MEFATTR_H
#define MLRT_GRAPHRT_MEF_MEFATTR_H

#include "MLRT/GraphRT/MEF/MEFEncoding.h"
#include "MLRT/GraphRT/MEF/MEFTypes.h"
#include "Support/LLVMForwardDecls.h"

#include <cstdint>
#include <type_traits>

namespace M {

//===----------------------------------------------------------------------===//
// MEFDecoder
//===----------------------------------------------------------------------===//

/// The MEFDecoder class provides a simple deserialization mechanism for
/// specifying how non-trivial types can be decoded from MEF attributes.  It may
/// be specialized for user-defined types that are encoded into a MEF file.
///
/// To implement this, provide a static `decode` method, which is invoked on a
/// low-level MEF encoded type (like int8_t) into a user type (or fail with an
/// error if the encoded value is invalid).
///
/// This can be used to provide checking for serialized enums and other similar
/// values.
///
template <typename UserType>
struct MEFDecoder; // {
//   static ErrorOr<UserType> decode(NativeMEFType value);
// };

/// Define trivial decoders for all our native types.
#define DEFINE_BUILTIN_TYPE_DECODER(TYPE)                                      \
  template <>                                                                  \
  struct MEFDecoder<TYPE> {                                                    \
    static ErrorOr<TYPE> decode(TYPE value) { return value; }                  \
  }

DEFINE_BUILTIN_TYPE_DECODER(bool);
DEFINE_BUILTIN_TYPE_DECODER(int8_t);
DEFINE_BUILTIN_TYPE_DECODER(uint8_t);
DEFINE_BUILTIN_TYPE_DECODER(int16_t);
DEFINE_BUILTIN_TYPE_DECODER(uint16_t);
DEFINE_BUILTIN_TYPE_DECODER(int32_t);
DEFINE_BUILTIN_TYPE_DECODER(uint32_t);
DEFINE_BUILTIN_TYPE_DECODER(int64_t);
DEFINE_BUILTIN_TYPE_DECODER(uint64_t);
DEFINE_BUILTIN_TYPE_DECODER(float);
DEFINE_BUILTIN_TYPE_DECODER(double);
DEFINE_BUILTIN_TYPE_DECODER(StringRef);
#undef DEFINE_BUILTIN_TYPE_DECODER

/// ArrayRef's are generic over element type, but are also noop conversions.
template <typename EltType>
struct MEFDecoder<ArrayRef<EltType>> {
  static ErrorOr<ArrayRef<EltType>> decode(ArrayRef<EltType> value) {
    return value;
  }
};

template <>
struct MEFDecoder<MEF::TypeKind> {
  static ErrorOr<MEF::TypeKind> decode(MEF::TypeKind value) { return value; }
};

#ifndef __clang__
/// GCC complains about matching SmallVector without the Size template
/// parameter, and CLANG complains about ambiguous partial specialization.
///
/// TODO: Can we implement this function in terms of
/// MEFDecoder<SmallVector<EltType, Size>>?
template <typename EltType>
struct MEFDecoder<SmallVector<EltType>> {
  static ErrorOr<SmallVector<EltType>> decode(MEFListAttr value) {
    SmallVector<EltType> result;
    result.reserve(value.size());
    for (auto element : value) {
      EltType resultElt{};
      if (auto err = element.decode(&resultElt))
        return err.takeError();
      result.push_back(resultElt);
    }
    return result;
  }
};
#endif // __clang__

//===----------------------------------------------------------------------===//
// MEFAttr and specializations
//===----------------------------------------------------------------------===//

/// This CRTP class is used to provide common interfaces across the various
/// MEFAttr implementations.
template <typename ConcreteType, typename StoredType>
class MEFAttrCommon {
public:
  MEFAttrCommon(StoredType init, ArrayRef<uint8_t> attrSection)
      : value(init), attrSection(attrSection) {}

  using ValueType = typename std::remove_reference<StoredType>::type;

  const StoredType &get() const { return value; }
  const ValueType *operator->() const { return &value; }
  const StoredType &operator*() const { return value; }

  ArrayRef<uint8_t> getAttrSection() const { return attrSection; }

  /// Given a pointer to the start of attribute value, return the attribute
  /// kind.
  static MEF::AttributeKind getAttrKind(const uint8_t *value) {
    // The attribute kind is encoded in the byte before the attribute value.
    return static_cast<MEF::AttributeKind>(*(value - 1));
  }

  /// Check that the dynamic attribute kind encoded into the MEF file matches
  /// the C++ type for this.
  static inline void checkAttributeKind(const uint8_t *value,
                                        ArrayRef<uint8_t> attrSection) {
    assert(ConcreteType::matchesAttrKind(value, attrSection.end()) &&
           "attribute passed to primitive has unexpected type;"
           " mismatch between mlir syntax and primitive definition");
  }

private:
  StoredType value;
  /// This represent the beginning and end pointers collected into an ArrayRef
  /// to represent the range of bytes making up the whole Attribute Section.
  /// This is useful since we want to check the pointer for the attribute is:
  ///    1. Past the beginning of the Attribute Section in the PrimitiveRegistry
  ///    2. Past the end of the Attribute Section downstream to MEFType for
  ///    decoding so we know the bounds.
  ArrayRef<uint8_t> attrSection;
};

/// Primitives use this to declare that they have an attribute input with the
/// specified C++ type.
template <typename T>
class MEFAttr : public MEFAttrCommon<MEFAttr<T>, const T &> {
public:
  using BaseClass = MEFAttrCommon<MEFAttr<T>, const T &>;

  explicit MEFAttr(const uint8_t *value, ArrayRef<uint8_t> attrSection)
      : BaseClass(*reinterpret_cast<const T *>(value), attrSection) {
    // Check that the C++ type matches the encoded type for the attribute.
    static_assert(MEF::AttrKindForCXXType<T>::kind !=
                      MEF::AttributeKind::kInvalid &&
                  "C++ type passed to MEFAttr isn't supported");
    assert(value >= attrSection.begin() &&
           "MEF datum not within bounds of the MEF Attribute Section");
    this->checkAttributeKind(value, attrSection);
  }

  static bool matchesAttrKind(const uint8_t *pointer,
                              const uint8_t *attrSectionEnd) {
    return BaseClass::getAttrKind(pointer) == MEF::AttrKindForCXXType<T>::kind;
  }
};

/// Given a pointer to the start of a MEF array or string attribute, return the
/// length of array or string by decoding the prefix encoded VBR.
size_t decodeMEFArrayLength(const uint8_t *dataPtr);

/// String attributes have a special encoding that includes their length,
/// allowing projection into a StringRef.
template <>
class MEFAttr<StringRef> : public MEFAttrCommon<MEFAttr<StringRef>, StringRef> {
public:
  using BaseClass = MEFAttrCommon<MEFAttr<StringRef>, StringRef>;

  explicit MEFAttr(const uint8_t *pointer, ArrayRef<uint8_t> attrSection)
      : BaseClass(StringRef(reinterpret_cast<const char *>(pointer),
                            decodeMEFArrayLength(pointer)),
                  attrSection) {
    assert(pointer >= attrSection.begin() &&
           "MEF datum not within bounds of the MEF Attribute Section");
    this->checkAttributeKind(pointer, attrSection);
  }

  static bool matchesAttrKind(const uint8_t *pointer,
                              const uint8_t *attrSectionEnd) {
    return BaseClass::getAttrKind(pointer) == MEF::AttributeKind::kString;
  }
};

/// Array attributes have a special encoding that includes their length,
/// allowing projection into a ArrayRef.
template <typename T>
class MEFAttr<ArrayRef<T>>
    : public MEFAttrCommon<MEFAttr<ArrayRef<T>>, ArrayRef<T>> {
public:
  using BaseClass = MEFAttrCommon<MEFAttr<ArrayRef<T>>, ArrayRef<T>>;

  explicit MEFAttr(const uint8_t *pointer, ArrayRef<uint8_t> attrSection)
      : BaseClass(ArrayRef<T>(reinterpret_cast<const T *>(pointer),
                              decodeMEFArrayLength(pointer)),
                  attrSection) {
    assert(pointer >= attrSection.begin() &&
           "MEF datum not within bounds of the MEF Attribute Section");
    this->checkAttributeKind(pointer, attrSection);
  }

  static bool matchesAttrKind(const uint8_t *pointer,
                              const uint8_t *attrSectionEnd) {
    // If the element type matches, this matches.
    MEF::AttributeKind kind = BaseClass::getAttrKind(pointer);
    return kind == MEF::getArrayOf(MEF::AttrKindForCXXType<T>::kind);
  }

  size_t size() const { return this->get().size(); }
  const T &operator[](size_t i) const { return this->get()[i]; }
};

/// Symbols are essentially strings, but we just can't call them strings.
class MEFSymbolRef {
public:
  MEFSymbolRef(StringRef value) : name(value) {}

  StringRef name;
};

template <>
class MEFAttr<MEFSymbolRef>
    : public MEFAttrCommon<MEFAttr<MEFSymbolRef>, MEFSymbolRef> {
public:
  using BaseClass = MEFAttrCommon<MEFAttr<MEFSymbolRef>, MEFSymbolRef>;

  explicit MEFAttr(const uint8_t *pointer, ArrayRef<uint8_t> attrSection)
      : BaseClass(StringRef(reinterpret_cast<const char *>(pointer),
                            decodeMEFArrayLength(pointer)),
                  attrSection) {
    assert(pointer >= attrSection.begin() &&
           "MEF datum not within bounds of the MEF Attribute Section");
    this->checkAttributeKind(pointer, attrSection);
  }

  static bool matchesAttrKind(const uint8_t *pointer,
                              const uint8_t *attrSectionEnd) {
    return BaseClass::getAttrKind(pointer) == MEF::AttributeKind::kSymbolRef;
  }
};

//===----------------------------------------------------------------------===//
// MEFAttr<*MEFType> support
//===----------------------------------------------------------------------===//

namespace Detail {
struct TypeAttrEmptyStruct {};
} // namespace Detail

template <typename MEFTypeClass>
class MEFTypeAttrCommon
    : public MEFAttrCommon<MEFAttr<MEFTypeClass>, Detail::TypeAttrEmptyStruct>,
      public MEFTypeClass {
public:
  using BaseClass =
      MEFAttrCommon<MEFAttr<MEFTypeClass>, Detail::TypeAttrEmptyStruct>;

  explicit MEFTypeAttrCommon(const uint8_t *pointer,
                             ArrayRef<uint8_t> attrSection)
      : BaseClass(Detail::TypeAttrEmptyStruct(), attrSection),
        MEFTypeClass(pointer, attrSection.end()) {
    assert(pointer >= attrSection.begin() &&
           "MEF datum not within bounds of the MEF Attribute Section");
    this->checkAttributeKind(pointer, attrSection);
  }

  static bool matchesAttrKind(const uint8_t *pointer,
                              const uint8_t *attrSectionEnd) {
    auto dynType = MEFType(pointer, attrSectionEnd);
    return BaseClass::getAttrKind(pointer) == MEF::AttributeKind::kType &&
           MEFTypeClass::classof(&dynType);
  }
};

template <>
class MEFAttr<MEFType> : public MEFTypeAttrCommon<MEFType> {
  using MEFTypeAttrCommon::MEFTypeAttrCommon;
};

template <>
class MEFAttr<MEFFunctionType> : public MEFTypeAttrCommon<MEFFunctionType> {
  using MEFTypeAttrCommon::MEFTypeAttrCommon;
};

template <>
class MEFAttr<MEFTensorType> : public MEFTypeAttrCommon<MEFTensorType> {
  using MEFTypeAttrCommon::MEFTypeAttrCommon;
};

template <>
class MEFAttr<MEFQuantizedType> : public MEFTypeAttrCommon<MEFQuantizedType> {
  using MEFTypeAttrCommon::MEFTypeAttrCommon;
};

template <>
class MEFAttr<MEFQuantizedPerAxisType>
    : public MEFTypeAttrCommon<MEFQuantizedPerAxisType> {
  using MEFTypeAttrCommon::MEFTypeAttrCommon;
};

//===----------------------------------------------------------------------===//
// MEFDynamicAttr
//===----------------------------------------------------------------------===//

namespace Detail {
class Dynamic {};

/// This pair of structs provides a check to see if a class T is a type wrapped
/// in a MEFAttr. If T is already a MEFAttr, it exposes the inner type, and
/// unconditionally provides a return type that can be used by dyn_cast and
/// family.
template <class T>
struct MEFAttrDynCastRet {
  using InnerType = T;
  using Type = MEFAttr<T>;

  static InnerType getValue(const Type &value) { return value.get(); }
};

template <class T>
struct MEFAttrDynCastRet<MEFAttr<T>> {
  using InnerType = T;
  using Type = MEFAttr<T>;

  static Type getValue(MEFAttr<T> value) { return value; }
};

/// Convenience typedef to simplify declarations.
template <typename T>
using MEFAttrDynCastRetType = typename MEFAttrDynCastRet<T>::Type;
} // namespace Detail

/// Dynamic attributes allow checked casting to other attribute kinds.
template <>
class MEFAttr<Detail::Dynamic>
    : public MEFAttrCommon<MEFAttr<Detail::Dynamic>, const uint8_t *> {
public:
  using BaseClass = MEFAttrCommon<MEFAttr<Detail::Dynamic>, const uint8_t *>;

  explicit MEFAttr(const uint8_t *pointer, ArrayRef<uint8_t> attrSection)
      : BaseClass(pointer, attrSection) {
    assert(pointer >= attrSection.begin() &&
           "MEF datum not within bounds of the MEF Attribute Section");
  }

  /// Return the AttributeKind for this attribute.
  MEF::AttributeKind getAttrKind() const {
    return MEFAttrCommon::getAttrKind(get());
  }

  template <typename T>
  auto cast() const {
    auto result = dyn_cast<T>(this);
    assert(result && "dynamic attribute is not of this kind");
    return *result;
  }

  /// Parse the MEFDynamicAttr as a specified C++ type, rejecting it with an
  /// error if not present or the wrong type.
  template <typename ValueType>
  ErrorOrSuccess decode(ValueType *result) const {
    using EncodedType = typename llvm::function_traits<
        decltype(&M::MEFDecoder<ValueType>::decode)>::template arg_t<0>;
    using InnerType =
        typename Detail::MEFAttrDynCastRet<EncodedType>::InnerType;

    auto rawAttrValue = dyn_cast<InnerType>(this);
    if (!rawAttrValue)
      return Error("type mismatch, have " + MEF::getAsString(getAttrKind()) +
                   " but caller expected " + llvm::getTypeName<ValueType>());

    // If this is a C type like int8_t then look through the attr, otherwise
    // leave it intact.
    auto valueToDecode =
        Detail::MEFAttrDynCastRet<EncodedType>::getValue(*rawAttrValue);
    auto decodedValueOr = MEFDecoder<ValueType>::decode(valueToDecode);
    if (decodedValueOr.isError())
      return decodedValueOr.takeError();
    *result = std::move(*decodedValueOr);
    return success();
  }

  /// Dynamic attributes are compatible with everything.
  static bool matchesAttrKind(const uint8_t *pointer,
                              const uint8_t *attrSectionEnd) {
    return true;
  }
};

using MEFDynamicAttr = MEFAttr<Detail::Dynamic>;

// Since MEFDynamicAttr can be constructed from a nullptr, the template
// specializations in llvm/Support/Casting.h fail to apply. As such, we
// have to explicitly define the equals and not equals operators for
// MEFDynamicAttr.
inline bool operator==(MEFDynamicAttr lhs, MEFDynamicAttr rhs) {
  return lhs.get() == rhs.get();
}

inline bool operator!=(MEFDynamicAttr lhs, MEFDynamicAttr rhs) {
  return !(lhs == rhs);
}

//===----------------------------------------------------------------------===//
// MEFListAttr
//===----------------------------------------------------------------------===//

namespace Detail {
class List {};
} // namespace Detail

/// List of attributes.
template <>
class MEFAttr<Detail::List>
    : public MEFAttrCommon<MEFAttr<Detail::List>, Detail::List> {
public:
  using BaseClass = MEFAttrCommon<MEFAttr<Detail::List>, Detail::List>;
  explicit MEFAttr(const uint8_t *pointer, ArrayRef<uint8_t> attrSection)
      : BaseClass(Detail::List{}, attrSection),
        data(pointer, decodeMEFArrayLength(pointer)) {
    assert(pointer >= attrSection.begin() &&
           "MEF datum not within bounds of the MEF Attribute Section");
    this->checkAttributeKind(pointer, attrSection);
  }

  static bool matchesAttrKind(const uint8_t *pointer,
                              const uint8_t *attrSectionEnd) {
    return BaseClass::getAttrKind(pointer) == MEF::AttributeKind::kList;
  }

  /// Provides an iterator on the List. Because we don't want to allocate
  /// in this data structure, we parse each attribute's name and value every
  /// time the iterator is accessed.
  class Iterator
      : public llvm::iterator_facade_base<Iterator, std::input_iterator_tag,
                                          MEFDynamicAttr> {
  public:
    using Base = llvm::iterator_facade_base<Iterator, std::input_iterator_tag,
                                            MEFDynamicAttr>;

    Iterator(const uint8_t *ptr, size_t iterIdx, ArrayRef<uint8_t> attrSection)
        : basePtr(ptr), ptr(const_cast<uint8_t *>(ptr)), iterIdx(iterIdx),
          attrSection(attrSection) {}

    /// Increment the iterator. Decrementing is sufficiently difficult that it's
    /// left as an exercise to the person who wants to use it.
    Iterator &operator+=(Base::difference_type n);

    /// Compare Iterator objects.
    bool operator==(const Iterator &rhs) const {
      assert(basePtr == rhs.basePtr &&
             "Comparing iterators from different lists!");
      return iterIdx == rhs.iterIdx;
    }

    /// Parse out and return the name and value contained in this iterator.
    MEFDynamicAttr operator*() const;

  private:
    const uint8_t *basePtr;
    uint8_t *ptr;
    size_t iterIdx;
    ArrayRef<uint8_t> attrSection;
  };

  /// Range access into the List object.
  Iterator begin() const { return {data.begin(), 0, getAttrSection()}; }
  Iterator end() const {
    return Iterator{data.begin(), data.size(), getAttrSection()};
  }

  /// Return the size of the List in number of elements.
  size_t size() const { return data.size(); }

  /// Return true if there are no attributes in this list.
  bool empty() const { return data.empty(); }

  /// Parse this list attribute into the specified type (typically a
  /// SmallVector), rejecting it with an error if not present or the wrong type.
  template <typename ValueType>
  ErrorOrSuccess decode(ValueType *result) const {
    return MEFDynamicAttr(data.data(), getAttrSection())
        .decode<ValueType>(result);
  }

private:
  ArrayRef<uint8_t> data;
};

using MEFListAttr = MEFAttr<Detail::List>;

//===----------------------------------------------------------------------===//
// MEFDictionaryAttr
//===----------------------------------------------------------------------===//

namespace Detail {
class Dictionary {};
} // namespace Detail

/// Dictionaries are lists of name, value pairs.
template <>
class MEFAttr<Detail::Dictionary>
    : public MEFAttrCommon<MEFAttr<Detail::Dictionary>, Detail::Dictionary> {
public:
  using BaseClass =
      MEFAttrCommon<MEFAttr<Detail::Dictionary>, Detail::Dictionary>;
  explicit MEFAttr(const uint8_t *pointer, ArrayRef<uint8_t> attrSection)
      : BaseClass(Detail::Dictionary{}, attrSection),
        data(pointer, decodeMEFArrayLength(pointer)) {
    assert(pointer >= attrSection.begin() &&
           "MEF datum not within bounds of the MEF Attribute Section");
    this->checkAttributeKind(pointer, attrSection);
  }

  static bool matchesAttrKind(const uint8_t *pointer,
                              const uint8_t *attrSectionEnd) {
    return BaseClass::getAttrKind(pointer) == MEF::AttributeKind::kDictionary;
  }

  /// Provides something for the Iterator to dereference. Conceptually, this is
  /// a single element in the MEF Dictionary attribute - it consists of a base
  /// pointer and offsets to both the name and value of this element.
  class Element {
  public:
    Element(const uint8_t *basePtr, size_t nameOffset, size_t attrOffset,
            ArrayRef<uint8_t> attrSection)
        : basePtr(basePtr), nameOffset(nameOffset), attrOffset(attrOffset),
          attrSection(attrSection) {}

    /// Get the name of this element.
    StringRef getName() const {
      return *MEFAttr<StringRef>(basePtr - nameOffset, attrSection);
    }

    /// Get the value of this element.
    MEFDynamicAttr getValue() const {
      return MEFDynamicAttr(basePtr - attrOffset, attrSection);
    }

    /// Templated convenience function to allow the user to get the value as a
    /// certain type. Asserts that T matches the actual attribute stored.
    template <typename T>
    MEFAttr<T> getValue() const {
      return MEFAttr<T>(basePtr - attrOffset, attrSection);
    }

  private:
    const uint8_t *basePtr;
    const size_t nameOffset;
    const size_t attrOffset;
    ArrayRef<uint8_t> attrSection;
  };

  /// Provides an iterator on the Dictionary. Because we don't want to allocate
  /// in this data structure, we parse each attribute's name and value every
  /// time the iterator is accessed.
  class Iterator
      : public llvm::iterator_facade_base<Iterator, std::input_iterator_tag,
                                          Element> {
  public:
    using Base =
        llvm::iterator_facade_base<Iterator, std::input_iterator_tag, Element>;

    Iterator(const uint8_t *ptr, size_t iterIdx, ArrayRef<uint8_t> attrSection)
        : basePtr(ptr), ptr(const_cast<uint8_t *>(ptr)), iterIdx(iterIdx),
          attrSection(attrSection) {}

    /// Increment the iterator. Decrementing is sufficiently difficult that it's
    /// left as an exercise to the person who wants to use it.
    Iterator &operator+=(Base::difference_type n);

    /// Compare Iterator objects.
    bool operator==(const Iterator &rhs) const {
      assert(basePtr == rhs.basePtr &&
             "Comparing iterators from different dictionaries!");
      return iterIdx == rhs.iterIdx;
    }

    /// Parse out and return the name and value contained in this iterator.
    Element operator*() const;

  private:
    const uint8_t *basePtr;
    uint8_t *ptr;
    size_t iterIdx;
    ArrayRef<uint8_t> attrSection;
  };

  /// Range access into the Dictionary object.
  Iterator begin() const { return {data.begin(), 0, getAttrSection()}; }
  Iterator end() const {
    return Iterator{data.begin(), data.size(), getAttrSection()};
  }

  /// Return the size of the Dictionary in number of elements.
  size_t size() const { return data.size(); }

  /// Return true if there are no attributes in this dictionary.
  bool empty() const { return data.empty(); }

  /// Find an element in the dictionary by name. Parses every name in sequence
  /// until it finds the right one and returns that attribute. If the name is
  /// not found, return None.
  std::optional<MEFDynamicAttr> find(StringRef name) const;

  /// Templated access to the value in find. If nothing is found *or* the type
  /// provided is incorrect, then None is returned.
  template <typename T>
  std::optional<Detail::MEFAttrDynCastRetType<T>> find(StringRef name) const {
    std::optional<MEFDynamicAttr> found = find(name);
    if (!found)
      return std::nullopt;

    return dyn_cast<T>(found);
  }

  /// Parse a named field of the specified attribute dictionary as the specified
  /// type, rejecting it with an error if not present or the wrong type.
  template <typename ValueType>
  ErrorOrSuccess decode(StringRef name, ValueType *result) const {
    auto dynAttrValue = this->find(name);
    if (dynAttrValue)
      return dynAttrValue->decode<ValueType>(result);
    return Error(Twine("missing attribute '") + name + "'");
  }

  /// Parse a named field of the specified attribute dictionary as the
  /// specified type, ignoring it if not present or the wrong type.
  template <typename ValueType>
  ErrorOrSuccess decodeIfPresent(StringRef name, ValueType *result) const {
    // The attribute doesn't have to be present.
    auto dynAttrValue = this->find(name);
    if (!dynAttrValue)
      return success();

    // ... but if it is, it must have the expected type.
    return dynAttrValue->decode<ValueType>(result);
  }

  /// Print the contents of this dictionary to stderr for debugging.
  void dump() const;

private:
  ArrayRef<uint8_t> data;
};

using MEFDictionaryAttr = MEFAttr<Detail::Dictionary>;

} // namespace M

#endif // MLRT_GRAPHRT_MEF_MEFATTR_H
