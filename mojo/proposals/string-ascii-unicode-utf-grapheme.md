# String, ASCII, Unicode, UTF, Graphemes

Edit 0: 2025-02-05. Created.
Edit 1: 2025-04-27. Reduced the scope to only affect `StringSlice` and fixed
some details after feedback.

The current proposal will attempt to unify the handling of the standards.
Providing nice ergonomics, as Python-like defaults as possible, and keeping the
door open for optimizations.

## String

String is currently an owning type which encodes string data in UTF-8, which is
a variable length encoding format. Unicode Codepoints can be 1 byte up to 4
bytes long. ASCII text, which is where English text falls, is 1 byte long. As
such, the defaults should optimize for that character set given it is most
typical on the internet and on backend systems.

## ASCII

Ascii is pretty optimizable, there are many known tricks. The big problem
with supporting only ASCII is that other encodings can get corrupted.

## UTF standards

- UTF-8 is 1-4 sets of 8 bits long
- UTF-16 is 1-2 sets of 16 bits long
- UTF-32 is 1 set of 32 bits long

### When slicing by unicode codepoint e.g. "🔥🔥🔥" (\U0001f525\U0001f525\U0001f525)

- UTF-8: 12 sets (12 bytes) long. The first byte of each fire can be used to
know the length, and the next bytes are what is known as a continuation byte.
There are several approaches to achieve the slicing, they can be explored with
benchmarking later on.
- UTF-16: 6 sets long. It's very similar in procedure to UTF-8.
- UTF-32: It is fastest since it's direct index access. This is not the case
when supporting graphemes.

## Graphemes

Graphemes are an extension which allows unicode codepoints to modify the end
result through concatenation with other codepoints. Visually they are shown as a
single unit e.g. é is actually comprised of `e` followed by `´`  (\x65\u0301).

Graphemes are more expensive to slice because one can't use faster algorithms,
only skipping each unicode codepoint one at a time and checking the last byte
to see if it's an extended cluster *.

*: extended clusters is the full feature complete grapheme standard. There
exists the posibility of implementing only a subset, but that defeats the
purpose of going through the hassle of supporting it in the first place.

## Context

C, C++, and Rust use UTF-8 with no default support for graphemes.

### Swift

Swift is an interesting case study. For compatibility with Objective-C they went
for UTF-16. They decided to support grapheme clusters by default. They recently
changed the preferred encoding from UTF-16 to UTF-8. They have a `Character`
type which is a generic representation of a Character in any encoding, that can
be from one codepoint up to any grapheme cluster.

### Python

Python currently uses UTF-32 for its string type. So the slicing and indexing
is simple and fast (but consumes a lot of memory, they have some tricks to
reduce it). They do not support graphemes by default. Pypi is implementing a
UTF-8 version of Python strings, which keeps the length state every x
characters.

### Mojo

Mojo aims to be close to Python yet be faster and customizable, taking advantage
of heterogeneous hardware and modern type system features.

## Value vs. Reference

Our current `Codepoint` type uses a UInt32 as storage, every time an iterator
that yields `Codepoint` is used, an instance is parsed from the internal UTF-8
encoded `StringSlice` (into UTF-32).

The default iterator for `String` returns a `StringSlice` which is a view into
the character in the UTF-8 encoded `StringSlice`. This is much more efficient
and does not add any complexity into the type system nor developer headspace.

## Now, onto the Proposal

### Goals

#### Hold off on developing Char further and remove it from stdlib.builtin

Status: Implemented. It was renamed to `Codepoint` as well.

`Char` is currently expensive to create and use compared to a `StringSlice`
which is a view over the original data. There is also the problem that it forces
UTF-32 on the data, and those transformations are expensive.

We can revisit `Char` later on making it take encoding into account. But the
current state of the type makes it add more complexity than strictly necessary.

#### Full ASCII optimizations

If someone wishes to use a `String` as if it's an ASCII-only String, then there
should either be a parameter that signifies that, or the stdlib/community should
add an `ASCIIString` type which has all optimizations possible for such
scenarios.

#### Mostly ASCII optimizations

Many functions can make use of branch predition, instruction and data
prefetching, and algorithmic tricks to make processing faster for languages
which have mostly ASCII characters but still keep full unicode support.

#### Grapheme support
Grapheme support should exist but be opt-in due to their high compute cost.

### One concrete (tentative) way forward

With a clear goal in mind, this is a concrete (tentative) way forward.

#### Add parameters to `StringSlice`

Note: We can define the defaults later. There are several options and each have
their own merit.

```mojo
struct Encoding:
    alias UTF8 = 0
    alias UTF16 = 1
    alias UTF32 = 2
    alias ASCII = 3
    # maybe in the future we will even implement parsing for http header encoding
    # alias ISO_8859_1 = 4

struct Indexing:
    alias RAW = 0
    alias CODEPOINT = 1
    alias GRAPHEME = 2

alias ASCIIString = StringSlice[encoding=Encoding.ASCII, indexing=Indexing.RAW]


struct StringSlice[
    encoding: Encoding = Encoding.UTF8,
    indexing: Indexing = Indexing.RAW,
]:
    ... # data is bitcasted to bigger DTypes when encoding is 16 or 32 bits

struct String:
    fn as_string_slice(
      ref self
    ) -> StringSlice[__origin_of(self), Encoding.UTF8, Indexing.RAW]:
        ...
```

#### What this requires

1. Phase 1: Add parameters and constraints.
- Add the parameters and constraint on the supported encodings. This
will enable progressive implementation for each encoding and indexing scheme
without breaking any existing code, since our type currently already implements
UTF-8 encoding and (to my chagrin) raw indexing.

2. Phase 2: Add parametric support for other encodings.
- Rewrite every function signature that uses `StringSlice`s, where the code
doesn't require them to be the defaults. This will be a lot of work because it
basically affects every place `StringSlice` is used in a function signature with
a specified origin or mutability.

3. Phase 3: Add parsing for other encodings.
- We will need to add parametrized `StringSlice` constructors that parse
`StringSlice`s from another encoding. But the default same-encoding internal
methods will remain the same **(as long as the code is encoding and
indexing-agnostic)**.

4. Phase 4: Optimize for each encoding.
- Add support for functions like `.isspace()`, `.split()` and `.splitlines()`
for those encodings.


#### Adapt CodepointIter

The default iterator should iterate in the way in which the `StringSlice` is
parametrized. The iterator could then have functions which return the other
types of iterators for ergonomics (or have constructors for each) *.

*: this is a workaround until we have generic iterator support.

e.g.
```mojo
data = StringSlice("123 \n\u2029🔥")
for c in data:
  ... # StringSlice in the same encoding and indexing scheme by default

# We could also have lazy iterators which return StringSlices according to the
# encoding and indexing parameter.
# In the case of Encoding.ASCII unicode separators (\x85, \u2028, \u2029) can
# be ignored and thus the processing is much faster.
for c in iter(data).split():
  ... # StringSlice lazily split by all whitespace
for c in iter(data).splitlines():
  ... # StringSlice lazily split by all newline characters
```

#### What this enables

```mojo
# ability to go down raw for people who want to
direct_utf8 = StringSlice[indexing=Indexing.RAW]("some long long string")
# performance benefits of raw indexing
direct_utf8[0: direct_utf8.find("end_of_something")]

# faster split, splitlines, and indexing for UTF-32
utf32 = StringSlice[encoding=Encoding.UTF32]("some long long string")
utf32.splitlines() # no bounds checking *, direct value comparison, etc.
ascii_val[0:3] # raw indexing, no counting codepoints

# faster everything for ASCII only strings
ascii_val = StringSlice[encoding=Encoding.ASCII]("some long long string")
ascii_val.splitlines() # no bounds checking *, direct value comparison, etc.
ascii_val[0:3] # raw indexing, no counting codepoints
ascii_val.uppercase() # incredibly fast due to bitflip trick
```

#### An example part of the implementation

```mojo
struct StringSlice[
    encoding: Encoding = Encoding.UTF8,
    indexing: Indexing = Indexing.RAW,
]:
  fn __getitem__(self, slice: Slice) -> Self:
    @parameter
    if encoding is Encoding.UTF32:
      return Self(unsafe_from_utf32=rebind[Span[UInt32, origin]](self._slice)[slice])
    elif encoding is Encoding.UTF16 and indexing is Indexing.RAW:
      return Self(unsafe_from_utf16=rebind[Span[UInt16, origin]](self._slice)[slice])
    elif encoding is Encoding.ASCII and indexing is not Indexing.GRAPHEME:
      return Self(unsafe_from_ascii=self._slice[slice])
    elif encoding is Encoding.UTF8 and indexing is Indexing.RAW:
      ... # current implementation
    else:
      constrained[False, "given encoding and indexing scheme not supported yet"]()
```
