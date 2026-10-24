# Modular Executor Format (MEF) File Format

This document describes the goals and rationale for the in-memory representation
the [GraphRT](README.md) executor uses to run primitives, known as the Modular
Executor Format "MEF".

We have the following goals:

- We want this to be simple and relatively stable - though it is versioned
  just in case (tm). We don't want MEF to contain domain specific
  abstractions that will require changing it frequently over time.

- This format is designed to be usable as an immutable region in memory,
  allowing it to be built offline and conveniently `mmap`'d in (or stored in
  the `.rodata` segment of an executable, etc).

- This is aligned with the abstractions used by the GraphRT executor and
  includes key abstractions that it can directly use to make it speedy and
  memory efficient. The design puts more burden on the encoder of a MEF file
  (for example requiring multiple passes) to make the decoder fast.

- This needs to be producible online for example in training or interactive use
  cases.

- We need to represent location information in order to reflect runtime errors
  (for example shape errors) back to the user source program.

- We should follow normal best practices for binary encodings, for example
  having tools to round trip them to a textual form (MLIR in our case), support
  arbitrary sized files (not limited to 4GB), be efficient to read, make
  reasonable attempts to be dense, etc.

In addition to describing the format, this document describes some of the
rationale for the design decisions.

## Fundamental Concepts

**Grammar:**

```none
  BYTE                  ::= `0x00`...`0xFF`
  VBR                   ::= (`0x80`...`0xFF`)* (`0x00`...`0x7F`)
  FIXED16               ::= BYTE BYTE
  FIXED32               ::= BYTE BYTE BYTE BYTE
  FIXED64               ::= BYTE BYTE BYTE BYTE BYTE BYTE BYTE BYTE
  NUL_TERMINATED_STRING ::= (`0x01`...`0xFF`)* `0x00`
  OFFSET                ::= VBR
  INDEX                 ::= VBR
```

A MEF file is formed as a byte stream whose top-level structure is a list of
"sections" that hold various sorts of data. It uses a few fundamental concepts:

- *VBRs* are encoded in a "Variable Byte Rate" (`VBR`) encoding, allowing small
  unsigned integers (less than 2^7) to be stored in one byte, unsigned integers
  up to 2^14 to be stored in two bytes, etc. This is done by using the high bit
  of each byte in the stream to indicate "more bytes are coming". *VBR* values
  are stored in big-endian byte order, while *reverse VBR* values are stored in
  little-endian byte order. *signed VBRs* are encoded using an extra sign bit.
  Positive numbers are shifted left by 1 and negative numbers are negated and
  shifted left by 1 with bit 0 set, this final value is then encoded as a
  regular *VBR*.

  **Example:**

  ```none
   2 = 0b00000010 (two's complement)
  signed VBR = 0b00000100
         VBR = 0b00000010

  -2 = 0b11111110 (two's complement)
  signed VBR = 0b00000101
  ```

- *Fixed Integers* are unsigned integers with fixed bit width. The values are
  stored in little-endian byte order.

- *NUL Terminated Strings* are terminated with the ASCII `NUL` character
  (whose byte value is zero). These are not used in cases where a string may
  contain an embedded `NUL` character.

- *Offsets* are VBRs that demarcate a byte offset in the stream from a
  fixed position (typically the start of a section). These are useful when the
  section contains a bunch of variable length things, for example when referring
  to a string in the string table. Because variable sized integers are used
  pervasively in MEF files, offsets are very common.

- *Indexes* are VBRs that demarcate an entry in a table, for example a register
  or primitives in an invokable. These are typically used when indexing into
  a table where the structure is expected to be decoded and held in memory by
  the reader, for example as is the case with the primitives and types tables.

MEF files can be created by an arbitrary producer, but the standard ways is to
use the [MLIRToMEF](../../lib/GraphRT/MLIRToMEF) translation library
(or standalone tool),
which turns an MLIR representation of a program into a MEF file.

The `MEFToMLIR` translation library performs the reverse translation from MEF
file to MLIR representation of the program. It is useful for writing MEF
encoder testcases and for dealing with that "random file someone sent you".

## Sections

**Grammar:**

```none
  SECTION                ::= SECTION_HEADER SECTION_DATA
  SECTION_HEADER         ::= SECTION_ID VBR<LENGTH_AND_ALIGNMENT>
                             SECTION_BODY_ALIGNMENT?
  SECTION_ID             ::= BYTE
  LENGTH_AND_ALIGNMENT   ::= (SECTION_LENGTH << 1) | (SECTION_ALIGNMENT_FLAG)
  SECTION_BODY_ALIGNMENT ::= BYTE<"Alignment"> BYTE<"AlignmentPadding">*
```

*Sections* are the top level entities in the file (after the header). Each
section contains a *Section ID*, a length (allowing the section to be skipped
over entirely by the reader) and an optional alignment for the contents of the
section.

Section IDs and other fundamental constants are defined in
[`MEFEncoding.h`](../../include/MLRT/GraphRT/MEF/MEFEncoding.h), and utilities
for decoding the basic file structures like VBR integers are defined in
[`MEFReader.h`](../../include/MLRT/GraphRT/MEF/MEFReader.h).

The `LENGTH_AND_ALIGNMENT` contains `SECTION_LENGTH` and
`SECTION_ALIGNMENT_FLAG`. The `SECTION_LENGTH` contains one bit shifted value
of the section length and the `SECTION_ALIGNMENT_FLAG` (0 bit) indicates if
`SECTION_BODY_ALIGNMENT` exists or not; 0 means the section body starts
immediately and 1 means `SECTION_BODY_ALIGNMENT` follows.

## Top Level Structure

**Grammar:**

```none
  MEF_FILE     ::= `0xBE` `0xF0` SECTION*

  SECTION_DATA ::= STRINGS_SECTION
  SECTION_DATA ::= ATTRIBUTES_SECTION
  SECTION_DATA ::= PRIMITIVE_NAMES_SECTION
  SECTION_DATA ::= INVOKABLES_INDEX_SECTION
  SECTION_DATA ::= INVOKABLES_BODIES_SECTION
  SECTION_DATA ::= LOCATION_FILENAMES_SECTION
  SECTION_DATA ::= LOCATIONS_SECTION
  SECTION_DATA ::= ATTRIBUTE_INDEX_MD_SECTION
  SECTION_DATA ::= ATTRIBUTE_NAMES_MD_SECTION
  SECTION_DATA ::= MLIRTYPE_STRINGS_MD_SECTION
  SECTION_DATA ::= MLIRTYPE_MD_SECTION

  // Unknown section.
  SECTION_DATA ::= BYTE*
```

The top level structure of the file is a two-byte "magic number" of `0xBEF0`
and a list of sections. The `0xBEF0` magic number allows us to encode version
numbers, for example the following would be `0xBEF1`.

The MEF reader skips over unknown sections, which could be useful for future
evolution of the format, for example if we want to store extra metadata in the
MEF format for some purpose, or store binary tensor blobs.

### Strings Section

**Grammar:**

```none
  STRINGS_SECTION ::= NUL_TERMINATED_STRING*
```

The Strings section contains a list of `NUL` terminated strings used by the
program (for example for type and primitive names). Entries in this section are
referenced by an Offset from the start of the section. This section does not
support embedded `NUL` in strings.

### `ATTRIBUTES` Section

**Grammar:**

```none
  ATTRIBUTES_SECTION ::= BYTE*
```

The `ATTRIBUTES_SECTION` contains the value of attributes used by primitives in
the MEF program. They are stored on the natural alignment boundary of the type,
and the address of the attribute is directly passed into the primitive
implementation C/C++ function as an const pointer.

MEF files only support a subset of MLIR attributes, currently including:

- `i1` integers: stored as 1-byte integers.
- `si8`/`si16`/`si32`/`si64` integers: stored as 1/2/4/8-byte little endian
  integers.
- `ui8`/`ui16`/`ui32`/`ui64` integers: stored as 1/2/4/8-byte little endian
  integers.
- `f32` floats: stored as IEEE single precision floats.
- `f64` floats: stored as IEEE double precision floats.
- `strings`: stored as arrays of bytes, not necessarily NUL terminated.
- `arrays`: all elements of which are of the same type and fixed in width (eg.
  `si32`, `f32`, etc).

The encoding of an attribute has three pieces:

1) For arrays and strings, the length of the object (in units) is encoded as a
   *reverse VBR* byte sequence, allowing readers to understand how long the
   array or string is.
2) All attributes have a one-byte encoding of the "kind" of attribute it is,
   which is described by the `AttributeKind` enum in
   [`MEFEncoding.h`](../../include/MLRT/GraphRT/MEF/MEFEncoding.h).
3) The value of the attribute is stored in little-endian byte order.

Note that emission of the attribute is always done so that the attribute value
is naturally aligned, for example that a `si32` value is emitted on a 4-byte
alignment.

***Rationale***

The GraphRT executor needs to refer to attribute values, and since the MEF file
is designed to be memory mapped in, we can directly use the encoding in memory.
This means that primitives will need to `bswap` attributes for big-endian
systems, but they generally have `bswap`'ing loads anyway so this shouldn't
impose a performance penalty.

### `PRIMITIVE_NAMES` Section

**Grammar:**

```none
  PRIMITIVE_NAMES_SECTION ::= VBR<NumPrimitiveNames>
                              OFFSET<"PrimitiveNameOffset">*
```

The `PRIMITIVE_NAMES_SECTION` defines a table of primitive names, directly
corresponding to the names in the MLIR GraphRT executor program. This allows
references from the `INVOKABLES_BODIES` section to use dense indexes into the
table.

This section contains one VBR value for each primitive, which is an offset into
the string table for the primitive name.

***Rationale***

We want dense indexes from primitives in the `INVOKABLES_BODIES` section, and
the GraphRT Executor wants to resolve names to primitive implementations at
startup time anyway, as such, it makes sense for the GraphRT executor to build
its own mutable array in memory for primitives.

We encode the count of primitive names so the executor can allocate one block
of memory up front, rather than use a vector-style resizing mechanism.

### `INVOKABLES_INDEX` Section

**Grammar:**

```none
  INVOKABLES_INDEX_SECTION ::= VBR<"NumInvokables"> INVOKABLE_ENTRY*
  INVOKABLE_ENTRY          ::= OFFSET<"Body"> VBR<"NumArguments">
                               VBR<"NumResults">
```

The `INVOKABLES_INDEX` section defines a table of "invokables" in the MEF file,
including an Offset into the
[`INVOKABLES_BODIES_SECTION`](#invokables_bodies-section), a primitive name (an
Offset into the [`STRINGS_SECTION`](#strings-section), may be empty), a symbol
name (an Offset into the [`STRINGS_SECTION`](#strings-section), may be empty),
an offset to the body, and the number of arguments and results.

***Rationale***

This defines a symbol table for the MEF file, allowing clients to look up
invokables by name. While we could intersperse this information into the
[`INVOKABLES_BODIES_SECTION`](#invokables_bodies-section) itself, doing so would
require the reader to make a pass over all of the `INVOKABLES_BODIES` ahead of
time. We prefer to have a quick index that the reader can scan at load time,
deferring processing of any individual invokable until it is needed.

We encode the count of invokables so the executor can allocate one block
of memory up front, rather than use a vector-style resizing mechanism.

### `INVOKABLES_BODIES` Section

**Grammar:**

```none
  INVOKABLES_BODIES_SECTION ::= INVOKABLE_BODY*
```

The `INVOKABLES_BODIES` section is a series of bodies emitted to the byte stream
and then addressed by an Offset indicated by the
[`INVOKABLES_INDEX` section](#invokables_index-section). This section is 4-byte
aligned.

***Rationale***

Regions in MLIR are a generalization of a unit of computation. The MEF format
and GraphRT executor support a subset of MLIR region features that are core to
the abstractions we need to model - in particular, while MLIR regions can have
multiple basic blocks in them, invokables only support a single block - this
guarantees that they are always be a DAG of computation.

Invokables are used for top-level MLIR `mef.invokable` operations (indexed
by the invokable name table) as well as in nested positions for control flow
and other concepts that occur with MLIR regions.

### Invokable Body Definition

**Grammar:**

```none
  INVOKABLE_BODY  ::= VBR<"NumPrimitives"> REGISTER_TABLE PRIMITIVE_TABLE
                      RESULT_REGS BYTE<"AlignmentPadding">* PRIMITIVE+

  REGISTER_TABLE  ::= VBR<"NumRegs"> REGISTER_ENTRY*
  REGISTER_ENTRY  ::= VBR<"NumUses">

  PRIMITIVE_TABLE ::= PRIMITIVE_ENTRY*
  PRIMITIVE_ENTRY ::= OFFSET<"PrimitiveOffset"> VBR<"NumOperands">

  RESULT_REGS     ::= INDEX<"Register">*
```

Each invokable is defined by a location (an offset into the
[LocationPositions section](#locationpositions-section)), a register table, a
primitive table, a list of result registers, a list of primitives with 4-byte
alignment, and ends with a FIXED32 integer of value zero.

The Register Table is a count of registers, and an entry for each register -
indicating the number of primitives in this section that use the register.

The Primitive Table for an invokable is a count of primitives, an offset
(from the end of the Primitive Table) of the start of the primitive and the
number of operands that the primitive has.

The primitive list that is following the Primitive Table contains all the
primitives used in this invokable. Note that every invokable has a
pseudo-primitive that is the single entry point to the rest of the primitives.
Specifically, a pseudo-primitive defines registers for invokable arguments and a
pseudo-register that is conceptually used by primitives that take no arguments.

The result registers specify the register values to return, and must align with
the invokable result types from the [`INVOKABLES_INDEX`
section](#invokables_index-section).

***Rationale***

These two tables are key to allowing the GraphRT executor to efficiently
dispatch and destroy primitives in a lock-free way. At startup time, the
executor inflates these two tables into arrays, resolving the type descriptors
for the types, and builds a table of ready counts for the primitives. Having a
table of primitives allows the use of primitive indexes.

### Primitive Definition

**Grammar:**

```none
  PRIMITIVE              ::= PRIMITIVE_HEADER
                             PRIMITIVE_BODY

  PRIMITIVE_HEADER       ::= FIXED32<"PrimitiveCode">
                             FIXED32<"PrimitiveLocation">
                             FIXED16<"NumOperands">
                             FIXED16<"NumAttributes">
                             FIXED16<"NumInvokables">
                             FIXED16<"NumResults">

  PRIMITIVE_BODY         ::= FIXED64<PrimitiveAttribute>* (occurs NumAttributes times)
                             FIXED32<PrimitiveOperand>* (occurs NumOperands times)
                             FIXED32<PrimitiveInvokable>* (occurs NumInvokables times)
                             FIXED32<PrimitiveResult>? (occurs if NumResults > 0)
                             FIXED32<NumUsedBys>*
                             FIXED32<PrimitiveUsedBy>*
```

Each primitive has these properties:

- Opcode (Index into Primitive Names section)
- Location (Index into Location Positions section)
- Operands/Inputs (Indexes into invokable's Register table)
- Constants (Offsets into Attributes section)
- Invokables/Regions (Indexes into Invokable Index section)
- Results (Indexes into invokable's Register table)
- Result used-by counts

It is organized as a fixed-size header containing counts for the variable-size
portions of the primitive, followed by a variable-size portion whose size
depends on the header. Additionally, the size of `PrimitiveUsedBy` depends on
the values of `NumUsedBys`.

MEF primitives must have consecutive result registers. Hence, when `NumResults
>= 2`,`PrimitiveResult`still only consists of one`FIXED32`value,
representing the first result register.  When`NumResults = 0`,`PrimitiveResult`
is absent.

'Used by' records for the same result are grouped together and these groups are
in the same order of 'result' records. For example, if there are two results, A
and B, and A has one user (a0) and B has two users (b0, b1), then
`FIXED32<"NumResults">` will be `0x00000002`; `FIXED32<"NumUsedBys">` will have
two entries, `0x00000001` and `0x00000002`; and there will be three
`FIXED32<PrimitiveUsedBy>` records, a0, b0 and b1, consecutively.

***Rationale***

This record allows us to efficiently form the array of operands and results
that get passed to a C/C++ function that implements the primitive. When the
primitive completes, the UsedBy records allow us to efficiently trigger
execution of data dependent primitives if they are ready, by decrementing the
"NumOperands" field for the primitive.

### LocationFilenames Section

**Grammar:**

```none
  LOCATION_FILENAMES_SECTION ::= NUL_TERMINATED_STRING*
```

This section contains a list of `NUL` terminated strings used by locations
section. A filename field of `FileLineCol` location and name field of Name
location are stored in this section.

***Rationale***

We choose to store these separately from the string table, because these
locations are only ever decoded in the case of an error. There is no reason to
dirty data cache lines with them, and we expect no reuse with other general
strings.

### LocationPositions Section

**Grammar:**

```none
  LOCATIONS_SECTION ::= LOCATION*
  LOCATION ::= OFFSET<"Filename"> VBR<"LineNum"> VBR<"ColumnNum">
```

This section contains a list of "locations", which are a VBR offset into the
LocationFilenames section, and VBRs for the line and column of the position.

### `ATTRIBUTE_INDEX_MD` Section

**Grammar:**

```none
  ATTRIBUTE_INDEX_MD_SECTION ::= VBR<"NumAttributes"> ATTRIBUTE_INDEX_ENTRY*
  ATTRIBUTE_INDEX_MD_ENTRY   ::= OFFSET<"Attributes">
```

The `ATTRIBUTE_INDEX_MD` section specifies the kind of each attribute emitted
to the attribute section, allowing `MEFToMLIR` conversion by allowing us to
know the starting offset for each attribute in the file.

### `ATTRIBUTE_NAMES_MD` Section

**Grammar:**

```none
  ATTRIBUTE_NAMES_SECTION ::= PRIMITIVE_TABLE*
  PRIMITIVE_TABLE         ::= OFFSET<"AttributeName">*
```

The AttributeNames sections is an optional section which is not needed for MEF
execution, but which makes `MEFToMLIR` conversion much nicer. This
section describes the attribute names used by each primitive. There is a 1:1
mapping between Primitive Table in this section and entries in
[`INVOKABLES_INDEX` section](#invokables_index-section). Each attribute and
region for a primitive has an entry here specifying the name. The offset is
into the `STRINGS_MD` section (not the `STRINGS` section).

***Rationale***

Attribute names are strings that are not needed by the MEF Executor and have
nothing to do with types or other metadata so we put them in their own section
looked up lazily by offset.

### `MLIRTYPE_STRINGS_MD` Section

**Grammar:**

```none
  MLIRTYPE_STRINGS_MD_SECTION ::= NUL_TERMINATED_STRING*
```

The `MLIRTYPE_STRINGS_MD` section defines a table of MLIR type names used by the
program, in a format that can be parsed by `MEF2MLIR`. These are referenced by
index into the section, not offset.

***Rationale***

Type names are not needed by the MEF executor, and are decoded in a single dense
pass up-front in `MEF2MLIR`. As such, they are referenced by index (not offset).

### `MLIRTYPE_MD` Section

**Grammar:**

```none
  MLIRTYPE_MD_SECTION ::= INDEX<"MLIRTYPE_STRINGS_MD">*
```

The `MLIRTYPE_MD` section is an optional section which is not needed for MEF
execution, but allows `MEF2MLIR` conversion to regenerate the correct types for
invokable arguments and the result types of primitives. This section
contains a list of VBR encoded indexes into the
[`MLIRTYPE_STRINGS_MD` section](#mlirtype_strings_md-section), and is encoded in
the same order as the invokable table. Each invokable has a list of argument
types and a type for each primitive result.
