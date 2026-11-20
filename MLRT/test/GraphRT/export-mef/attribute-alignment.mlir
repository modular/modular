// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t
// RUN: od -t x1 %t | FileCheck %s

// In the MEF we have two requirements for alignment:
// 1. The value is aligned based on the natural alignment for its type.
// 2. The attribute kind is exactly one byte before the value.
// Note that the names and number of functions here matter, since they are encoded
// in the MEF file before the attributes, so a change to the name or addition of more
// test functions will require the check validations to be adjusted.

mef.callable @after_si8() {
  "simple.primitive"() { x = 42 : si8, y = 43 : ui32 }: () -> ()
  mef.output
}

// The first byte we are about is way the the end, after 12 bytes that we don't validate ->
// CHECK: 0000100 {{[0-9a-f][0-9a-f] [0-9a-f][0-9a-f] [0-9a-f][0-9a-f] [0-9a-f][0-9a-f] [0-9a-f][0-9a-f] [0-9a-f][0-9a-f] [0-9a-f][0-9a-f] [0-9a-f][0-9a-f] [0-9a-f][0-9a-f] [0-9a-f][0-9a-f] [0-9a-f][0-9a-f] [0-9a-f][0-9a-f]}} 02 2a cb 14
// CHECK: 0000120 2b
// 
// Here is a breakdown of the bytes we expect:
// 000010B 02 # kind for si8
// 000010C 2a # value for si8 (42) 0 aligned at a 1 byte offset
// 000010D cb # padding to align at a 4 byte offset for the ui32 value at 0000104
// 000010D 14 # kind for ui32
// 0000120 2b # value for ui32 (43)


mef.callable @after_ui32() {
  "simple.primitive"() { x = 44 : ui32, y = 45 : ui32 }: () -> ()
  mef.output
}

// CHECK: 0000140 2c 00 00 00 cb cb cb 14 2d 00 00 00
// 
// Here is a breakdown of the bytes we expect:
// 0000140 2c # value for the first ui32 (44)
// 0000141 00 # value for the first ui32 (44)
// 0000142 00 # value for the first ui32 (44)
// 0000143 00 # value for the first ui32 (44)
// 0000144 cb # padding byte (0xcb) to align at a 4 byte offset for the second ui32 (45)
// 0000145 cb # padding byte (0xcb) to align at a 4 byte offset for the second ui32 (45)
// 0000146 cb # padding byte (0xcb) to align at a 4 byte offset for the second ui32 (45)
// 0000147 14 # kind for ui32
// 0000148 2d # value for second ui32 (45)
