//===- Runtime.cpp - LLCL Runtime implementation --------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the core LLCL Runtime.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime.h"
#include "LLCL/Allocator.h"
using namespace LLCL;

void Allocator::vtableAnchor() {}

Runtime::Runtime(std::unique_ptr<Allocator> allocator)
    : allocator(std::move(allocator)) {}

Runtime::~Runtime() {}
