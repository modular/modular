//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_MOTR_H
#define MOTR_MOTR_H

#if MOTR_ENABLED != 1
#include "motr/MacrosDisabled.h"
#else
// ORDER MATTERS: DO NOT reorder these #includes
// clang-format off
#include "motr/Macros.h"
#include "motr/Hash.h"
#include "motr/Constants.h"
#include "motr/Message.h"
#include "motr/Queue.h"
#include "motr/SharedMemory.h"
#include "motr/Mailbox.h"
#include "motr/Time.h"
#include "motr/Core.h"
#include "motr/Span.h"
#include "motr/Tags.h"
#include "motr/Flags.h"
// clang-format on

#endif
#endif
