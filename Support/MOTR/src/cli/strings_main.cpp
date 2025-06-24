//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "motr/Log.h"
#include "motr/Mailbox.h"
#include "motr/motr.h"

using Outbox = M::motr::ServerOutboxString;
using StringViews = Outbox::StringViews;

static StringViews argsToStringViews(int argc, char **argv) {
  StringViews views;
  views.reserve(argc);
  for (int i = 0; i < argc; i++)
    views.emplace_back(argv[i]);
  return views;
}

static void logStringViews(const StringViews &views) {
  for (auto &sv : views) {
    M::motr::Hash::Value hash{sv};
    MOTR_LOG("[0x{:016x}] [{:3d}] \"{}\"", hash.v, sv.size(), sv);
  }
}

int stringsMain(int argc, char **argv) {
  MOTR_TraceProgram(stringsMainTrace, "motr strings");

  // todo: add an option to wait for the string msg to be consumed
  if (argc < 3) {
    MOTR_LOG("Usage: motr strings <string1> ... <stringN>", "");
    return 1;
  }

  // shift "motr strings" out of the way
  argc -= 2;
  argv += 2;

  StringViews views = argsToStringViews(argc, argv);
  logStringViews(views);
  Outbox::send(views);

  return 0;
}
