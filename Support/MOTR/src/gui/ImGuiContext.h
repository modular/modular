//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef IMGUI_CONTEXT_H
#define IMGUI_CONTEXT_H

#include "RenderWindow.h"
#include <imgui.h>

struct ImGuiContext {
  ImGuiContext(RenderWindow &renderWindow);
  ~ImGuiContext();
};

#endif // IMGUI_CONTEXT_H
