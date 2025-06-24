//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef RENDER_WINDOW_H
#define RENDER_WINDOW_H

#include "RenderCore.h"
#include <GLFW/glfw3.h>
#include <string>

struct RenderWindow {
  GLFWwindow *window{nullptr};
  RenderCore renderCore;

  RenderWindow(const std::string &canvas_css_selector,
               const std::string &container_css_selector);
  RenderWindow(const RenderWindow &) = delete;
  RenderWindow &operator=(const RenderWindow &) = delete;
  RenderWindow(RenderWindow &&) = delete;
  RenderWindow &operator=(RenderWindow &&) = delete;
  ~RenderWindow();
};

#endif // RENDER_WINDOW_H
