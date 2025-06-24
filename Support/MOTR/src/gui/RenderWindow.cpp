//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "RenderWindow.h"
#include "motr/Log.h"
#include <GLFW/glfw3.h>
#include <cstdio>

#ifdef __EMSCRIPTEN__
#include <GLFW/emscripten_glfw3.h>
#endif

static void glfw_error_callback(int error, const char *description) {
  printf("GLFW Error %d: %s\n", error, description);
}

void key_callback(GLFWwindow *window, int key, int scancode, int action,
                  int mods) {
  printf("key_callback: %d, %d, %d, %d\n", key, scancode, action, mods);
}

void mouse_cursor_enter_leave_callback(GLFWwindow *window, int entered) {
  if (entered) {
  } else {
    // Important to release the focus when the mouse leaves the window
    // Otherwise, the window will steal keyboard events
    // from html elements like <textarea>
    glfwSetWindowAttrib(window, GLFW_FOCUSED, GLFW_FALSE);
  }
}

RenderWindow::RenderWindow(const std::string &canvas_css_selector,
                           const std::string &container_css_selector) {
  static bool once = false;
  if (once) {
    assert(false && "Multiple RenderWindows not supported yet");
  }
  once = true;

  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit())
    return;

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE);
  glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);

  emscripten::glfw3::SetNextWindowCanvasSelector(canvas_css_selector);
  window = glfwCreateWindow(100, 100, "MOTR", nullptr, nullptr);

  emscripten::glfw3::MakeCanvasResizable(window, container_css_selector);

  if (window == nullptr)
    return;

  // the call to MakeCanvasResizable needs to be done later
  // emscripten::glfw3::MakeCanvasResizable(renderWindow.window,
  // "#canvas-container");

  // DearImGui handles keyboard events, so no need to handle them here
  // glfwSetKeyCallback(window, key_callback);

  // Set the cursor enter callback
  glfwSetCursorEnterCallback(window, mouse_cursor_enter_leave_callback);

  if (renderCore.initWGPU(window, canvas_css_selector)) {
    glfwShowWindow(window);
  } else {
    glfwDestroyWindow(window);
    glfwTerminate();
    window = nullptr;
  }
}

RenderWindow::~RenderWindow() {
  if (window)
    glfwDestroyWindow(window);
  glfwTerminate();
}
