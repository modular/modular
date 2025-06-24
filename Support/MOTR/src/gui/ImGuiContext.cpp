//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ImGuiContext.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_wgpu.h"
#include "implot.h"
#include "motr/Log.h"

#ifdef __EMSCRIPTEN__
#include <GLFW/emscripten_glfw3.h>
#endif

#include <vector>

#include <dirent.h>
#include <stdio.h>

// Helper function to recursively print files
void printDir(const char *path, int depth = 0) {
  DIR *dir = opendir(path);
  if (!dir)
    return;

  struct dirent *entry;
  while ((entry = readdir(dir))) {
    // Skip . and ..
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
      continue;

    // Print indentation
    for (int i = 0; i < depth; i++)
      printf("  ");

    // Print filename
    printf("%s\n", entry->d_name);

    // If directory, recurse
    if (entry->d_type == DT_DIR) {
      char newpath[1024];
      snprintf(newpath, sizeof(newpath), "%s/%s", path, entry->d_name);
      printDir(newpath, depth + 1);
    }
  }
  closedir(dir);
}

ImGuiContext::ImGuiContext(RenderWindow &renderWindow) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGuiIO &io = ImGui::GetIO();

  // printDir(".");
  // custom icons can be added via:
  // https://github.com/ocornut/imgui/blob/master/docs/FONTS.md#using-custom-colorful-icons
  std::vector<float> sizes = {12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56};
  for (auto size : sizes) {
    io.Fonts->AddFontFromFileTTF("/fonts/JetBrainsMonoNL-Bold.ttf", size);
  }

  for (auto size : sizes) {
    // https://www.s-ings.com/projects/microns-icon-font/
    ImWchar icons_ranges[] = {0xe700, 0xe7ff, 0};
    io.Fonts->AddFontFromFileTTF("/fonts/microns.ttf", size, nullptr,
                                 icons_ranges);
  }
  io.Fonts->Build();

  io.FontDefault = io.Fonts->Fonts[2];

  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  ImGuiStyle &style = ImGui::GetStyle();
  style.WindowRounding = 12.0f;
  style.FramePadding = ImVec2(18.0f, 5.0f);
  style.ScrollbarSize = 18.0f;
  style.ScrollbarRounding = 12.0f;
  style.WindowBorderSize = 0.0f;
  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForOther(renderWindow.window, true);
  ImGui_ImplWGPU_InitInfo init_info;
  init_info.Device = renderWindow.renderCore.wgpuDevice;
  init_info.NumFramesInFlight = 3;
  init_info.RenderTargetFormat = renderWindow.renderCore.wgpuPreferredFmt;
  init_info.DepthStencilFormat = WGPUTextureFormat_Undefined;
  ImGui_ImplWGPU_Init(&init_info);

  // Load Fonts
  // - If no fonts are loaded, dear imgui will use the default font. You can
  // also load multiple fonts and use ImGui::PushFont()/PopFont() to select
  // them.
  // - AddFontFromFileTTF() will return the ImFont* so you can store it if you
  // need to select the font among multiple.
  // - If the file cannot be loaded, the function will return a nullptr. Please
  // handle those errors in your application (e.g. use an assertion, or display
  // an error and quit).
  // - The fonts will be rasterized at a given size (w/ oversampling) and stored
  // into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which
  // ImGui_ImplXXXX_NewFrame below will call.
  // - Use '#define IMGUI_ENABLE_FREETYPE' in your imconfig file to use Freetype
  // for higher quality font rendering.
  // - Read 'docs/FONTS.md' for more instructions and details.
  // - Remember that in C/C++ if you want to include a backslash \ in a string
  // literal you need to write a double backslash \\ !
  // - Emscripten allows preloading a file or folder to be accessible at
  // runtime. See Makefile for details.
  // io.Fonts->AddFontDefault();
#ifndef IMGUI_DISABLE_FILE_FUNCTIONS
  // io.Fonts->AddFontFromFileTTF("fonts/segoeui.ttf", 18.0f);
  // io.Fonts->AddFontFromFileTTF("fonts/DroidSans.ttf", 16.0f);
  // io.Fonts->AddFontFromFileTTF("fonts/Roboto-Medium.ttf", 16.0f);
  // io.Fonts->AddFontFromFileTTF("fonts/Cousine-Regular.ttf", 15.0f);
  // ImFont* font = io.Fonts->AddFontFromFileTTF("fonts/ArialUni.ttf", 18.0f,
  // nullptr, io.Fonts->GetGlyphRangesJapanese()); IM_ASSERT(font != nullptr);
#endif

  // For an Emscripten build we are disabling file-system access, so let's not
  // attempt to do a fopen() of the imgui.ini file. You may manually call
  // LoadIniSettingsFromMemory() to load settings from your own storage.
  io.IniFilename = nullptr;
}

ImGuiContext::~ImGuiContext() {
  ImGui_ImplWGPU_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();
}
