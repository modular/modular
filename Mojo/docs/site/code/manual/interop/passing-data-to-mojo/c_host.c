//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
// C host for the interop/passing-data-to-mojo.mdx examples. It makes the
// calls that the page's C, C++, and Rust callers make, and loads the library
// with dlopen() because Bazel builds this binary before the lit test builds
// the Mojo library.

#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

struct Stats;
struct Point {
  double x, y;
};
typedef void (*progress_t)(double, void *);
typedef struct Stats *(*stats_new_t)(void);
typedef double (*stats_add_t)(struct Stats *, double);
typedef void (*stats_free_t)(struct Stats *);
typedef double (*distance_t)(struct Point, struct Point);
typedef double (*point_order_t)(struct Point, struct Point);
struct Label {
  int id;
  char name[16];
};
typedef int (*label_id_plus_length_t)(struct Label);
typedef struct Label (*make_label_t)(int);
typedef double (*rms_t)(const float *, size_t);
typedef char *(*greet_t)(const char *);
typedef void (*free_string_t)(char *);
typedef double (*integrate_sin_t)(int, progress_t, void *);

static void *lib;

static void *symbol(const char *name) {
  void *sym = dlsym(lib, name);
  if (!sym) {
    fprintf(stderr, "dlsym failed: %s\n", dlerror());
    exit(2);
  }
  return sym;
}

static void report(double fraction, void *context) {
  int *calls = context;
  ++*calls;
  printf("%.0f%%\n", fraction * 100);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <shared-lib-path>\n", argv[0]);
    return 2;
  }
  lib = dlopen(argv[1], RTLD_NOW);
  if (!lib) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 2;
  }

  stats_new_t stats_new = (stats_new_t)symbol("stats_new");
  stats_add_t stats_add = (stats_add_t)symbol("stats_add");
  stats_free_t stats_free = (stats_free_t)symbol("stats_free");
  struct Stats *stats = stats_new();
  stats_add(stats, 2.0);
  printf("stats mean = %.4f\n", stats_add(stats, 4.0));
  stats_free(stats);

  distance_t distance = (distance_t)symbol("distance");
  struct Point a = {1.0, 10.0}, b = {4.0, 14.0};
  printf("distance = %.4f\n", distance(a, b));
  point_order_t point_order = (point_order_t)symbol("point_order");
  struct Point first = {1.0, 2.0}, second = {3.0, 4.0};
  printf("point order = %.0f\n", point_order(first, second));

  label_id_plus_length_t label_id_plus_length =
      (label_id_plus_length_t)symbol("label_id_plus_length");
  struct Label label = {7, "sensor0"};
  int score = label_id_plus_length(label);
  printf("label = %d\n", score);

  make_label_t make_label = (make_label_t)symbol("make_label");
  struct Label made = make_label(9);
  printf("made label = %d %s\n", made.id, made.name);

  rms_t rms = (rms_t)symbol("rms");
  float samples[] = {3.0f, -4.0f, 3.0f, -4.0f};
  size_t count = sizeof samples / sizeof samples[0];
  printf("rms = %.4f\n", count == 0 ? 0.0 : rms(samples, count));

  greet_t greet = (greet_t)symbol("greet");
  free_string_t free_string = (free_string_t)symbol("free_string");
  char *greeting = greet("Mojo");
  printf("greet = %s\n", greeting);
  free_string(greeting);

  integrate_sin_t integrate_sin = (integrate_sin_t)symbol("integrate_sin");
  int calls = 0;
  double area = integrate_sin(4, report, &calls);
  printf("area = %.4f, calls = %d\n", area, calls);
  return 0;
}
