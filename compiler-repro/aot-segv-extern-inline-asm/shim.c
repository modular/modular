#include <stdint.h>

int probe_a(uintptr_t ptr, intptr_t a, intptr_t b, uintptr_t ptr2) {
    (void)ptr;
    (void)a;
    (void)b;
    (void)ptr2;
    return 0;
}

void probe_b(uintptr_t ptr, intptr_t a, intptr_t b, uintptr_t ptr2) {
    (void)ptr;
    (void)a;
    (void)b;
    (void)ptr2;
}
