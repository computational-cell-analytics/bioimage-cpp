#pragma once

// BIOIMAGE_FORCE_INLINE: request inlining regardless of the compiler's
// per-translation-unit growth budget. Use it only for small helpers that sit
// inside a hot loop and whose being called out-of-line would change codegen
// materially (e.g. a per-sample interpolation kernel). It also removes a
// link-time hazard for ISA-specialized translation units: a helper that is not
// inlined becomes a weak COMDAT symbol, and the linker may pick the copy
// compiled with a different instruction set.
#if defined(_MSC_VER) && !defined(__clang__)
#define BIOIMAGE_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define BIOIMAGE_FORCE_INLINE inline __attribute__((always_inline))
#else
#define BIOIMAGE_FORCE_INLINE inline
#endif
