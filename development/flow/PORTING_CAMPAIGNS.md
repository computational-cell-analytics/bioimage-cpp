# Flow tracer: porting campaigns for Apple Silicon and MSVC

Two self-contained prompts to start an optimization session on a machine that
is not covered by the Linux/GCC measurements in `PERFORMANCE_NOTES.md`. Each
prompt assumes a checkout of branch `flow-optim-fable` (commit `54fd4a9`) and
the harnesses in this directory.

State at the time of writing:

- **arm64 macOS** gets no specialized tracer. The CMake block
  `BIOIMAGE_FLOW_FMA_DISPATCH` matches only x86, so the portable scalar header
  kernel runs. The packed kernel uses 26 distinct 128-bit intrinsics that all
  map 1:1 onto NEON, and NEON+FMA is baseline on arm64, so no runtime dispatch
  is needed.
- **Windows/MSVC** already builds and dispatches the packed kernel
  (`/arch:AVX2`, `__cpuid`/`_xgetbv`), but it has never been measured there.
  The open items are inlining under `__forceinline`, COMDAT/ODR leakage, and
  the `/fp:precise` contraction question for the scalar fallback.

## Prompt: Apple Silicon (arm64 clang)

```text
Context: bioimage-cpp, branch flow-optim-fable (commit 54fd4a9). On x86 the flow
tracer (`bic.flow.compute_flow_density`) has a packed-channel 128-bit SSE+FMA kernel in
src/cpp/flow/flow_density_fma.cxx, dispatched from `try_trace_all_fma` in
include/bioimage_cpp/flow/flow_density.hxx, giving -30..-40 % on Zen 3 and Tiger Lake.
On arm64 nothing is specialized: the CMake block `BIOIMAGE_FLOW_FMA_DISPATCH` matches
only x86, so this Mac runs the portable scalar header kernel. Read
development/flow/PERFORMANCE_NOTES.md, section "Packed-channel FMA tracer (2026-09-02)",
for the design, the per-step diagnosis, and the rejected experiments before you start.

Goal: a NEON port of the packed kernel for arm64 macOS, gated on the same
correctness bar, with paired numbers recorded in PERFORMANCE_NOTES.md.

Steps:
1. Build (pip install -e . --no-build-isolation) and confirm
   `_core._flow_trace_backend()` reports "scalar". Run
   development/flow/paired_bench.py --a X.so --b X.so --cases fixture3d,fixture2d
   without --cpu (no taskset/sched_setaffinity here); every row must read "noise".
   If single-thread spread exceeds ~3 %, address P/E-core placement or use more repeats.
2. Baseline the scalar kernel at 1T and at the physical P-core count. Write the
   numbers down before changing anything.
3. Port src/cpp/flow/flow_density_fma.cxx to <arm_neon.h> in a new TU
   (e.g. src/cpp/flow/flow_density_neon.cxx). All 26 intrinsics used are 128-bit and
   map 1:1 (vld1q_f32, vfmaq_f32, vcvtq_s32_f32, vgetq_lane_s32, vminq/vmaxq,
   vcgtq/vcltq + vbslq, vextq/vzipq/vdupq_laneq, a movemask idiom via vshrn/vaddvq).
   Check the truncating-convert difference: NEON saturates and maps NaN to 0, SSE
   returns INT_MIN; verify the clip/bounds/mask tests do not depend on SSE behavior.
   No runtime dispatch and no CPUID are needed (NEON+FMA is baseline on arm64);
   keep BIOIMAGE_CPP_FLOW_FORCE_SCALAR as the opt-out and make trace_backend()
   return "neon". Keep BIOIMAGE_FORCE_INLINE on the per-step helpers and confirm
   with `otool -tV` that nothing in the hot loop is called out of line.
4. Re-sweep the lockstep lane count K (4, 6, 8): the x86 choice of 4 came from an
   FP-register-file bottleneck that M-series cores may not have.
5. Gate: differential_check.py --a base.so --b neon.so (400 cases, sums equal,
   diff fraction below 1 %) and --env-b BIOIMAGE_CPP_FLOW_FORCE_SCALAR=1;
   check_flow_density.py --dim both; full pytest; the parity test
   tests/test_flow.py::test_forced_scalar_matches_fma_backend must still pass or be
   generalized to the "neon" name.
6. Decide, based on measured parity, whether to unify the SSE and NEON kernels behind
   a small detail/simd4.hxx wrapper or keep two TUs. Do not unify before you have
   numbers for both.
7. Add a "Reproduction on Apple Silicon" subsection to PERFORMANCE_NOTES.md with
   the paired table, K sweep, and anything rejected. Do not touch MIGRATION_GUIDE.md
   unless the public API changes.
```

## Prompt: Windows x86-64 (MSVC)

```text
Context: bioimage-cpp, branch flow-optim-fable (commit 54fd4a9). The flow tracer
(`bic.flow.compute_flow_density`) has a packed-channel SSE+FMA kernel in
src/cpp/flow/flow_density_fma.cxx that CMake already builds under MSVC with
/arch:AVX2 and dispatches at runtime via __cpuid/_xgetbv in
include/bioimage_cpp/flow/flow_density.hxx (`runtime_fma_supported`,
`try_trace_all_fma`). It has only been measured with GCC on Linux (-30..-40 %).
Read development/flow/PERFORMANCE_NOTES.md, section "Packed-channel FMA tracer
(2026-09-02)", especially "Codegen trap" and the residual MSVC caveats.

Goal: validate and, where needed, tune the kernel under MSVC on Windows x86-64, and
record paired numbers in PERFORMANCE_NOTES.md.

Steps:
1. Build with the MSVC toolchain cibuildwheel uses (pip install -e . --no-build-isolation
   from a VS developer shell). Confirm `_core._flow_trace_backend()` reports "fma".
   Note that CMakeLists.txt passes -O3, which cl ignores; confirm /O2 /Ob2 are in
   effect for the Release config.
2. Make development/flow/paired_bench.py and differential_check.py run on Windows:
   they call `taskset` when --cpu is given, so add a Windows pinning branch
   (psutil cpu_affinity or `start /affinity`) or run without --cpu. Set the power
   plan to High performance. Calibrate with --a X.pyd --b X.pyd; every row must read
   "noise" before any A/B is trusted.
3. Static codegen check with `dumpbin /disasm` on the object files: (a) every
   BIOIMAGE_FORCE_INLINE helper (`sample_flow`, `round_to_flat_index`,
   `position_is_in_mask`, `trace_particle`, `trace_particle_block`, the TU's
   `step_packed`/`sample_packed`) must be inlined into the hot loop; (b) no VEX-encoded
   instructions may appear outside the FMA TU's own functions, since /OPT:ICF and
   COMDAT folding can otherwise pick an AVX copy of a shared helper for the
   portable path. If (a) fails, that is the first thing to fix (on GCC it cost 1.5-2x).
4. Paired A/B: build the baseline commit 492ff9b and the branch into separate dirs,
   swap the two _core .pyd files, run paired_bench.py on fixture3d/fixture2d and the
   adversarial cases at 1T and at the physical core count. Also measure
   BIOIMAGE_CPP_FLOW_FORCE_SCALAR=1 against the branch: on MSVC /fp:precise the
   scalar-FMA fallback is not expected to contract, so it may equal the portable kernel.
5. Gate: differential_check.py base vs branch and fma vs FORCE_SCALAR (400 cases,
   sums equal, diff fraction below 1 %); check_flow_density.py --dim both; full
   pytest. Report whether densities are bitwise identical to the GCC build on the
   fixtures, and if not, whether the difference is confined to contraction order.
6. Only if step 3 or 4 shows a real gap: tune (inlining pragmas, /Ob3, lane count K),
   each change measured paired and gated as in step 5.
7. Add a "Reproduction on Windows/MSVC" subsection to PERFORMANCE_NOTES.md with the
   paired table, the dumpbin findings, and anything rejected.
```
