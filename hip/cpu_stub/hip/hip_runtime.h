#ifndef UMBRA_CPU_HIP_RUNTIME_STUB_H
#define UMBRA_CPU_HIP_RUNTIME_STUB_H

/* Host-only syntax/correctness harness for device leaf functions. */
#define __device__
#define __forceinline__ inline __attribute__((always_inline))

#endif
