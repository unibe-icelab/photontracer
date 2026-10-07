// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

// Size of the OptiX payload of the ray tracing pipeline, shared by the host (pipeline creation)
// and the device code. The payload is
//   0-2 origin, 3-5 direction, 6-9 Stokes vector, 10-12 Q- axis, 13 ray state,
//   14 medium history, 15 optical path length of the last segment,
// followed by the state of the random number generator.

constexpr int PAYLOAD_VALUES_WITHOUT_RNG = 16;

/// 32-bit words of the generator state in the payload
constexpr int PCG32_PAYLOAD_WORDS = 4;    // state and increment, 64 bit each
constexpr int MRG32K3A_PAYLOAD_WORDS = 6; // two times three 32-bit values
