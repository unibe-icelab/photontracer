// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#ifndef COMPLEX_F_CUH
#define COMPLEX_F_CUH

struct Complexf {
    float re;
    float im;

    __host__ __device__ Complexf() = default;
    __host__ __device__ Complexf(float r, float i) : re(r), im(i) {}

    __host__ __device__ float real() const { return re; }
    __host__ __device__ float imag() const { return im; }
};

// basic ops
__host__ __device__ inline Complexf conj(Complexf a) {
    return Complexf(a.re, -a.im);
}

__host__ __device__ inline float norm(Complexf a) {
    return a.re * a.re + a.im * a.im;
}

// operators
__host__ __device__ inline Complexf operator+(Complexf a, Complexf b) {
    return Complexf(a.re + b.re, a.im + b.im);
}

__host__ __device__ inline Complexf operator-(Complexf a, Complexf b) {
    return Complexf(a.re - b.re, a.im - b.im);
}

__host__ __device__ inline Complexf operator*(Complexf a, Complexf b) {
    return Complexf(
        a.re * b.re - a.im * b.im,
        a.re * b.im + a.im * b.re
    );
}

__host__ __device__ inline Complexf operator*(float s, Complexf a) {
    return Complexf(s * a.re, s * a.im);
}

__host__ __device__ inline Complexf operator*(Complexf a, float s) {
    return Complexf(s * a.re, s * a.im);
}

__host__ __device__ inline Complexf operator/(Complexf a, float s) {
    return Complexf(a.re / s, a.im / s);
}

__host__ __device__ inline Complexf operator/(Complexf a, Complexf b) {
    float denom = norm(b);
    return Complexf(
        (a.re * b.re + a.im * b.im) / denom,
        (a.im * b.re - a.re * b.im) / denom
    );
}
#endif // COMPLEX_F_CUH