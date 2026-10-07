// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#ifndef COMPLEX_F_H
#define COMPLEX_F_H

#include "portable_math.h"

struct Complexf {
    float re;
    float im;

    PT_HD Complexf() = default;
    PT_HD Complexf(float r, float i) : re(r), im(i) {}

    PT_HD float real() const { return re; }
    PT_HD float imag() const { return im; }
};

// basic ops
PT_HD inline Complexf conj(Complexf a) {
    return Complexf(a.re, -a.im);
}

PT_HD inline float norm(Complexf a) {
    return a.re * a.re + a.im * a.im;
}

// operators
PT_HD inline Complexf operator+(Complexf a, Complexf b) {
    return Complexf(a.re + b.re, a.im + b.im);
}

PT_HD inline Complexf operator-(Complexf a, Complexf b) {
    return Complexf(a.re - b.re, a.im - b.im);
}

PT_HD inline Complexf operator*(Complexf a, Complexf b) {
    return Complexf(
        a.re * b.re - a.im * b.im,
        a.re * b.im + a.im * b.re
    );
}

PT_HD inline Complexf operator*(float s, Complexf a) {
    return Complexf(s * a.re, s * a.im);
}

PT_HD inline Complexf operator*(Complexf a, float s) {
    return Complexf(s * a.re, s * a.im);
}

PT_HD inline Complexf operator/(Complexf a, float s) {
    return Complexf(a.re / s, a.im / s);
}

PT_HD inline Complexf operator/(Complexf a, Complexf b) {
    float denom = norm(b);
    return Complexf(
        (a.re * b.re + a.im * b.im) / denom,
        (a.im * b.re - a.re * b.im) / denom
    );
}
#endif // COMPLEX_F_H