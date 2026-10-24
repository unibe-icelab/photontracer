// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <atomic>
#include <iostream>

/// Whether the library reports progress (device in use, compile and launch times); off by default
inline std::atomic<bool> &verboseOutput()
{
    static std::atomic<bool> verbose{false};
    return verbose;
}

/// Writes the arguments and a line break to stdout if verbose output is on
template <typename... Args>
void logInfo(const Args &...args)
{
    if (verboseOutput())
    {
        (std::cout << ... << args) << std::endl;
    }
}
