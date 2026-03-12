// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include "photontracer.h"
#include <cuda_runtime.h>
#include <optix_device.h>

__device__ __forceinline__ uint32_t ptLinearizeLaunchIndex(const uint3 idx, const uint3 dims)
{
    return idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;
}

__device__ __forceinline__ uint32_t ptCurrentLaunchIndex()
{
    return ptLinearizeLaunchIndex(optixGetLaunchIndex(), optixGetLaunchDimensions());
}

#ifdef DEBUG_PRINT_KERNEL

__device__ inline void dbg_write_char(char *dst, int &len, int maxLen, char c)
{
    if (len < maxLen)
        dst[len++] = c;
}

__device__ inline void dbg_write_cstr(char *dst, int &len, int maxLen, const char *s)
{
    if (!s)
        return;
    while (*s && len < maxLen)
    {
        dst[len++] = *s++;
    }
}

__device__ inline void dbg_write_uint(char *dst, int &len, int maxLen, unsigned int v)
{
    char tmp[16];
    int n = 0;

    if (v == 0)
    {
        dbg_write_char(dst, len, maxLen, '0');
        return;
    }
    while (v && n < (int)sizeof(tmp))
    {
        tmp[n++] = '0' + (v % 10u);
        v /= 10u;
    }
    for (int i = n - 1; i >= 0 && len < maxLen; --i)
    {
        dst[len++] = tmp[i];
    }
}

__device__ inline void dbg_write_uint64(char *dst, int &len, int maxLen, unsigned long long v)
{
    char tmp[32];
    int n = 0;

    if (v == 0ull)
    {
        dbg_write_char(dst, len, maxLen, '0');
        return;
    }
    while (v && n < (int)sizeof(tmp))
    {
        tmp[n++] = '0' + (int)(v % 10ull);
        v /= 10ull;
    }
    for (int i = n - 1; i >= 0 && len < maxLen; --i)
    {
        dst[len++] = tmp[i];
    }
}

__device__ inline void dbg_write_int(char *dst, int &len, int maxLen, int v)
{
    if (v < 0)
    {
        dbg_write_char(dst, len, maxLen, '-');
        unsigned int positive = (unsigned int)(-(long long)v);
        dbg_write_uint(dst, len, maxLen, positive);
    }
    else
    {
        dbg_write_uint(dst, len, maxLen, (unsigned int)v);
    }
}

__device__ inline void dbg_write_bool(char *dst, int &len, int maxLen, bool v)
{
    dbg_write_cstr(dst, len, maxLen, v ? "true" : "false");
}

__device__ inline void dbg_write_label(char *dst, int &len, int maxLen, const char *label)
{
    dbg_write_char(dst, len, maxLen, '[');
    dbg_write_cstr(dst, len, maxLen, label);
    dbg_write_cstr(dst, len, maxLen, "] ");
}

__device__ inline void dbg_write_float_fixed(char *dst, int &len, int maxLen, float value, int precision)
{
    if (!isfinite(value))
    {
        if (isnan(value))
        {
            dbg_write_cstr(dst, len, maxLen, "nan");
        }
        else if (value < 0.0f)
        {
            dbg_write_cstr(dst, len, maxLen, "-inf");
        }
        else
        {
            dbg_write_cstr(dst, len, maxLen, "inf");
        }
        return;
    }

    if (precision < 0)
        precision = 6;
    if (precision > 9)
        precision = 9;

    if (value < 0.0f)
    {
        dbg_write_char(dst, len, maxLen, '-');
        value = -value;
    }

    unsigned long long intPart = (unsigned long long)value;
    float fracPart = value - (float)intPart;
    dbg_write_uint64(dst, len, maxLen, intPart);

    if (precision == 0)
    {
        return;
    }

    dbg_write_char(dst, len, maxLen, '.');
    for (int i = 0; i < precision; ++i)
    {
        fracPart *= 10.0f;
        int digit = (int)fracPart;
        dbg_write_char(dst, len, maxLen, (char)('0' + digit));
        fracPart -= digit;
    }
}

__device__ inline void dbg_write_float_sci(char *dst, int &len, int maxLen, float value)
{
    if (!isfinite(value))
    {
        if (isnan(value))
        {
            dbg_write_cstr(dst, len, maxLen, "nan");
        }
        else if (value < 0.0f)
        {
            dbg_write_cstr(dst, len, maxLen, "-inf");
        }
        else
        {
            dbg_write_cstr(dst, len, maxLen, "inf");
        }
        return;
    }

    if (value == 0.0f)
    {
        dbg_write_cstr(dst, len, maxLen, "0.00000e+00");
        return;
    }

    bool negative = value < 0.0f;
    float absVal = negative ? -value : value;

    float expF = floorf(log10f(absVal));
    int exponent = (int)expF;
    float mantissa = absVal / powf(10.0f, (float)exponent);
    if (mantissa >= 10.0f)
    {
        mantissa *= 0.1f;
        ++exponent;
    }

    if (negative)
    {
        dbg_write_char(dst, len, maxLen, '-');
    }
    dbg_write_float_fixed(dst, len, maxLen, mantissa, 5);
    dbg_write_char(dst, len, maxLen, 'e');
    dbg_write_int(dst, len, maxLen, exponent);
}

#define DBG_LOG_WRITE(labelExpr, BODY)                                          \
    do                                                                          \
    {                                                                           \
        uint32_t _logIdx = ptCurrentLaunchIndex();                              \
        uint32_t _logBase = _logIdx * LOG_BYTES_PER_RAY;                        \
        uint32_t _logOff = params.deviceOutputBuffers.logOffsets[_logIdx];      \
        if (_logOff < LOG_BYTES_PER_RAY)                                        \
        {                                                                       \
            char *_logDst = &params.deviceOutputBuffers.logs[_logBase];         \
            int _logPos = (int)_logOff;                                         \
            int _logMax = (int)LOG_BYTES_PER_RAY;                               \
            dbg_write_label(_logDst, _logPos, _logMax, (labelExpr));            \
            BODY                                                                \
                dbg_write_char(_logDst, _logPos, _logMax, '\n');                \
            if (_logPos > _logMax)                                              \
                _logPos = _logMax;                                              \
            params.deviceOutputBuffers.logOffsets[_logIdx] = (uint32_t)_logPos; \
        }                                                                       \
    } while (0)

#define DBG_LOG_STRING(label, text) \
    DBG_LOG_WRITE(label, { dbg_write_cstr(_logDst, _logPos, _logMax, (text)); })

#define DBG_LOG_INT(label, value) \
    DBG_LOG_WRITE(label, { dbg_write_int(_logDst, _logPos, _logMax, (int)(value)); })

#define DBG_LOG_BOOL(label, value) \
    DBG_LOG_WRITE(label, { dbg_write_bool(_logDst, _logPos, _logMax, (bool)(value)); })

#define DBG_LOG_FLOAT(label, value) \
    DBG_LOG_WRITE(label, { dbg_write_float_sci(_logDst, _logPos, _logMax, (float)(value)); })

#define DBG_LOG_FLOAT3(label, value)                                     \
    do                                                                   \
    {                                                                    \
        float3 _logValue = (value);                                      \
        DBG_LOG_WRITE(label, {                                           \
            dbg_write_float_sci(_logDst, _logPos, _logMax, _logValue.x); \
            dbg_write_cstr(_logDst, _logPos, _logMax, ", ");             \
            dbg_write_float_sci(_logDst, _logPos, _logMax, _logValue.y); \
            dbg_write_cstr(_logDst, _logPos, _logMax, ", ");             \
            dbg_write_float_sci(_logDst, _logPos, _logMax, _logValue.z); \
        });                                                              \
    } while (0)

#define DBG_LOG_TEXT(text) \
    DBG_LOG_WRITE("log", { dbg_write_cstr(_logDst, _logPos, _logMax, (text)); })

#else
#define DBG_LOG_STRING(label, text) \
    do                              \
    {                               \
    } while (0)
#define DBG_LOG_INT(label, value) \
    do                            \
    {                             \
    } while (0)
#define DBG_LOG_BOOL(label, value) \
    do                             \
    {                              \
    } while (0)
#define DBG_LOG_FLOAT(label, value) \
    do                              \
    {                               \
    } while (0)
#define DBG_LOG_FLOAT3(label, value) \
    do                               \
    {                                \
    } while (0)
#define DBG_LOG_TEXT(text) \
    do                     \
    {                      \
    } while (0)
#endif