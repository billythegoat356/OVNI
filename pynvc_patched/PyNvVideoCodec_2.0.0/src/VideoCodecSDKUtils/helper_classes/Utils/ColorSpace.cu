/*
 * This copyright notice applies to this file only
 *
 * SPDX-FileCopyrightText: Copyright (c) 2010-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "ColorSpace.h"

#include <cuda.h>

__constant__ float matYuv2Rgb[3][3];
__constant__ float matRgb2Yuv[3][3];


void inline GetConstants(int iMatrix, float &wr, float &wb, int &black, int &white, int &max) {
    black = 16; white = 235;
    max = 255;

    switch (iMatrix)
    {
    case ColorSpaceStandard_BT709:
    default:
        wr = 0.2126f; wb = 0.0722f;
        break;

    case ColorSpaceStandard_FCC:
        wr = 0.30f; wb = 0.11f;
        break;

    case ColorSpaceStandard_BT470:
    case ColorSpaceStandard_BT601:
        wr = 0.2990f; wb = 0.1140f;
        break;

    case ColorSpaceStandard_SMPTE240M:
        wr = 0.212f; wb = 0.087f;
        break;

    case ColorSpaceStandard_BT2020:
    case ColorSpaceStandard_BT2020C:
        wr = 0.2627f; wb = 0.0593f;
        // 10-bit only
        black = 64 << 6; white = 940 << 6;
        max = (1 << 16) - 1;
        break;
    }
}

void SetMatYuv2Rgb(int iMatrix, CUstream stream = nullptr) {
    float wr, wb;
    int black, white, max;
    GetConstants(iMatrix, wr, wb, black, white, max);
    float mat[3][3] = {
        1.0f, 0.0f, (1.0f - wr) / 0.5f,
        1.0f, -wb * (1.0f - wb) / 0.5f / (1 - wb - wr), -wr * (1 - wr) / 0.5f / (1 - wb - wr),
        1.0f, (1.0f - wb) / 0.5f, 0.0f,
    };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat[i][j] = (float)(1.0 * max / (white - black) * mat[i][j]);
        }
    }
    cudaMemcpyToSymbolAsync(matYuv2Rgb, mat, sizeof(mat), 0, cudaMemcpyHostToDevice, stream);
}

void SetMatRgb2Yuv(int iMatrix) {
    float wr, wb;
    int black, white, max;
    GetConstants(iMatrix, wr, wb, black, white, max);
    float mat[3][3] = {
        wr, 1.0f - wb - wr, wb,
        -0.5f * wr / (1.0f - wb), -0.5f * (1 - wb - wr) / (1.0f - wb), 0.5f,
        0.5f, -0.5f * (1.0f - wb - wr) / (1.0f - wr), -0.5f * wb / (1.0f - wr),
    };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat[i][j] = (float)(1.0 * (white - black) / max * mat[i][j]);
        }
    }
    cudaMemcpyToSymbol(matRgb2Yuv, mat, sizeof(mat));
}

template<class T>
__device__ static T Clamp(T x, T lower, T upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

template<class Rgb, class YuvUnit>
__device__ inline Rgb YuvToRgbForPixel(YuvUnit y, YuvUnit u, YuvUnit v) {
    const int 
        low = 1 << (sizeof(YuvUnit) * 8 - 4),
        mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    float fy = (int)y - low, fu = (int)u - mid, fv = (int)v - mid;
    const float maxf = (1 << sizeof(YuvUnit) * 8) - 1.0f;
    YuvUnit 
        r = (YuvUnit)Clamp(matYuv2Rgb[0][0] * fy + matYuv2Rgb[0][1] * fu + matYuv2Rgb[0][2] * fv, 0.0f, maxf),
        g = (YuvUnit)Clamp(matYuv2Rgb[1][0] * fy + matYuv2Rgb[1][1] * fu + matYuv2Rgb[1][2] * fv, 0.0f, maxf),
        b = (YuvUnit)Clamp(matYuv2Rgb[2][0] * fy + matYuv2Rgb[2][1] * fu + matYuv2Rgb[2][2] * fv, 0.0f, maxf);
    
    Rgb rgb{};
    const int nShift = abs((int)sizeof(YuvUnit) - (int)sizeof(rgb.c.r)) * 8;
    if (sizeof(YuvUnit) >= sizeof(rgb.c.r)) {
        rgb.c.r = r >> nShift;
        rgb.c.g = g >> nShift;
        rgb.c.b = b >> nShift;
    } else {
        rgb.c.r = r << nShift;
        rgb.c.g = g << nShift;
        rgb.c.b = b << nShift;
    }
    return rgb;
}

template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void YuvToRgbKernel(uint8_t *pYuv, int nYuvPitch, uint8_t *pRgb, int nRgbPitch, int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t *pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;
    uint8_t *pDst = pRgb + x * sizeof(Rgb) + y * nRgbPitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)pSrc;
    YuvUnitx2 l1 = *(YuvUnitx2 *)(pSrc + nYuvPitch);
    YuvUnitx2 ch = *(YuvUnitx2 *)(pSrc + (nHeight - y / 2) * nYuvPitch);

    *(RgbIntx2 *)pDst = RgbIntx2 {
        YuvToRgbForPixel<Rgb>(l0.x, ch.x, ch.y).d,
        YuvToRgbForPixel<Rgb>(l0.y, ch.x, ch.y).d,
    };
    *(RgbIntx2 *)(pDst + nRgbPitch) = RgbIntx2 {
        YuvToRgbForPixel<Rgb>(l1.x, ch.x, ch.y).d, 
        YuvToRgbForPixel<Rgb>(l1.y, ch.x, ch.y).d,
    };
}

template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void Yuv422ToRgbKernel(uint8_t *pYuv, int nYuvPitch, uint8_t *pRgb, int nRgbPitch, int nWidth, int nSurfaceHeight, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }

    uint8_t *pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;
    uint8_t *pDst = pRgb + x * sizeof(Rgb) + y * nRgbPitch;

    YuvUnitx2 l = *(YuvUnitx2 *)pSrc;
    YuvUnitx2 ch = *(YuvUnitx2 *)(pSrc + (nSurfaceHeight * nYuvPitch));

    *(RgbIntx2 *)pDst = RgbIntx2 {
        YuvToRgbForPixel<Rgb>(l.x, ch.x, ch.y).d,
        YuvToRgbForPixel<Rgb>(l.y, ch.x, ch.y).d,
    };
}

template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void Yuv444ToRgbKernel(uint8_t *pYuv, int nYuvPitch, uint8_t *pRgb, int nRgbPitch, int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (x + 1 >= nWidth || y  >= nHeight) {
        return;
    }

    uint8_t *pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;
    uint8_t *pDst = pRgb + x * sizeof(Rgb) + y * nRgbPitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)pSrc;
    YuvUnitx2 ch1 = *(YuvUnitx2 *)(pSrc + (nHeight * nYuvPitch));
    YuvUnitx2 ch2 = *(YuvUnitx2 *)(pSrc + (2 * nHeight * nYuvPitch));

    *(RgbIntx2 *)pDst = RgbIntx2{
        YuvToRgbForPixel<Rgb>(l0.x, ch1.x, ch2.x).d,
        YuvToRgbForPixel<Rgb>(l0.y, ch1.y, ch2.y).d,
    };
}

template<class YuvUnitx2, class Rgb, class RgbUnitx2>
__global__ static void YuvToRgbPlanarKernel(uint8_t *pYuv, int nYuvPitch, uint8_t *pRgbp, int nRgbpPitch, int nWidth, int nHeight, int nDstHeight = 0) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (nDstHeight == 0)
        nDstHeight = nHeight;

    if (x + 1 >= nWidth || y + 1 >= nDstHeight) {
        return;
    }

    uint8_t *pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)pSrc;
    YuvUnitx2 l1 = *(YuvUnitx2 *)(pSrc + nYuvPitch);
    YuvUnitx2 ch = *(YuvUnitx2 *)(pSrc + (nHeight - y / 2) * nYuvPitch);

    Rgb rgb0 = YuvToRgbForPixel<Rgb>(l0.x, ch.x, ch.y),
        rgb1 = YuvToRgbForPixel<Rgb>(l0.y, ch.x, ch.y),
        rgb2 = YuvToRgbForPixel<Rgb>(l1.x, ch.x, ch.y),
        rgb3 = YuvToRgbForPixel<Rgb>(l1.y, ch.x, ch.y);

    uint8_t *pDst = pRgbp + x * sizeof(RgbUnitx2) / 2 + y * nRgbpPitch;
    *(RgbUnitx2 *)pDst = RgbUnitx2 {rgb0.v.x, rgb1.v.x};
    *(RgbUnitx2 *)(pDst + nRgbpPitch) = RgbUnitx2 {rgb2.v.x, rgb3.v.x};
    pDst += nRgbpPitch * nDstHeight;
    *(RgbUnitx2 *)pDst = RgbUnitx2 {rgb0.v.y, rgb1.v.y};
    *(RgbUnitx2 *)(pDst + nRgbpPitch) = RgbUnitx2 {rgb2.v.y, rgb3.v.y};
    pDst += nRgbpPitch * nDstHeight;
    *(RgbUnitx2 *)pDst = RgbUnitx2 {rgb0.v.z, rgb1.v.z};
    *(RgbUnitx2 *)(pDst + nRgbpPitch) = RgbUnitx2 {rgb2.v.z, rgb3.v.z};
}

template<class YuvUnitx2, class Rgb, class RgbUnitx2>
__global__ static void Yuv422ToRgbPlanarKernel(uint8_t *pYuv, int nYuvPitch, uint8_t *pRgbp, int nRgbpPitch, int nWidth, int nHeight, int nDstHeight = 0) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (nDstHeight == 0)
        nDstHeight = nHeight;
    if (x + 1 >= nWidth || y >= nDstHeight) {
        return;
    }

    uint8_t *pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;

    YuvUnitx2 l = *(YuvUnitx2 *)pSrc;
    YuvUnitx2 ch = *(YuvUnitx2 *)(pSrc + nHeight * nYuvPitch);

    Rgb rgb0 = YuvToRgbForPixel<Rgb>(l.x, ch.x, ch.y),
        rgb1 = YuvToRgbForPixel<Rgb>(l.y, ch.x, ch.y);

    uint8_t *pDst = pRgbp + x * sizeof(RgbUnitx2) / 2 + y * nRgbpPitch;
    *(RgbUnitx2 *)pDst = RgbUnitx2 {rgb0.v.x, rgb1.v.x};

    pDst += nRgbpPitch * nDstHeight;
    *(RgbUnitx2 *)pDst = RgbUnitx2 {rgb0.v.y, rgb1.v.y};

    pDst += nRgbpPitch * nDstHeight;
    *(RgbUnitx2 *)pDst = RgbUnitx2 {rgb0.v.z, rgb1.v.z};
}

template<class YuvUnitx2, class Rgb, class RgbUnitx2>
__global__ static void Yuv444ToRgbPlanarKernel(uint8_t *pYuv, int nYuvPitch, uint8_t *pRgbp, int nRgbpPitch, int nWidth, int nHeight, int nDstHeight = 0) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (nDstHeight == 0)
        nDstHeight = nHeight;
    if (x + 1 >= nWidth || y >= nDstHeight) {
        return;
    }

    uint8_t *pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;

    YuvUnitx2 l0 = *(YuvUnitx2 *)pSrc;
    YuvUnitx2 ch1 = *(YuvUnitx2 *)(pSrc + (nHeight * nYuvPitch));
    YuvUnitx2 ch2 = *(YuvUnitx2 *)(pSrc + (2 * nHeight * nYuvPitch));

    Rgb rgb0 = YuvToRgbForPixel<Rgb>(l0.x, ch1.x, ch2.x),
        rgb1 = YuvToRgbForPixel<Rgb>(l0.y, ch1.y, ch2.y);

    uint8_t *pDst = pRgbp + x * sizeof(RgbUnitx2) / 2 + y * nRgbpPitch;
    *(RgbUnitx2 *)pDst = RgbUnitx2{ rgb0.v.x, rgb1.v.x };

    pDst += nRgbpPitch * nDstHeight;
    *(RgbUnitx2 *)pDst = RgbUnitx2{ rgb0.v.y, rgb1.v.y };

    pDst += nRgbpPitch * nDstHeight;
    *(RgbUnitx2 *)pDst = RgbUnitx2{ rgb0.v.z, rgb1.v.z };
}

template <class COLOR32>
void Nv12ToColor32(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    YuvToRgbKernel<uchar2, COLOR32, uint2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpNv12, nNv12Pitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR64>
void Nv12ToColor64(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    YuvToRgbKernel<uchar2, COLOR64, ulonglong2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpNv12, nNv12Pitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR32>
void YUV444ToColor32(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    Yuv444ToRgbKernel<uchar2, COLOR32, uint2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2) >>>
        (dpYUV444, nPitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR64>
void YUV444ToColor64(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    Yuv444ToRgbKernel<uchar2, COLOR64, ulonglong2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2) >>>
        (dpYUV444, nPitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR32>
void P016ToColor32(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    YuvToRgbKernel<ushort2, COLOR32, uint2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpP016, nP016Pitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR64>
void P016ToColor64(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    YuvToRgbKernel<ushort2, COLOR64, ulonglong2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpP016, nP016Pitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR32>
void YUV444P16ToColor32(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    Yuv444ToRgbKernel<ushort2, COLOR32, uint2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2) >>>
        (dpYUV444, nPitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR64>
void YUV444P16ToColor64(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    Yuv444ToRgbKernel<ushort2, COLOR64, ulonglong2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2) >>>
        (dpYUV444, nPitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR32>
void Nv12ToColorPlanar(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    YuvToRgbPlanarKernel<uchar2, COLOR32, uchar2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpNv12, nNv12Pitch, dpBgrp, nBgrpPitch, nWidth, nHeight);
}

template <class COLOR32>
void P016ToColorPlanar(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    YuvToRgbPlanarKernel<ushort2, COLOR32, uchar2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpP016, nP016Pitch, dpBgrp, nBgrpPitch, nWidth, nHeight);
}

template <class COLOR32>
void YUV444ToColorPlanar(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    Yuv444ToRgbPlanarKernel<uchar2, COLOR32, uchar2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2) >>>
        (dpYUV444, nPitch, dpBgrp, nBgrpPitch, nWidth, nHeight);
}

template <class COLOR32>
void YUV444P16ToColorPlanar(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    Yuv444ToRgbPlanarKernel<ushort2, COLOR32, uchar2>
        << <dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2) >> >
        (dpYUV444, nPitch, dpBgrp, nBgrpPitch, nWidth, nHeight);
}

// Explicit Instantiation
template void Nv12ToColor32<BGRA32>(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void Nv12ToColor32<RGBA32>(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void Nv12ToColor64<BGRA64>(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void Nv12ToColor64<RGBA64>(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444ToColor32<BGRA32>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444ToColor32<RGBA32>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444ToColor64<BGRA64>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444ToColor64<RGBA64>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void P016ToColor32<BGRA32>(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void P016ToColor32<RGBA32>(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void P016ToColor64<BGRA64>(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void P016ToColor64<RGBA64>(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444P16ToColor32<BGRA32>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444P16ToColor32<RGBA32>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444P16ToColor64<BGRA64>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444P16ToColor64<RGBA64>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void Nv12ToColorPlanar<BGRA32>(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix);
template void Nv12ToColorPlanar<RGBA32>(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix);
template void P016ToColorPlanar<BGRA32>(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix);
template void P016ToColorPlanar<RGBA32>(uint8_t *dpP016, int nP016Pitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444ToColorPlanar<BGRA32>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444ToColorPlanar<RGBA32>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444P16ToColorPlanar<BGRA32>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444P16ToColorPlanar<RGBA32>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgrp, int nBgrpPitch, int nWidth, int nHeight, int iMatrix);


template <class COLOR24>
void Nv12ToColor24(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpRGB, int nRGBPitch, int nWidth, int nHeight, int iMatrix, CUstream stream) {
    SetMatYuv2Rgb(iMatrix, stream);
    YuvToRgbKernel<uchar2, COLOR24, uchar3_2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
        (dpNv12, nNv12Pitch, dpRGB, nRGBPitch, nWidth, nHeight);
}
template void Nv12ToColor24<RGB24>(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpRGB, int nRGBPitch, int nWidth, int nHeight, int iMatrix, CUstream stream);

template <class COLOR24>
void Nv12ToColor24Planar(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpRGBP, int nRGBPPitch, int nWidth, int nHeight, int nDstHeight, int iMatrix, CUstream stream) {
    SetMatYuv2Rgb(iMatrix, stream);
    YuvToRgbPlanarKernel<uchar2, COLOR24, uchar2>
        <<<dim3((nWidth + 63) / 32 / 2, (nDstHeight + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
        (dpNv12, nNv12Pitch, dpRGBP, nRGBPPitch, nWidth, nHeight, nDstHeight);
}
template void Nv12ToColor24Planar<RGB24>(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpRGBP, int nRGBPPitch, int nWidth, int nHeight, int nDstHeight, int iMatrix, CUstream stream);

template <class COLOR24>
void P016ToColor24(uint8_t *dpP016, int nP016Pitch, uint8_t *dpRGB, int nRGBPitch, int nWidth, int nHeight, int iMatrix, CUstream stream) {
    SetMatYuv2Rgb(iMatrix, stream);
    YuvToRgbKernel<ushort2, COLOR24, uchar3_2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
        (dpP016, nP016Pitch, dpRGB, nRGBPitch, nWidth, nHeight);
}
template void P016ToColor24<RGB24>(uint8_t *dpP016, int nP016Pitch, uint8_t *dpRGB, int nRGBPitch, int nWidth, int nHeight, int iMatrix, CUstream stream);

template <class COLOR24>
void P016ToColor24Planar(uint8_t *dpP016, int nP016Pitch, uint8_t *dpRGBP, int nRGBPPitch, int nWidth, int nHeight, int nDstHeight, int iMatrix, CUstream stream) {
    SetMatYuv2Rgb(iMatrix, stream);
    YuvToRgbPlanarKernel<ushort2, COLOR24, uchar2>
        <<<dim3((nWidth + 63) / 32 / 2, (nDstHeight + 3) / 2 / 2), dim3(32, 2), 0, stream>>>
        (dpP016, nP016Pitch, dpRGBP, nRGBPPitch, nWidth, nHeight, nDstHeight);
}
template void P016ToColor24Planar<RGB24>(uint8_t *dpP016, int nP016Pitch, uint8_t *dpRGBP, int nRGBPPitch, int nWidth, int nHeight, int nDstHeight, int iMatrix, CUstream stream);

template <class COLOR24>
void YUV444ToColor24(uint8_t *dpYUV444, int nPitch, uint8_t *dpRGB, int nRGBPitch, int nWidth, int nHeight, int iMatrix, CUstream stream) {
    SetMatYuv2Rgb(iMatrix, stream);
    Yuv444ToRgbKernel<uchar2, COLOR24, uchar3_2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2), 0, stream >>>
        (dpYUV444, nPitch, dpRGB, nRGBPitch, nWidth, nHeight);
}
template void YUV444ToColor24<RGB24>(uint8_t *dpYUV444, int nPitch, uint8_t *dpRGB, int nRGBPitch, int nWidth, int nHeight, int iMatrix, CUstream stream);

template <class COLOR24>
void YUV444ToColor24Planar(uint8_t *dpYUV444, int nPitch, uint8_t *dpRGBP, int nRGBPPitch, int nWidth, int nHeight, int nDstHeight, int iMatrix, CUstream stream) {
    SetMatYuv2Rgb(iMatrix, stream);
    Yuv444ToRgbPlanarKernel<uchar2, COLOR24, uchar2>
        <<<dim3((nWidth + 63) / 32 / 2, (nDstHeight + 3) / 2), dim3(32, 2), 0, stream>>>
        (dpYUV444, nPitch, dpRGBP, nRGBPPitch, nWidth, nHeight, nDstHeight);
}
template void YUV444ToColor24Planar<RGB24>(uint8_t *dpYUV444, int nPitch, uint8_t *dpRGBP, int nRGBPPitch, int nWidth, int nHeight, int nDstHeight, int iMatrix, CUstream stream);

template <class COLOR24>
void YUV444P16ToColor24(uint8_t *dpYUV444, int nPitch, uint8_t *dpRGB, int nRGBPitch, int nWidth, int nHeight, int iMatrix, CUstream stream) {
    SetMatYuv2Rgb(iMatrix, stream);
    Yuv444ToRgbKernel<ushort2, COLOR24, uchar3_2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2), 0, stream >>>
        (dpYUV444, nPitch, dpRGB, nRGBPitch, nWidth, nHeight);
}
template void YUV444P16ToColor24<RGB24>(uint8_t *dpYUV444, int nPitch, uint8_t *dpRGB, int nRGBPitch, int nWidth, int nHeight, int iMatrix, CUstream stream);

template <class COLOR24>
void YUV444P16ToColor24Planar(uint8_t *dpYUV444, int nPitch, uint8_t *dpRGBP, int nRGBPPitch, int nWidth, int nHeight, int nDstHeight, int iMatrix, CUstream stream) {
    SetMatYuv2Rgb(iMatrix, stream);
    Yuv444ToRgbPlanarKernel<ushort2, COLOR24, uchar2>
        << <dim3((nWidth + 63) / 32 / 2, (nDstHeight + 3) / 2), dim3(32, 2), 0, stream >> >
        (dpYUV444, nPitch, dpRGBP, nRGBPPitch, nWidth, nHeight, nDstHeight);
}
template void YUV444P16ToColor24Planar<RGB24>(uint8_t *dpYUV444, int nPitch, uint8_t *dpRGBP, int nRGBPPitch, int nWidth, int nHeight, int nDstHeight, int iMatrix, CUstream stream);

template <class COLOR24>
void Nv16ToColor24(uint8_t *dpNv16, int nNv16Pitch, uint8_t *dpRGB, int nRGBPitch, int nWidth, int nSurfaceHeight, int nHeight, int iMatrix, CUstream stream) {
    SetMatYuv2Rgb(iMatrix, stream);
    Yuv422ToRgbKernel<uchar2, COLOR24, uchar3_2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 1) / 2), dim3(32, 2), 0, stream>>>
        (dpNv16, nNv16Pitch, dpRGB, nRGBPitch, nWidth, nSurfaceHeight, nHeight);
}
template void Nv16ToColor24<RGB24>(uint8_t *dpNv16, int nNv16Pitch, uint8_t *dpRGB, int nRGBPitch, int nWidth, int nSurfaceHeight, int nHeight, int iMatrix, CUstream stream);

template <class COLOR24>
void Nv16ToColor24Planar(uint8_t *dpNv16, int nNv16Pitch, uint8_t *dpRGBP, int nRGBPPitch, int nWidth, int nHeight, int nDstHeight, int iMatrix, CUstream stream) {
    SetMatYuv2Rgb(iMatrix, stream);
    Yuv422ToRgbPlanarKernel<uchar2, COLOR24, uchar2>
        <<<dim3((nWidth + 63) / 32 / 2, (nDstHeight + 1) / 2), dim3(32, 2), 0, stream>>>
        (dpNv16, nNv16Pitch, dpRGBP, nRGBPPitch, nWidth, nHeight, nDstHeight);
}
template void Nv16ToColor24Planar<RGB24>(uint8_t *dpNv16, int nNv16Pitch, uint8_t *dpRGBP, int nRGBPPitch, int nWidth, int nHeight, int nDstHeight, int iMatrix, CUstream stream);

template <class COLOR24>
void P216ToColor24(uint8_t *dpP216, int nP216Pitch, uint8_t *dpRGB, int nRGBPitch, int nWidth, int nSurfaceHeight, int nHeight, int iMatrix, CUstream stream) {
    SetMatYuv2Rgb(iMatrix, stream);
    Yuv422ToRgbKernel<ushort2, COLOR24, uchar3_2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 1) / 2), dim3(32, 2), 0, stream>>>
        (dpP216, nP216Pitch, dpRGB, nRGBPitch, nWidth, nSurfaceHeight, nHeight);
}
template void P216ToColor24<RGB24>(uint8_t *dpP216, int nP216Pitch, uint8_t *dpRGB, int nRGBPitch, int nWidth, int nSurfaceHeight, int nHeight, int iMatrix, CUstream stream);

template <class COLOR24>
void P216ToColor24Planar(uint8_t *dpP216, int nP216Pitch, uint8_t *dpRGBP, int nRGBPPitch, int nWidth, int nHeight, int nDstHeight, int iMatrix, CUstream stream) {
    SetMatYuv2Rgb(iMatrix, stream);
    Yuv422ToRgbPlanarKernel<ushort2, COLOR24, uchar2>
        <<<dim3((nWidth + 63) / 32 / 2, (nDstHeight + 1) / 2), dim3(32, 2), 0, stream>>>
        (dpP216, nP216Pitch, dpRGBP, nRGBPPitch, nWidth, nHeight, nDstHeight);
}
template void P216ToColor24Planar<RGB24>(uint8_t *dpP216, int nP216Pitch, uint8_t *dpRGBP, int nRGBPPitch, int nWidth, int nHeight, int nDstHeight, int iMatrix, CUstream stream);

template<class YuvUnit, class RgbUnit>
__device__ inline YuvUnit RgbToY(RgbUnit r, RgbUnit g, RgbUnit b) {
    const YuvUnit low = 1 << (sizeof(YuvUnit) * 8 - 4);
    return matRgb2Yuv[0][0] * r + matRgb2Yuv[0][1] * g + matRgb2Yuv[0][2] * b + low;
}

template<class YuvUnit, class RgbUnit>
__device__ inline YuvUnit RgbToU(RgbUnit r, RgbUnit g, RgbUnit b) {
    const YuvUnit mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    return matRgb2Yuv[1][0] * r + matRgb2Yuv[1][1] * g + matRgb2Yuv[1][2] * b + mid;
}

template<class YuvUnit, class RgbUnit>
__device__ inline YuvUnit RgbToV(RgbUnit r, RgbUnit g, RgbUnit b) {
    const YuvUnit mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    return matRgb2Yuv[2][0] * r + matRgb2Yuv[2][1] * g + matRgb2Yuv[2][2] * b + mid;
}

template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void RgbToYuvKernel(uint8_t *pRgb, int nRgbPitch, uint8_t *pYuv, int nYuvPitch, int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t *pSrc = pRgb + x * sizeof(Rgb) + y * nRgbPitch;
    RgbIntx2 int2a = *(RgbIntx2 *)pSrc;
    RgbIntx2 int2b = *(RgbIntx2 *)(pSrc + nRgbPitch);

    Rgb rgb[4] = {int2a.x, int2a.y, int2b.x, int2b.y};
    decltype(Rgb::c.r)
        r = (rgb[0].c.r + rgb[1].c.r + rgb[2].c.r + rgb[3].c.r) / 4,
        g = (rgb[0].c.g + rgb[1].c.g + rgb[2].c.g + rgb[3].c.g) / 4,
        b = (rgb[0].c.b + rgb[1].c.b + rgb[2].c.b + rgb[3].c.b) / 4;

    uint8_t *pDst = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;
    *(YuvUnitx2 *)pDst = YuvUnitx2 {
        RgbToY<decltype(YuvUnitx2::x)>(rgb[0].c.r, rgb[0].c.g, rgb[0].c.b),
        RgbToY<decltype(YuvUnitx2::x)>(rgb[1].c.r, rgb[1].c.g, rgb[1].c.b),
    };
    *(YuvUnitx2 *)(pDst + nYuvPitch) = YuvUnitx2 {
        RgbToY<decltype(YuvUnitx2::x)>(rgb[2].c.r, rgb[2].c.g, rgb[2].c.b),
        RgbToY<decltype(YuvUnitx2::x)>(rgb[3].c.r, rgb[3].c.g, rgb[3].c.b),
    };
    *(YuvUnitx2 *)(pDst + (nHeight - y / 2) * nYuvPitch) = YuvUnitx2 {
        RgbToU<decltype(YuvUnitx2::x)>(r, g, b), 
        RgbToV<decltype(YuvUnitx2::x)>(r, g, b),
    };
}

void Bgra64ToP016(uint8_t *dpBgra, int nBgraPitch, uint8_t *dpP016, int nP016Pitch, int nWidth, int nHeight, int iMatrix) {
    SetMatRgb2Yuv(iMatrix);
    RgbToYuvKernel<ushort2, BGRA64, ulonglong2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpBgra, nBgraPitch, dpP016, nP016Pitch, nWidth, nHeight);
}
