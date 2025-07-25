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

#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>

#include "../../../Interface/nvcuvid.h"
#include "NvDecoder/NvDecoder.h"
#include "ColorSpace.h"

#include <iterator>
#include <fstream>

#define __ALIGN_MASK(x,mask)    (((x)+(mask))&~(mask))
#define ALIGN(x,a)              __ALIGN_MASK(x,(int)a-1)

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();


#define START_TIMER auto start = logger->ShouldLogFor(DEBUG) ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point();

#define STOP_TIMER(print_message) { \
    if (logger->ShouldLogFor(DEBUG)) { \
        elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>( \
            std::chrono::high_resolution_clock::now() - start).count(); \
        LOG(DEBUG) << print_message << elapsedTime << " ms"; \
    } \
}

#ifndef ENABLE_PERF
std::mutex NvDecoderPerf::m_initMutex;
std::condition_variable NvDecoderPerf::m_cvInit;
uint32_t NvDecoderPerf::m_sessionInitCounter = 0;
uint32_t NvDecoderPerf::m_sessionCount = 1;
#define ENABLE_PERF
#endif // !ENABLE_PERF

static const char * GetVideoCodecString(cudaVideoCodec eCodec) {
    static struct {
        cudaVideoCodec eCodec;
        const char *name;
    } aCodecName [] = {
        { cudaVideoCodec_MPEG1,     "MPEG-1"       },
        { cudaVideoCodec_MPEG2,     "MPEG-2"       },
        { cudaVideoCodec_MPEG4,     "MPEG-4 (ASP)" },
        { cudaVideoCodec_VC1,       "VC-1/WMV"     },
        { cudaVideoCodec_H264,      "AVC/H.264"    },
        { cudaVideoCodec_JPEG,      "M-JPEG"       },
        { cudaVideoCodec_H264_SVC,  "H.264/SVC"    },
        { cudaVideoCodec_H264_MVC,  "H.264/MVC"    },
        { cudaVideoCodec_HEVC,      "H.265/HEVC"   },
        { cudaVideoCodec_VP8,       "VP8"          },
        { cudaVideoCodec_VP9,       "VP9"          },
        { cudaVideoCodec_AV1,       "AV1"          },
        { cudaVideoCodec_NumCodecs, "Invalid"      },
        { cudaVideoCodec_YUV420,    "YUV  4:2:0"   },
        { cudaVideoCodec_YV12,      "YV12 4:2:0"   },
        { cudaVideoCodec_NV12,      "NV12 4:2:0"   },
        { cudaVideoCodec_YUYV,      "YUYV 4:2:2"   },
        { cudaVideoCodec_UYVY,      "UYVY 4:2:2"   },
    };

    if (eCodec >= 0 && eCodec <= cudaVideoCodec_NumCodecs) {
        return aCodecName[eCodec].name;
    }
    for (int i = cudaVideoCodec_NumCodecs + 1; i < sizeof(aCodecName) / sizeof(aCodecName[0]); i++) {
        if (eCodec == aCodecName[i].eCodec) {
            return aCodecName[eCodec].name;
        }
    }
    return "Unknown";
}

static const char * GetVideoChromaFormatString(cudaVideoChromaFormat eChromaFormat) {
    static struct {
        cudaVideoChromaFormat eChromaFormat;
        const char *name;
    } aChromaFormatName[] = {
        { cudaVideoChromaFormat_Monochrome, "YUV 400 (Monochrome)" },
        { cudaVideoChromaFormat_420,        "YUV 420"              },
        { cudaVideoChromaFormat_422,        "YUV 422"              },
        { cudaVideoChromaFormat_444,        "YUV 444"              },
    };

    if (eChromaFormat >= 0 && eChromaFormat < sizeof(aChromaFormatName) / sizeof(aChromaFormatName[0])) {
        return aChromaFormatName[eChromaFormat].name;
    }
    return "Unknown";
}

static float GetChromaHeightFactor(cudaVideoSurfaceFormat eSurfaceFormat)
{
    float factor = 0.5;
    switch (eSurfaceFormat)
    {
    case cudaVideoSurfaceFormat_NV12:
    case cudaVideoSurfaceFormat_P016:
        factor = 0.5;
        break;
    case cudaVideoSurfaceFormat_YUV444:
    case cudaVideoSurfaceFormat_YUV444_16Bit:
        factor = 1.0;
        break;
    case cudaVideoSurfaceFormat_NV16:
    case cudaVideoSurfaceFormat_P216:
        factor = 1.0;
        break;
    }

    return factor;
}

static int GetChromaPlaneCount(cudaVideoSurfaceFormat eSurfaceFormat)
{
    int numPlane = 1;
    switch (eSurfaceFormat)
    {
    case cudaVideoSurfaceFormat_NV12:
    case cudaVideoSurfaceFormat_P016:
        numPlane = 1;
        break;
    case cudaVideoSurfaceFormat_YUV444:
    case cudaVideoSurfaceFormat_YUV444_16Bit:
        numPlane = 2;
        break;
    case cudaVideoSurfaceFormat_NV16:
    case cudaVideoSurfaceFormat_P216:
        numPlane = 1;
        break;
    }

    return numPlane;
}

std::map<int, int64_t> NvDecoder::sessionOverHead = { {0,0}, {1,0} };


/**
*   @brief  This function is used to get codec string from codec id
*/
const char *NvDecoder::GetCodecString(cudaVideoCodec eCodec)
{
    return GetVideoCodecString(eCodec);
}

/* Called when the parser encounters sequence header for AV1 SVC content
*  return value interpretation:
*      < 0 : fail, >=0: succeeded (bit 0-9: currOperatingPoint, bit 10-10: bDispAllLayer, bit 11-30: reserved, must be set 0)
*/
int NvDecoder::GetOperatingPoint(CUVIDOPERATINGPOINTINFO *pOPInfo)
{
    if (pOPInfo->codec == cudaVideoCodec_AV1)
    {
        if (pOPInfo->av1.operating_points_cnt > 1)
        {
            // clip has SVC enabled
            if (m_nOperatingPoint >= pOPInfo->av1.operating_points_cnt)
                m_nOperatingPoint = 0;

            // printf("AV1 SVC clip: operating point count %d  ", pOPInfo->av1.operating_points_cnt);
            // printf("Selected operating point: %d, IDC 0x%x bOutputAllLayers %d\n", m_nOperatingPoint, pOPInfo->av1.operating_points_idc[m_nOperatingPoint], m_bDispAllLayers);
            return (m_nOperatingPoint | (m_bDispAllLayers << 10));
        }
    }
    return -1;
}

int NvDecoder::HandleVideoSequencePerf(CUVIDEOFORMAT* pVideoFormat)
{


    auto sessionStart = std::chrono::high_resolution_clock::now();

    int nDecodeSurface = NvDecoder::HandleVideoSequence(pVideoFormat);


    std::unique_lock<std::mutex> lock(NvDecoderPerf::m_initMutex);

    NvDecoderPerf::IncrementSessionInitCounter();

    // Wait for all threads to finish initialization of the decoder session.
    // This ensures that all threads start decoding frames at the same
    // time and saturate the decoder engines. This also leads to more
    // accurate measurement of decoding performance.
    if (NvDecoderPerf::GetSessionInitCounter() == NvDecoderPerf::GetSessionCount())
    {
        NvDecoderPerf::m_cvInit.notify_all();
    }
    else
    {
        NvDecoderPerf::m_cvInit.wait(lock, [] { return NvDecoderPerf::GetSessionInitCounter() >= NvDecoderPerf::GetSessionCount(); });
    }

    auto sessionEnd = std::chrono::high_resolution_clock::now();
    int64_t elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(sessionEnd - sessionStart).count();

    m_nvdecSessionPerf.SetSessionInitTime(elapsedTime);


    return nDecodeSurface;
}

/* Return value from HandleVideoSequence() are interpreted as   :
*  0: fail, 1: succeeded, > 1: override dpb size of parser (set by CUVIDPARSERPARAMS::ulMaxNumDecodeSurfaces while creating parser)
*/
int NvDecoder::HandleVideoSequence(CUVIDEOFORMAT *pVideoFormat)
{

    NVTX_SCOPED_RANGE("seq")
    START_TIMER
    int64_t elapsedTime = 0;
    m_videoInfo.str("");
    m_videoInfo.clear();
    m_videoInfo << "Video Input Information" << std::endl
        << "\tCodec        : " << GetVideoCodecString(pVideoFormat->codec) << std::endl
        << "\tFrame rate   : " << pVideoFormat->frame_rate.numerator << "/" << pVideoFormat->frame_rate.denominator
            << " = " << 1.0 * pVideoFormat->frame_rate.numerator / pVideoFormat->frame_rate.denominator << " fps" << std::endl
        << "\tSequence     : " << (pVideoFormat->progressive_sequence ? "Progressive" : "Interlaced") << std::endl
        << "\tCoded size   : [" << pVideoFormat->coded_width << ", " << pVideoFormat->coded_height << "]" << std::endl
        << "\tDisplay area : [" << pVideoFormat->display_area.left << ", " << pVideoFormat->display_area.top << ", "
            << pVideoFormat->display_area.right << ", " << pVideoFormat->display_area.bottom << "]" << std::endl
        << "\tChroma       : " << GetVideoChromaFormatString(pVideoFormat->chroma_format) << std::endl
        << "\tBit depth    : " << pVideoFormat->bit_depth_luma_minus8 + 8
    ;
    m_videoInfo << std::endl;

    int nDecodeSurface = m_bLowLatency ? pVideoFormat->min_num_decode_surfaces : pVideoFormat->min_num_decode_surfaces + 4;

    if (!m_bDecodeCapsSet)
    {
        CUVIDDECODECAPS decodecaps;
        memset(&decodecaps, 0, sizeof(decodecaps));

        decodecaps.eCodecType = pVideoFormat->codec;
        decodecaps.eChromaFormat = pVideoFormat->chroma_format;
        decodecaps.nBitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;

        CUDA_DRVAPI_CALL(cuCtxPushCurrent(m_cuContext));
        NVDEC_API_CALL(m_api.cuvidGetDecoderCaps(&decodecaps));
        CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
        if (!decodecaps.bIsSupported) {
            PYNVVC_THROW_ERROR_UNSUPPORTED("Codec not supported on this GPU", CUDA_ERROR_NOT_SUPPORTED);
            return nDecodeSurface;
        }
        memcpy(&m_decodecaps, &decodecaps, sizeof(decodecaps));
        m_bDecodeCapsSet = true;
    }

    if ((pVideoFormat->coded_width>>4)*(pVideoFormat->coded_height>>4) > m_decodecaps.nMaxMBCount){

        std::ostringstream errorString;
        errorString << "MBCount not supported on this GPU" << std::endl
                    << "MBCount             : " << (pVideoFormat->coded_width >> 4)*(pVideoFormat->coded_height >> 4) << std::endl
                    << "Max Supported mbcnt : " << m_decodecaps.nMaxMBCount;

        const std::string cErr = errorString.str();
        PYNVVC_THROW_ERROR_UNSUPPORTED(cErr, CUDA_ERROR_NOT_SUPPORTED);
        return nDecodeSurface;
    }

    if (m_nWidth && m_nLumaHeight && m_nChromaHeight) {

        // cuvidCreateDecoder() has been called before, and now there's possible config change
        return ReconfigureDecoder(pVideoFormat);
    }

    // eCodec has been set in the constructor (for parser). Here it's set again for potential correction
    m_eCodec = pVideoFormat->codec;
    m_eChromaFormat = pVideoFormat->chroma_format;
    m_nBitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
    m_nBPP = m_nBitDepthMinus8 > 0 ? 2 : 1;

    // Set the output surface format same as chroma format
    if (m_eChromaFormat == cudaVideoChromaFormat_420 || cudaVideoChromaFormat_Monochrome)
        m_eOutputFormat = pVideoFormat->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
    else if (m_eChromaFormat == cudaVideoChromaFormat_444)
        m_eOutputFormat = pVideoFormat->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_YUV444_16Bit : cudaVideoSurfaceFormat_YUV444;
    else if (m_eChromaFormat == cudaVideoChromaFormat_422)
        m_eOutputFormat = pVideoFormat->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_P216 : cudaVideoSurfaceFormat_NV16;

    // Check if output format supported. If not, check falback options
    if (!(m_decodecaps.nOutputFormatMask & (1 << m_eOutputFormat)))
    {
        if (m_decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_NV12))
            m_eOutputFormat = cudaVideoSurfaceFormat_NV12;
        else if (m_decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_P016))
            m_eOutputFormat = cudaVideoSurfaceFormat_P016;
        else if (m_decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444))
            m_eOutputFormat = cudaVideoSurfaceFormat_YUV444;
        else if (m_decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444_16Bit))
            m_eOutputFormat = cudaVideoSurfaceFormat_YUV444_16Bit;
        else if (m_decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_NV16))
            m_eOutputFormat = cudaVideoSurfaceFormat_NV16;
        else if (m_decodecaps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_P216))
            m_eOutputFormat = cudaVideoSurfaceFormat_P216;
        else 
            PYNVVC_THROW_ERROR_UNSUPPORTED("No supported output format found", CUDA_ERROR_NOT_SUPPORTED);
    }
    m_videoFormat = *pVideoFormat;

    CUVIDDECODECREATEINFO videoDecodeCreateInfo = { 0 };
    videoDecodeCreateInfo.CodecType = pVideoFormat->codec;
    videoDecodeCreateInfo.ChromaFormat = pVideoFormat->chroma_format;
    videoDecodeCreateInfo.OutputFormat = m_eOutputFormat;
    videoDecodeCreateInfo.bitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
    if (pVideoFormat->progressive_sequence)
        videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    else
        videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
    videoDecodeCreateInfo.ulNumOutputSurfaces = 2;
    // With PreferCUVID, JPEG is still decoded by CUDA while video is decoded by NVDEC hardware
    videoDecodeCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    videoDecodeCreateInfo.ulNumDecodeSurfaces = nDecodeSurface;
    videoDecodeCreateInfo.ulWidth = pVideoFormat->coded_width;
    videoDecodeCreateInfo.ulHeight = pVideoFormat->coded_height;
    // AV1 has max width/height of sequence in sequence header
    if (pVideoFormat->codec == cudaVideoCodec_AV1 && pVideoFormat->seqhdr_data_length > 0)
    {
        // dont overwrite if it is already set from cmdline or reconfig.txt
        if (!(m_nMaxWidth > pVideoFormat->coded_width || m_nMaxHeight > pVideoFormat->coded_height))
        {
            CUVIDEOFORMATEX *vidFormatEx = (CUVIDEOFORMATEX *)pVideoFormat;
            m_nMaxWidth = vidFormatEx->av1.max_width;
            m_nMaxHeight = vidFormatEx->av1.max_height;
        }
    }
    if (m_nMaxWidth < (int)pVideoFormat->coded_width)
        m_nMaxWidth = pVideoFormat->coded_width;
    if (m_nMaxHeight < (int)pVideoFormat->coded_height)
        m_nMaxHeight = pVideoFormat->coded_height;
    videoDecodeCreateInfo.ulMaxWidth = m_nMaxWidth;
    videoDecodeCreateInfo.ulMaxHeight = m_nMaxHeight;

    if ((m_nMaxWidth  > m_decodecaps.nMaxWidth) ||
        (m_nMaxHeight > m_decodecaps.nMaxHeight)){

        std::ostringstream errorString;
        errorString << "Resolution not supported on this GPU" << std::endl
                    << "Resolution          : " << m_nMaxWidth << "x" << m_nMaxHeight << std::endl
                    << "Max Supported (wxh) : " << m_decodecaps.nMaxWidth << "x" << m_decodecaps.nMaxHeight;

        const std::string cErr = errorString.str();
        PYNVVC_THROW_ERROR_UNSUPPORTED(cErr, CUDA_ERROR_NOT_SUPPORTED);
        return nDecodeSurface;
    }

    if (!(m_resizeDim.w && m_resizeDim.h)) {
        m_nWidth = pVideoFormat->display_area.right - pVideoFormat->display_area.left;
        m_nLumaHeight = pVideoFormat->display_area.bottom - pVideoFormat->display_area.top;
        videoDecodeCreateInfo.ulTargetWidth = pVideoFormat->coded_width;
        videoDecodeCreateInfo.ulTargetHeight = pVideoFormat->coded_height;
    } else {
        if (m_resizeDim.w && m_resizeDim.h) {
            videoDecodeCreateInfo.display_area.left = pVideoFormat->display_area.left;
            videoDecodeCreateInfo.display_area.top = pVideoFormat->display_area.top;
            videoDecodeCreateInfo.display_area.right = pVideoFormat->display_area.right;
            videoDecodeCreateInfo.display_area.bottom = pVideoFormat->display_area.bottom;
            m_nWidth = m_resizeDim.w;
            m_nLumaHeight = m_resizeDim.h;
        }

        videoDecodeCreateInfo.ulTargetWidth = m_nWidth;
        videoDecodeCreateInfo.ulTargetHeight = m_nLumaHeight;
    }

    m_nChromaHeight = (int)(ceil(m_nLumaHeight * GetChromaHeightFactor(m_eOutputFormat)));
    m_nNumChromaPlanes = GetChromaPlaneCount(m_eOutputFormat);
    m_nSurfaceHeight = videoDecodeCreateInfo.ulTargetHeight;
    m_nSurfaceWidth = videoDecodeCreateInfo.ulTargetWidth;
    m_displayRect.b = videoDecodeCreateInfo.display_area.bottom;
    m_displayRect.t = videoDecodeCreateInfo.display_area.top;
    m_displayRect.l = videoDecodeCreateInfo.display_area.left;
    m_displayRect.r = videoDecodeCreateInfo.display_area.right;

    m_videoInfo << "Video Decoding Params:" << std::endl
        << "\tNum Surfaces : " << videoDecodeCreateInfo.ulNumDecodeSurfaces << std::endl
        << "\tCrop         : [" << videoDecodeCreateInfo.display_area.left << ", " << videoDecodeCreateInfo.display_area.top << ", "
        << videoDecodeCreateInfo.display_area.right << ", " << videoDecodeCreateInfo.display_area.bottom << "]" << std::endl
        << "\tResize       : " << videoDecodeCreateInfo.ulTargetWidth << "x" << videoDecodeCreateInfo.ulTargetHeight << std::endl
        << "\tDeinterlace  : " << std::vector<const char *>{"Weave", "Bob", "Adaptive"}[videoDecodeCreateInfo.DeinterlaceMode]
    ;
    m_videoInfo << std::endl;

    CUDA_DRVAPI_CALL(cuCtxPushCurrent(m_cuContext));



    NVDEC_API_CALL(m_api.cuvidCreateDecoder(&m_hDecoder, &videoDecodeCreateInfo));

    // + 34mb - those are cleaned on kill


    uint8_t* pFrame[8] = { NULL };
    if (m_bUseDeviceFrame)
    {
        if (m_bEnableAsyncAllocations)
        {
            for (size_t i = 0; i < 8; i++)
            {
                CUDA_DRVAPI_CALL(cuMemAllocAsync((CUdeviceptr*)&pFrame[i], GetFrameSize(), m_cuvidStream));
            }
        }
    }
    else
    {
        CUDA_DRVAPI_CALL(cuMemAlloc((CUdeviceptr *)&m_dpScratchFrame, GetOutputFrameSize()));
    }


    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));

    STOP_TIMER("Session Initialization Time: ");


    NvDecoder::addDecoderSessionOverHead(getDecoderSessionID(), elapsedTime);


    // +32MB
    
    return nDecodeSurface;
}

int NvDecoder::ReconfigureDecoder(CUVIDEOFORMAT *pVideoFormat)
{
    NVTX_SCOPED_RANGE("recon")
    int64_t elapsedTime = 0;
    if (pVideoFormat->bit_depth_luma_minus8 != m_videoFormat.bit_depth_luma_minus8 || pVideoFormat->bit_depth_chroma_minus8 != m_videoFormat.bit_depth_chroma_minus8){

        PYNVVC_THROW_ERROR("Reconfigure Not supported for bit depth change", CUDA_ERROR_NOT_SUPPORTED);
    }

    if (pVideoFormat->chroma_format != m_videoFormat.chroma_format) {

        PYNVVC_THROW_ERROR("Reconfigure Not supported for chroma format change", CUDA_ERROR_NOT_SUPPORTED);
    }

    bool bDecodeResChange = !(pVideoFormat->coded_width == m_videoFormat.coded_width && pVideoFormat->coded_height == m_videoFormat.coded_height);
    bool bDisplayRectChange = !(pVideoFormat->display_area.bottom == m_videoFormat.display_area.bottom && pVideoFormat->display_area.top == m_videoFormat.display_area.top \
        && pVideoFormat->display_area.left == m_videoFormat.display_area.left && pVideoFormat->display_area.right == m_videoFormat.display_area.right);

    int nDecodeSurface = m_bLowLatency ? pVideoFormat->min_num_decode_surfaces : pVideoFormat->min_num_decode_surfaces + 4;

    if ((pVideoFormat->coded_width > m_nMaxWidth) || (pVideoFormat->coded_height > m_nMaxHeight)) {
        // For VP9, let driver  handle the change if new width/height > maxwidth/maxheight
        if ((m_eCodec != cudaVideoCodec_VP9) || m_bReconfigExternal)
        {
            PYNVVC_THROW_ERROR("Reconfigure Not supported when width/height > maxwidth/maxheight", CUDA_ERROR_NOT_SUPPORTED);
        }
        return 1;
    }

    if (!bDecodeResChange) {
        // if the coded_width/coded_height hasn't changed but display resolution has changed, then need to update width/height for
        // correct output without cropping. Example : 1920x1080 vs 1920x1088
        if (bDisplayRectChange)
        {
            m_nWidth = pVideoFormat->display_area.right - pVideoFormat->display_area.left;
            m_nLumaHeight = pVideoFormat->display_area.bottom - pVideoFormat->display_area.top;
            m_nChromaHeight = (int)ceil(m_nLumaHeight * GetChromaHeightFactor(m_eOutputFormat));
            m_nNumChromaPlanes = GetChromaPlaneCount(m_eOutputFormat);
            
            // update display rect
            m_videoFormat.display_area.bottom = pVideoFormat->display_area.bottom;
            m_videoFormat.display_area.top    = pVideoFormat->display_area.top;
            m_videoFormat.display_area.left   = pVideoFormat->display_area.left;
            m_videoFormat.display_area.right  = pVideoFormat->display_area.right;
        }

        // no need for reconfigureDecoder(). Just return
        return 1;
    }

    CUVIDRECONFIGUREDECODERINFO reconfigParams = { 0 };

    reconfigParams.ulWidth = m_videoFormat.coded_width = pVideoFormat->coded_width;
    reconfigParams.ulHeight = m_videoFormat.coded_height = pVideoFormat->coded_height;

    // Dont change display rect and get scaled output from decoder. This will help display app to present apps smoothly
    reconfigParams.display_area.bottom = m_displayRect.b;
    reconfigParams.display_area.top = m_displayRect.t;
    reconfigParams.display_area.left = m_displayRect.l;
    reconfigParams.display_area.right = m_displayRect.r;
    reconfigParams.ulTargetWidth = m_nSurfaceWidth;
    reconfigParams.ulTargetHeight = m_nSurfaceHeight;
    

    // If external reconfigure is called along with resolution change even if post processing params is not changed,
    // do full reconfigure params update
    if ((m_bReconfigExternal && bDecodeResChange)) {
        // update display rect and target resolution if requested explicitly
        m_bReconfigExternal = false;
        m_videoFormat = *pVideoFormat;
        if (!(m_resizeDim.w && m_resizeDim.h)) {
            m_nWidth = pVideoFormat->display_area.right - pVideoFormat->display_area.left;
            m_nLumaHeight = pVideoFormat->display_area.bottom - pVideoFormat->display_area.top;
            reconfigParams.ulTargetWidth = pVideoFormat->coded_width;
            reconfigParams.ulTargetHeight = pVideoFormat->coded_height;
        }
        else {
            if (m_resizeDim.w && m_resizeDim.h) {
                reconfigParams.display_area.left = pVideoFormat->display_area.left;
                reconfigParams.display_area.top = pVideoFormat->display_area.top;
                reconfigParams.display_area.right = pVideoFormat->display_area.right;
                reconfigParams.display_area.bottom = pVideoFormat->display_area.bottom;
                m_nWidth = m_resizeDim.w;
                m_nLumaHeight = m_resizeDim.h;
            }
            reconfigParams.ulTargetWidth = m_nWidth;
            reconfigParams.ulTargetHeight = m_nLumaHeight;
        }

        m_nChromaHeight = (int)ceil(m_nLumaHeight * GetChromaHeightFactor(m_eOutputFormat));
        m_nNumChromaPlanes = GetChromaPlaneCount(m_eOutputFormat);
        m_nSurfaceHeight = reconfigParams.ulTargetHeight;
        m_nSurfaceWidth = reconfigParams.ulTargetWidth;
        m_displayRect.b = reconfigParams.display_area.bottom;
        m_displayRect.t = reconfigParams.display_area.top;
        m_displayRect.l = reconfigParams.display_area.left;
        m_displayRect.r = reconfigParams.display_area.right;
    }

    reconfigParams.ulNumDecodeSurfaces = nDecodeSurface;

    // reconfigure decoder
    START_TIMER
    CUDA_DRVAPI_CALL(cuCtxPushCurrent(m_cuContext));
    NVDEC_API_CALL(m_api.cuvidReconfigureDecoder(m_hDecoder, &reconfigParams));
    
    //deallocate earlier buffers
    for (uint8_t* pFrame : m_vpFrame)
    {
        if (m_bUseDeviceFrame)
        {
            if (m_bEnableAsyncAllocations)
            {
                CUDA_DRVAPI_CALL(cuMemFreeAsync((*(CUdeviceptr*)&pFrame), NULL));//sync on NULL stream to ensure that all work is completed before dtor
            }
        }
    }
    //recreate new buffers
    uint8_t* pFrame[8] = { NULL };
    if (m_bUseDeviceFrame)
        {
            if (m_bEnableAsyncAllocations)
            {
                for (size_t i = 0; i < 8; i++)
                {
                    CUDA_DRVAPI_CALL(cuMemAllocAsync((CUdeviceptr*)&pFrame[i], GetFrameSize(), m_cuvidStream));
                }
            }
        }
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
    STOP_TIMER("Session Reconfigure Time: ");

    return nDecodeSurface;
}


std::vector<std::tuple<CUdeviceptr, int64_t, SEI_MESSAGE, CUevent>> NvDecoder::PyDecode(uint8_t* bsl_data, uint64_t bsl, int32_t pts, int32_t decode_flag)
{
    
    int  numFrames = this->Decode(bsl_data,bsl, decode_flag,pts);


    std::vector<std::tuple<CUdeviceptr, int64_t, SEI_MESSAGE, CUevent>> frames;
    for (int i = 0; i < numFrames; i++)
    {
        int64_t timestamp = 0;
        SEI_MESSAGE seiMessage;
        CUevent event = nullptr;
        CUdeviceptr  data = (CUdeviceptr)this->GetFrame(&timestamp, &seiMessage, &event);
        auto outputFormat = this->GetOutputFormat();
        std::tuple<CUdeviceptr, int64_t, SEI_MESSAGE, CUevent> frame(data, timestamp, seiMessage, event);
         
        switch (outputFormat)
        {
        case cudaVideoSurfaceFormat_P016:
        case cudaVideoSurfaceFormat_YUV444:
        case cudaVideoSurfaceFormat_YUV444_16Bit:
        case cudaVideoSurfaceFormat_NV12:
        case cudaVideoSurfaceFormat_NV16:
        case cudaVideoSurfaceFormat_P216:
        {
            break;
        }
        default: throw std::runtime_error("TODO: not implemented buffer format");
        }
        frames.push_back(frame);
    }

    return frames;
}


int NvDecoder::setReconfigParams( const Dim& mResizeDim)
{
    
    setSeekPTS(0);
    if ((mResizeDim.w == 0 && mResizeDim.h  == 0))
    {
        return 0;
    }
    else if ((mResizeDim.w > m_nMaxWidth) || (mResizeDim.h > m_nMaxHeight))
    {
        throw std::runtime_error("Resize dimensions must be lower than max width and height, please recreate decoder instance");
    }
    else
    {
        m_bReconfigExternal = true;
        if ((mResizeDim.w != m_resizeDim.w) || (mResizeDim.h != m_resizeDim.h))
        {
            // Clear existing output buffers of different size
            uint8_t* pFrame = NULL;
            while (!m_vpFrame.empty())
            {
                pFrame = m_vpFrame.back();
                m_vpFrame.pop_back();
                if (m_bUseDeviceFrame)
                {
                    CUDA_DRVAPI_CALL(cuCtxPushCurrent(m_cuContext));
                    CUDA_DRVAPI_CALL(cuMemFree((CUdeviceptr)pFrame));
                    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
                }
                else
                {
                    delete pFrame;
                }
            }
        }
        m_resizeDim.w = mResizeDim.w;
        m_resizeDim.h = mResizeDim.h;

    }

    return 1;
}

void NvDecoder::GenerateNativeOutput(CUdeviceptr dpSrcFrame, unsigned int nSrcPitch, uint8_t* pDecodedFrame)
{
    // Copy luma plane
    CUDA_MEMCPY2D m = { 0 };
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = dpSrcFrame;
    m.srcPitch = nSrcPitch;
    m.dstMemoryType = m_bUseDeviceFrame ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
    m.dstDevice = (CUdeviceptr)(m.dstHost = pDecodedFrame);
    m.dstPitch = m_nDeviceFramePitch ? m_nDeviceFramePitch : GetWidth() * m_nBPP;
    m.WidthInBytes = GetWidth() * m_nBPP;
    m.Height = m_nLumaHeight;
    CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&m, m_cuvidStream));

    // Copy chroma plane
    // NVDEC output has luma height aligned by 2. Adjust chroma offset by aligning height
    m.srcDevice = (CUdeviceptr)((uint8_t*)dpSrcFrame + m.srcPitch * ((m_nSurfaceHeight + 1) & ~1));
    m.dstDevice = (CUdeviceptr)(m.dstHost = pDecodedFrame + m.dstPitch * m_nLumaHeight);
    m.Height = m_nChromaHeight;
    CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&m, m_cuvidStream));

    if (m_nNumChromaPlanes == 2)
    {
        m.srcDevice = (CUdeviceptr)((uint8_t*)dpSrcFrame + m.srcPitch * ((m_nSurfaceHeight + 1) & ~1) * 2);
        m.dstDevice = (CUdeviceptr)(m.dstHost = pDecodedFrame + m.dstPitch * m_nLumaHeight * 2);
        m.Height = m_nChromaHeight;
        CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&m, m_cuvidStream));
    }
}

void NvDecoder::GenerateRGBOutput(CUdeviceptr dpSrcFrame, unsigned int nSrcPitch, uint8_t* pDecodedFrame)
{   
    constexpr uint32_t perPixelComponents = 3;
    auto matrixCoefficients = GetVideoFormatInfo().video_signal_description.matrix_coefficients;
    auto outputFormat = GetOutputFormat();
    switch (outputFormat)
    {
        case cudaVideoSurfaceFormat_NV12:
            Nv12ToColor24<RGB24>((uint8_t*)dpSrcFrame, nSrcPitch, pDecodedFrame, 
                perPixelComponents * GetWidth(), GetWidth(), m_nSurfaceHeight, matrixCoefficients, m_cuvidStream);
            break;
        case cudaVideoSurfaceFormat_P016:
            P016ToColor24<RGB24>((uint8_t*)dpSrcFrame, nSrcPitch, pDecodedFrame, 
                perPixelComponents * GetWidth(), GetWidth(), m_nSurfaceHeight, matrixCoefficients, m_cuvidStream);
            break;
        case cudaVideoSurfaceFormat_YUV444:
            YUV444ToColor24<RGB24>((uint8_t*)dpSrcFrame, nSrcPitch, pDecodedFrame, 
                perPixelComponents * GetWidth(), GetWidth(), m_nSurfaceHeight, matrixCoefficients, m_cuvidStream);
            break;
        case cudaVideoSurfaceFormat_YUV444_16Bit:
            YUV444P16ToColor24<RGB24>((uint8_t*)dpSrcFrame, nSrcPitch, pDecodedFrame, 
                perPixelComponents * GetWidth(), GetWidth(), m_nSurfaceHeight, matrixCoefficients, m_cuvidStream);
            break;
        case cudaVideoSurfaceFormat_NV16:
            Nv16ToColor24<RGB24>((uint8_t*)dpSrcFrame, nSrcPitch, pDecodedFrame, 
                perPixelComponents * GetWidth(), GetWidth(), m_nSurfaceHeight, GetHeight(), matrixCoefficients, m_cuvidStream);
            break;
        case cudaVideoSurfaceFormat_P216:
            P216ToColor24<RGB24>((uint8_t*)dpSrcFrame, nSrcPitch, pDecodedFrame, 
                perPixelComponents * GetWidth(), GetWidth(), m_nSurfaceHeight, GetHeight(), matrixCoefficients, m_cuvidStream);
            break;
        default:
            break;
    }
}

void NvDecoder::GenerateRGBPOutput(CUdeviceptr dpSrcFrame, unsigned int nSrcPitch, uint8_t* pDecodedFrame)
{
    auto matrixCoefficients = GetVideoFormatInfo().video_signal_description.matrix_coefficients;
    auto outputFormat = GetOutputFormat();
    switch (outputFormat)
    {
        case cudaVideoSurfaceFormat_NV12:
            Nv12ToColor24Planar<RGB24>((uint8_t*)dpSrcFrame, nSrcPitch, pDecodedFrame,
                GetWidth(), GetWidth(), m_nSurfaceHeight, GetHeight(), matrixCoefficients, m_cuvidStream);
            break;
        case cudaVideoSurfaceFormat_P016:
            P016ToColor24Planar<RGB24>((uint8_t*)dpSrcFrame, nSrcPitch, pDecodedFrame,
                GetWidth(), GetWidth(), m_nSurfaceHeight, GetHeight(), matrixCoefficients, m_cuvidStream);
            break;
        case cudaVideoSurfaceFormat_YUV444:
            YUV444ToColor24Planar<RGB24>((uint8_t*)dpSrcFrame, nSrcPitch, pDecodedFrame,
                GetWidth(), GetWidth(), m_nSurfaceHeight, GetHeight(), matrixCoefficients, m_cuvidStream);
            break;
        case cudaVideoSurfaceFormat_YUV444_16Bit:
            YUV444P16ToColor24Planar<RGB24>((uint8_t*)dpSrcFrame, nSrcPitch, pDecodedFrame,
                GetWidth(), GetWidth(), m_nSurfaceHeight, GetHeight(), matrixCoefficients, m_cuvidStream);
            break;
        case cudaVideoSurfaceFormat_NV16:
            Nv16ToColor24Planar<RGB24>((uint8_t*)dpSrcFrame, nSrcPitch, pDecodedFrame,
                GetWidth(), GetWidth(), m_nSurfaceHeight, GetHeight(), matrixCoefficients, m_cuvidStream);
            break;
        case cudaVideoSurfaceFormat_P216:
            P216ToColor24Planar<RGB24>((uint8_t*)dpSrcFrame, nSrcPitch, pDecodedFrame,
                GetWidth(), GetWidth(), m_nSurfaceHeight, GetHeight(), matrixCoefficients, m_cuvidStream);
            break;
        default:
            break;
    }

}

void NvDecoder::GenerateOutput(CUdeviceptr dpSrcFrame, unsigned int nSrcPitch, uint8_t* pDecodedFrame)
{
    switch (m_eUserOutputColorType)
    {
        case OutputColorType::NATIVE:
            GenerateNativeOutput(dpSrcFrame, nSrcPitch, pDecodedFrame);
            return;
        case OutputColorType::RGB:
        {
            auto deviceFrame = m_bUseDeviceFrame ? pDecodedFrame : (uint8_t*)m_dpScratchFrame;
            GenerateRGBOutput(dpSrcFrame, nSrcPitch, deviceFrame);
            break;
        }
        case OutputColorType::RGBP:
        {
            auto deviceFrame = m_bUseDeviceFrame ? pDecodedFrame : (uint8_t*)m_dpScratchFrame;
            GenerateRGBPOutput(dpSrcFrame, nSrcPitch, deviceFrame);
            break;
        }
        default:
            // Not expected
            break;
    }
    if (!m_bUseDeviceFrame)
    {
        // copy the output to host
        cuMemcpyDtoH(pDecodedFrame, m_dpScratchFrame, GetOutputFrameSize());
    }
}

/* Return value from HandlePictureDecode() are interpreted as:
*  0: fail, >=1: succeeded
*/
int NvDecoder::HandlePictureDecode(CUVIDPICPARAMS *pPicParams) {
    
    NVTX_SCOPED_RANGE("decode")
    if (!m_hDecoder)
    {
        PYNVVC_THROW_ERROR("Decoder not initialized.", CUDA_ERROR_NOT_INITIALIZED);
        return false;
    }
    m_nPicNumInDecodeOrder[pPicParams->CurrPicIdx] = m_nDecodePicCnt++;
    CUDA_DRVAPI_CALL(cuCtxPushCurrent(m_cuContext));
    NVDEC_API_CALL(m_api.cuvidDecodePicture(m_hDecoder, pPicParams));
    if (m_bForce_zero_latency && ((!pPicParams->field_pic_flag) || (pPicParams->second_field)))
    {
        CUVIDPARSERDISPINFO dispInfo;
        memset(&dispInfo, 0, sizeof(dispInfo));
        dispInfo.picture_index = pPicParams->CurrPicIdx;
        dispInfo.progressive_frame = !pPicParams->field_pic_flag;
        dispInfo.top_field_first = pPicParams->bottom_field_flag ^ 1;
        HandlePictureDisplay(&dispInfo);
    }
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
    return 1;
}

/* Return value from HandlePictureDisplay() are interpreted as:
*  0: fail, >=1: succeeded
*/
int NvDecoder::HandlePictureDisplay(CUVIDPARSERDISPINFO *pDispInfo) {

    CUVIDPROCPARAMS videoProcessingParameters = {};
    videoProcessingParameters.progressive_frame = pDispInfo->progressive_frame;
    videoProcessingParameters.second_field = pDispInfo->repeat_first_field + 1;
    videoProcessingParameters.top_field_first = pDispInfo->top_field_first;
    videoProcessingParameters.unpaired_field = pDispInfo->repeat_first_field < 0;
    videoProcessingParameters.output_stream = m_cuvidStream;

    CUdeviceptr dpSrcFrame = 0;
    unsigned int nSrcPitch = 0;
    CUDA_DRVAPI_CALL(cuCtxPushCurrent(m_cuContext));
    NVTX_SCOPED_RANGE("display")
    if (m_nSeekPts == 0 || pDispInfo->timestamp >= m_nSeekPts)
    {

        NVDEC_API_CALL(m_api.cuvidMapVideoFrame(m_hDecoder, pDispInfo->picture_index, &dpSrcFrame,
            &nSrcPitch, &videoProcessingParameters));
        CUVIDGETDECODESTATUS DecodeStatus;
        memset(&DecodeStatus, 0, sizeof(DecodeStatus));
        CUresult result = m_api.cuvidGetDecodeStatus(m_hDecoder, pDispInfo->picture_index, &DecodeStatus);
        if (result == CUDA_SUCCESS && (DecodeStatus.decodeStatus == cuvidDecodeStatus_Error || DecodeStatus.decodeStatus == cuvidDecodeStatus_Error_Concealed))
        {
            printf("Decode Error occurred for picture %d\n", m_nPicNumInDecodeOrder[pDispInfo->picture_index]);
        }
    }

    uint8_t *pDecodedFrame = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_mtxVPFrame);
        if ((unsigned)++m_nDecodedFrame > m_vpFrame.size())
        {
            // Not enough frames in stock
            m_nFrameAlloc++;
            uint8_t *pFrame = NULL;
            if (m_bUseDeviceFrame)
            {
                if (m_bDeviceFramePitched)
                {
                    CUDA_DRVAPI_CALL(cuMemAllocPitch((CUdeviceptr *)&pFrame, &m_nDeviceFramePitch, GetWidth() * m_nBPP, m_nLumaHeight + (m_nChromaHeight * m_nNumChromaPlanes), 16));
                }
                else if (m_bEnableAsyncAllocations)
                {
                    CUDA_DRVAPI_CALL(cuMemAllocAsync((CUdeviceptr*)&pFrame, GetOutputFrameSize(), m_cuvidStream));
                }
                else
                {
                    CUDA_DRVAPI_CALL(cuMemAlloc((CUdeviceptr *)&pFrame, GetOutputFrameSize()));
                }
            }
            else
            {
                pFrame = new uint8_t[GetOutputFrameSize()];
            }
            m_vpFrame.push_back(pFrame);
            
            CUevent event;
            CUDA_DRVAPI_CALL(cuEventCreate(&event, 0));
            m_DecodedFrameEvent.push_back(event);
        }
        pDecodedFrame = m_vpFrame[m_nDecodedFrame - 1];
    }
    
    if (m_nSeekPts == 0 || pDispInfo->timestamp >= m_nSeekPts)
    {
        GenerateOutput(dpSrcFrame, nSrcPitch, pDecodedFrame);
        // Record event for this frame on  cuvid stream. This is essential for application sync
        CUDA_DRVAPI_CALL(cuEventRecord(m_DecodedFrameEvent[m_nDecodedFrame - 1], m_cuvidStream));
        if (m_bUseDeviceFrame)
        {
            if (m_bEnableAsyncAllocations)
            {
                CUDA_DRVAPI_CALL(cuEventRecord(m_bCUEvent, m_cuvidStream));
            }
            // we shouldn't need stream sync on null stream here. Keep it as a precaution.
            if (!m_cuvidStream)
            {
                CUDA_DRVAPI_CALL(cuStreamSynchronize(m_cuvidStream));
            }
        }

        CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
    }
    

    if ((int)m_vTimestamp.size() < m_nDecodedFrame) {
        m_vTimestamp.resize(m_vpFrame.size());
    }
    m_vTimestamp[m_nDecodedFrame - 1] = pDispInfo->timestamp;

    if (m_bExtractSEIMessage)
    {
        for (int field = 0; field < 2; field++)
        {
            if (m_SEIMessagesDisplayOrder[pDispInfo->picture_index][field].pSEIData)
            {
                // Write SEI Message
                uint8_t* seiBuffer = (uint8_t*)(m_SEIMessagesDisplayOrder[pDispInfo->picture_index][field].pSEIData);
                uint32_t seiNumMessages = m_SEIMessagesDisplayOrder[pDispInfo->picture_index][field].sei_message_count;
                CUSEIMESSAGE* seiMessagesInfo = m_SEIMessagesDisplayOrder[pDispInfo->picture_index][field].pSEIMessage;

                if ((int)m_vSEIMessage.size() < m_nDecodedFrame)
                {
                    m_vSEIMessage.resize(m_vpFrame.size());
                }
                m_vSEIMessage[m_nDecodedFrame - 1].resize(seiNumMessages);

                if (m_fpSEI)
                {
                    for (uint32_t i = 0; i < seiNumMessages; i++)
                    {
                        bool bIsUncompressed = false;
                        if ((m_eCodec == cudaVideoCodec_H264) ||
                            (m_eCodec == cudaVideoCodec_H264_SVC) ||
                            (m_eCodec == cudaVideoCodec_H264_MVC) ||
                            (m_eCodec == cudaVideoCodec_HEVC) ||
                            (m_eCodec == cudaVideoCodec_MPEG2))
                        {
                            switch (seiMessagesInfo[i].sei_message_type)
                            {
                            case SEI_TYPE_TIME_CODE:
                            case SEI_TYPE_TIME_CODE_H264:
                            {
                                bIsUncompressed = true;
                                if (m_eCodec != cudaVideoCodec_MPEG2)
                                {
                                    TIMECODE* timecode = (TIMECODE*)seiBuffer;
                                    fwrite(timecode, sizeof(TIMECODE), 1, m_fpSEI);
                                }
                                else
                                {
                                    TIMECODEMPEG2* timecode = (TIMECODEMPEG2*)seiBuffer;
                                    fwrite(timecode, sizeof(TIMECODEMPEG2), 1, m_fpSEI);
                                }
                            }
                            break;
                            case SEI_TYPE_USER_DATA_REGISTERED:
                            case SEI_TYPE_USER_DATA_UNREGISTERED:
                            {
                                fwrite(seiBuffer, seiMessagesInfo[i].sei_message_size, 1, m_fpSEI);
                            }
                            break;
                            case SEI_TYPE_MASTERING_DISPLAY_COLOR_VOLUME:
                            {
                                bIsUncompressed = true;
                                SEIMASTERINGDISPLAYINFO* masteringDisplayVolume = (SEIMASTERINGDISPLAYINFO*)seiBuffer;
                                fwrite(masteringDisplayVolume, sizeof(SEIMASTERINGDISPLAYINFO), 1, m_fpSEI);
                            }
                            break;
                            case SEI_TYPE_CONTENT_LIGHT_LEVEL_INFO:
                            {
                                bIsUncompressed = true;
                                SEICONTENTLIGHTLEVELINFO* contentLightLevelInfo = (SEICONTENTLIGHTLEVELINFO*)seiBuffer;
                                fwrite(contentLightLevelInfo, sizeof(SEICONTENTLIGHTLEVELINFO), 1, m_fpSEI);
                            }
                            break;
                            case SEI_TYPE_ALTERNATIVE_TRANSFER_CHARACTERISTICS:
                            {
                                bIsUncompressed = true;
                                SEIALTERNATIVETRANSFERCHARACTERISTICS* transferCharacteristics = (SEIALTERNATIVETRANSFERCHARACTERISTICS*)seiBuffer;
                                fwrite(transferCharacteristics, sizeof(SEIALTERNATIVETRANSFERCHARACTERISTICS), 1, m_fpSEI);
                            }
                            break;
                            }
                        }
                        if (m_eCodec == cudaVideoCodec_AV1)
                        {
                            fwrite(seiBuffer, seiMessagesInfo[i].sei_message_size, 1, m_fpSEI);
                        }
                        // Fill SEI_MESSAGE to send it to python
                        m_vSEIMessage[m_nDecodedFrame - 1][i].first.insert({"sei_type", seiMessagesInfo[i].sei_message_type});
                        m_vSEIMessage[m_nDecodedFrame - 1][i].first.insert({"sei_uncompressed", bIsUncompressed});
                        m_vSEIMessage[m_nDecodedFrame - 1][i].second.resize(seiMessagesInfo[i].sei_message_size);
                        m_vSEIMessage[m_nDecodedFrame - 1][i].second.assign(seiBuffer, seiBuffer + seiMessagesInfo[i].sei_message_size);
                        seiBuffer += seiMessagesInfo[i].sei_message_size;
                    }
                }
                free(m_SEIMessagesDisplayOrder[pDispInfo->picture_index][field].pSEIData);
                free(m_SEIMessagesDisplayOrder[pDispInfo->picture_index][field].pSEIMessage);
                m_SEIMessagesDisplayOrder[pDispInfo->picture_index][field].pSEIData = NULL;
                m_SEIMessagesDisplayOrder[pDispInfo->picture_index][field].pSEIMessage = NULL;
            }
        }
    }

    if (m_nSeekPts == 0 || pDispInfo->timestamp >= m_nSeekPts)
    {
        NVDEC_API_CALL(m_api.cuvidUnmapVideoFrame(m_hDecoder, dpSrcFrame));
    }

    return 1;
}

int NvDecoder::GetSEIMessage(CUVIDSEIMESSAGEINFO *pSEIMessageInfo)
{
    uint32_t seiNumMessages = pSEIMessageInfo->sei_message_count;
    CUSEIMESSAGE *seiMessagesInfo = pSEIMessageInfo->pSEIMessage;
    size_t totalSEIBufferSize = 0;
    if ((pSEIMessageInfo->picIdx < 0) || (pSEIMessageInfo->picIdx >= MAX_FRM_CNT))
    {
        printf("Invalid picture index (%d)\n", pSEIMessageInfo->picIdx);
        return 0;
    }
    for (uint32_t i = 0; i < seiNumMessages; i++)
    {
        totalSEIBufferSize += seiMessagesInfo[i].sei_message_size;
    }
    if (!m_pCurrSEIMessage)
    {
        printf("Out of Memory, Allocation failed for m_pCurrSEIMessage\n");
        return 0;
    }
    m_pCurrSEIMessage->pSEIData = malloc(totalSEIBufferSize);
    if (!m_pCurrSEIMessage->pSEIData)
    {
        printf("Out of Memory, Allocation failed for SEI Buffer\n");
        return 0;
    }
    memcpy(m_pCurrSEIMessage->pSEIData, pSEIMessageInfo->pSEIData, totalSEIBufferSize);
    m_pCurrSEIMessage->pSEIMessage = (CUSEIMESSAGE *)malloc(sizeof(CUSEIMESSAGE) * seiNumMessages);
    if (!m_pCurrSEIMessage->pSEIMessage)
    {
        free(m_pCurrSEIMessage->pSEIData);
        m_pCurrSEIMessage->pSEIData = NULL;
        return 0;
    }
    memcpy(m_pCurrSEIMessage->pSEIMessage, pSEIMessageInfo->pSEIMessage, sizeof(CUSEIMESSAGE) * seiNumMessages);
    m_pCurrSEIMessage->sei_message_count = pSEIMessageInfo->sei_message_count;
    if (m_SEIMessagesDisplayOrder[pSEIMessageInfo->picIdx][0].pSEIData == NULL)
    {
        m_SEIMessagesDisplayOrder[pSEIMessageInfo->picIdx][0] = *m_pCurrSEIMessage;
    }
    else
    {
        m_SEIMessagesDisplayOrder[pSEIMessageInfo->picIdx][1] = *m_pCurrSEIMessage;
    }
    return 1;
}

NvDecoder::NvDecoder(int32_t gpuId, CUstream cuStream,CUcontext cuContext, bool bUseDeviceFrame, cudaVideoCodec eCodec, 
    bool bLowLatency, bool bEnableAsyncAllocations, int maxWidth, int maxHeight, OutputColorType eOutputColorType,
    bool bDeviceFramePitched, bool extract_user_SEI_Message,
    unsigned int clkRate, bool force_zero_latency, bool bWaitForSessionWarmUp
    ) :
    m_GpuId(gpuId), m_cuvidStream(cuStream),m_cuContext(cuContext), m_bUseDeviceFrame(bUseDeviceFrame), m_eCodec(eCodec), m_bLowLatency(bLowLatency),
    m_bEnableAsyncAllocations(bEnableAsyncAllocations),
    m_bDeviceFramePitched(bDeviceFramePitched), m_bExtractSEIMessage(extract_user_SEI_Message), m_nMaxWidth (maxWidth), m_nMaxHeight(maxHeight),
    m_eUserOutputColorType(eOutputColorType), m_bForce_zero_latency(force_zero_latency), m_bWaitForSessionWarmUp(bWaitForSessionWarmUp)
{


    const char* err = loadCuvidSymbols(&this->m_api,
#ifdef _WIN32
        "nvcuvid.dll");
#else
        "libnvcuvid.so.1");
#endif
    if (err) {
        constexpr const char* explanation =
#if defined(_WIN32)
            "Could not dynamically load nvcuvid.dll. Please ensure "
            "Nvidia Graphics drivers are correctly installed!";
#else
            "Could not dynamically load libnvcuvid.so.1. Please "
            "ensure Nvidia Graphics drivers are correctly installed!\n"
            "If using Docker please make sure that your Docker image was "
            "launched with \"video\" driver capabilty (see "
            "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
            "user-guide.html#driver-capabilities)";
#endif
        auto description = cuvid_dlerror();
        if (description) {
            throw std::runtime_error(std::string(err) + ": " +
                std::string(description) + "\n" + explanation);
        }
        else {
            throw std::runtime_error(std::string(err) + "\n" + explanation);
        }
    }
    if (m_bEnableAsyncAllocations)
    {
        LOG(INFO) << "enabling stream aware allocations!" << std::endl;
        if (m_cuContext != 0 && m_cuvidStream != 0)
        {
            CUDA_DRVAPI_CALL(cuEventCreate(&m_bCUEvent, 0));
        }
        else
        {
            throw std::runtime_error("Please provide CUDA context and CUDA stream that application has created");
        }
    }
  
    m_nMaxWidth = ALIGN(m_nMaxWidth, 32);
    m_nMaxHeight = ALIGN(m_nMaxHeight, 32);

    decoderSessionID = 0;

    if (m_bExtractSEIMessage)
    {
        m_fpSEI = fopen("sei_message.txt", "wb");
        m_pCurrSEIMessage = new CUVIDSEIMESSAGEINFO;
        memset(&m_SEIMessagesDisplayOrder, 0, sizeof(m_SEIMessagesDisplayOrder));
    }

    CUVIDPARSERPARAMS videoParserParameters = {};
    videoParserParameters.CodecType = eCodec;
    videoParserParameters.ulMaxNumDecodeSurfaces = 1;
    videoParserParameters.ulClockRate = clkRate;
    videoParserParameters.ulMaxDisplayDelay = m_bLowLatency ? 0 : 1;
    videoParserParameters.pfnDisplayPicture = m_bForce_zero_latency ? NULL : HandlePictureDisplayProc;
    videoParserParameters.pUserData = this;
    if (m_bWaitForSessionWarmUp)
    {
        videoParserParameters.pfnSequenceCallback = HandleVideoSequenceProcPerf;
    }
    else {
        videoParserParameters.pfnSequenceCallback = HandleVideoSequenceProc;
    }
    videoParserParameters.pfnDecodePicture = HandlePictureDecodeProc;
    videoParserParameters.pfnDisplayPicture = m_bForce_zero_latency ? NULL : HandlePictureDisplayProc;
    videoParserParameters.pfnGetOperatingPoint = HandleOperatingPointProc;
    videoParserParameters.pfnGetSEIMsg = m_bExtractSEIMessage ? HandleSEIMessagesProc : NULL;
    NVDEC_API_CALL(m_api.cuvidCreateVideoParser(&m_hParser, &videoParserParameters));


}

NvDecoder::~NvDecoder() {



    START_TIMER
    int64_t elapsedTime = 0;

    if (m_pCurrSEIMessage) {
        delete m_pCurrSEIMessage;
        m_pCurrSEIMessage = NULL;
    }

    if (m_fpSEI) {
        fclose(m_fpSEI);
        m_fpSEI = NULL;
    }

    if (m_hParser) {
        m_api.cuvidDestroyVideoParser(m_hParser);
    }


    cuCtxPushCurrent(m_cuContext);

    if (m_hDecoder) {
        m_api.cuvidDestroyDecoder(m_hDecoder);
    }

    std::lock_guard<std::mutex> lock(m_mtxVPFrame);


    for (uint8_t *pFrame : m_vpFrame)
    {
        if (m_bUseDeviceFrame)
        {
            if (m_bEnableAsyncAllocations)
            {
                cuMemFreeAsync((*(CUdeviceptr*)&pFrame), NULL);//sync on NULL stream to ensure that all work is completed before dtor
            }
            else
            {
                cuMemFree((CUdeviceptr)pFrame);
            }
            
        }
        else
        {
            delete[] pFrame;
        }
    }
    if (!m_bUseDeviceFrame)
    {
        cuMemFree((CUdeviceptr)m_dpScratchFrame);
    }

    if (m_bEnableAsyncAllocations)
    {
        cuEventDestroy(m_bCUEvent);
    }

    for (auto& v : m_DecodedFrameEvent)
    {
        cuEventDestroy(v);
    }


    
    cuCtxPopCurrent(NULL);

    STOP_TIMER("Session Deinitialization Time: ");

    NvDecoder::addDecoderSessionOverHead(getDecoderSessionID(), elapsedTime);


}

void NvDecoder::CUStreamWaitOnEvent(CUstream _stream)
{
    if (m_bEnableAsyncAllocations)
    {
        CUDA_DRVAPI_CALL(cuStreamWaitEvent(_stream, m_bCUEvent, 0));
    }
    
}

void NvDecoder::CUStreamSyncOnEvent()
{
    if (m_bCUEvent != NULL && m_bEnableAsyncAllocations)
    {
        CUDA_DRVAPI_CALL(cuEventSynchronize(m_bCUEvent));
    }
    
}

int NvDecoder::Decode(const uint8_t *pData, int nSize, int nFlags, int64_t nTimestamp)
{

    // printf("called decode");getchar();
    NVTX_SCOPED_RANGE("decodehelper::decodeframe")
    m_nDecodedFrame = 0;
    m_nDecodedFrameReturned = 0;
    CUVIDSOURCEDATAPACKET packet = { 0 };
    packet.payload = pData;
    packet.payload_size = nSize;
    packet.flags = nFlags | CUVID_PKT_TIMESTAMP;
    packet.timestamp = nTimestamp;
    if ((!pData || nSize == 0) && (nFlags != CUVID_PKT_DISCONTINUITY)){
        packet.flags |= CUVID_PKT_ENDOFSTREAM;
    }
    NVDEC_API_CALL(m_api.cuvidParseVideoData(m_hParser, &packet));

    // printf("called decode bye");getchar();


    return m_nDecodedFrame;
}

uint8_t* NvDecoder::GetFrame(int64_t* pTimestamp, SEI_MESSAGE *pSEIMessage, CUevent* decoderFrameEvent)
{
    if (m_nDecodedFrame > 0)
    {
        std::lock_guard<std::mutex> lock(m_mtxVPFrame);
        m_nDecodedFrame--;
        if (pTimestamp)
            *pTimestamp = m_vTimestamp[m_nDecodedFrameReturned];
        if (m_bExtractSEIMessage && pSEIMessage)
            *pSEIMessage = m_vSEIMessage[m_nDecodedFrameReturned];
        if (decoderFrameEvent)
        {
            *decoderFrameEvent = m_DecodedFrameEvent[m_nDecodedFrameReturned];
        }
        return m_vpFrame[m_nDecodedFrameReturned++];
    }

    return NULL;
}

uint8_t* NvDecoder::GetLockedFrame(int64_t* pTimestamp, SEI_MESSAGE *pSEIMessage, CUevent* decoderFrameEvent)
{
    uint8_t *pFrame;
    uint64_t timestamp;
    if (m_nDecodedFrame > 0) {
        std::lock_guard<std::mutex> lock(m_mtxVPFrame);
        m_nDecodedFrame--;
        // Frame related
        pFrame = m_vpFrame[0];
        m_vpFrame.erase(m_vpFrame.begin(), m_vpFrame.begin() + 1);
        m_LockedFrames.push_back(pFrame);
        // event related
        auto event = m_DecodedFrameEvent[0];
        m_DecodedFrameEvent.erase(m_DecodedFrameEvent.begin(), m_DecodedFrameEvent.begin() + 1);
        m_LockedEvents.push_back(event);
        if (decoderFrameEvent)
        {
            *decoderFrameEvent = event;
        }
        //timestamp related  
        timestamp = m_vTimestamp[0];
        m_vTimestamp.erase(m_vTimestamp.begin(), m_vTimestamp.begin() + 1);
        
        if (pTimestamp)
            *pTimestamp = timestamp;

        // sei message related
        if (m_bExtractSEIMessage && pSEIMessage)
        {
            auto seiMessage = m_vSEIMessage[0];
            m_vSEIMessage.erase(m_vSEIMessage.begin(), m_vSEIMessage.begin() + 1);
            *pSEIMessage = seiMessage;
        }
        
        
        return pFrame;
    }

    return NULL;
}

void NvDecoder::UnlockFrame(uint8_t **pFrame)
{
    std::lock_guard<std::mutex> lock(m_mtxVPFrame);
    m_vpFrame.insert(m_vpFrame.end(), &pFrame[0], &pFrame[1]);
    
    // add a dummy entry for timestamp
    uint64_t timestamp[2] = {0};
    m_vTimestamp.insert(m_vTimestamp.end(), &timestamp[0], &timestamp[1]);
    if (m_bExtractSEIMessage)
    {
        m_vSEIMessage.resize(m_vSEIMessage.size() + 2);
    }
    
}

void NvDecoder::UnlockFrame(uint8_t* pFrame)
{
    std::lock_guard<std::mutex> lock(m_mtxVPFrame);

    if (m_LockedFrames.size() != m_LockedEvents.size())
    {
        LOG(ERROR) << "Locked frames and locked events queues have mismatch in size\n";
    }
    // Remove if present in m_LockedFrames
    auto lockedFrames = m_LockedFrames.begin();
    auto lockedEvents = m_LockedEvents.begin();
    for (; lockedFrames != m_LockedFrames.end() && lockedEvents != m_LockedEvents.end(); ++lockedFrames, ++lockedEvents )
    {
        if (*lockedFrames == pFrame)
        {
            m_vpFrame.insert(m_vpFrame.end(), pFrame);

            // add a dummy entry for timestamp
            uint64_t timestamp[1] = { 0 };
            m_vTimestamp.insert(m_vTimestamp.end(), timestamp[0]);

            m_LockedFrames.erase(lockedFrames);
            m_DecodedFrameEvent.insert(m_DecodedFrameEvent.end(), *lockedEvents);
            m_LockedEvents.erase(lockedEvents);
            break;
        }
    }
    if (m_bExtractSEIMessage)
    {
        m_vSEIMessage.resize(m_vSEIMessage.size() + 1);
    }
    
}

// Unlock in order of locking. We might need a overload where we take in frames to unlock. This
// will be needed if we decide to expose unlock to application.
void NvDecoder::UnlockLockedFrames(uint32_t size)
{
    std::lock_guard<std::mutex> lock(m_mtxVPFrame);
    if (size > m_LockedFrames.size())
    {
        LOG(WARNING) << "Size of unlock requests exceeds locked frames. Got "
                     << size << ". Max allowed is " << m_LockedFrames.size()
                     << ". Unlocking " << size << " frames";
        size = m_LockedFrames.size();
    }
    if (size > m_LockedEvents.size())
    {
        LOG(WARNING) << "Size of unlock requests exceeds locked events. Got "
                     << size << ". Max allowed is " << m_LockedEvents.size()
                     << ". Unlocking " << size << " frames";
        size = m_LockedEvents.size();
    }
    m_vpFrame.insert(m_vpFrame.end(), m_LockedFrames.begin(), m_LockedFrames.begin() + size);
    m_vTimestamp.insert(m_vTimestamp.end(), size, 0);
    m_DecodedFrameEvent.insert(m_DecodedFrameEvent.end(), m_LockedEvents.begin(), m_LockedEvents.begin() + size);
    if (m_bExtractSEIMessage)
    {
        m_vSEIMessage.resize(m_vSEIMessage.size() + size);
    }
    m_LockedFrames.erase(m_LockedFrames.begin(), m_LockedFrames.begin() + size);
    m_LockedEvents.erase(m_LockedEvents.begin(), m_LockedEvents.begin() + size);
}