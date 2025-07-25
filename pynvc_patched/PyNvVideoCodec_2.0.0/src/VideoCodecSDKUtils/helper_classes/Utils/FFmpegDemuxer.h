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
#pragma once

#ifdef _WIN32
    #include <string.h>
    #define strcasecmp _stricmp
#else
    #include <strings.h>
#endif

extern "C" {
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavcodec/avcodec.h>
/* Explicitly include bsf.h when building against FFmpeg 4.3 (libavcodec 58.45.100) or later for backward compatibility */
#if LIBAVCODEC_VERSION_INT >= 3824484
#include <libavcodec/bsf.h>
#endif
}
#ifndef DEMUX_ONLY
#include "cuviddec.h"
#include "nvcuvid.h"
#endif
#include "NvCodecUtils.h"
#include <algorithm>
#include <future>
#include <stdexcept>
#include <pybind11/functional.h>
namespace py = pybind11;
using namespace std;
//---------------------------------------------------------------------------
//! \file FFmpegDemuxer.h 
//! \brief Provides functionality for stream demuxing
//!
//! This header file is used by Decode/Transcode apps to demux input video clips before decoding frames from it. 
//---------------------------------------------------------------------------

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

struct ScannedStreamMetadata
{
    uint32_t width;
    uint32_t height;
    uint32_t numFrames;
    uint32_t averageFPS;
    double duration;
    float bitrate;
    std::string codecName;
    std::vector<uint32_t> keyFrameIndices;
    std::vector<uint32_t> packetSize;
    std::vector<int64_t> pts;
    std::vector<int64_t> dts;    
};

struct StreamMetadata
{
    uint32_t width;
    uint32_t height;
    uint32_t numFrames;
    uint32_t averageFPS;
    double duration;
    float bitrate;
    std::string codecName;
};

struct PacketInfo
{
    int64_t pts;
    int64_t dts;
    bool isKeyFrame;
    uint32_t packetSize;
};

enum SeekMode {
    /* Seek for exact frame number.
     * Suited for standalone demuxer seek. */
    EXACT_FRAME = 0,

    /* Seek for previous key frame in past.
     * Suitable for seek & decode.  */
     PREV_KEY_FRAME = 1,

     /* Seek for nearest key frame in future.
     * Suitable for seek & decode.  */
     NEAREST_FUTURE_KEY_FRAME = 2,

     SEEK_MODE_NUM_ELEMS
};

enum SeekCriteria {
    /* Seek frame by number.
     */
    BY_NUMBER = 0,

    /* Seek frame by timestamp.
     */
    BY_TIMESTAMP = 1,

    SEEK_CRITERIA_NUM_ELEMS
};

struct SeekContext {
    /* Will be set to false for default ctor, true otherwise;
     */
    bool use_seek;

    /* Frame we want to get. Set by user.
     * Shall be set to frame timestamp in case seek is done by time.
     */
    uint64_t seek_frame;

    /* Mode in which we seek. */
    SeekMode mode;

    /* Criteria by which we seek. */
    SeekCriteria crit;

    /* PTS of frame found after seek. */
    int64_t out_frame_pts;

    /* Duration of frame found after seek. */
    int64_t out_frame_duration;

    /* Number of frames that were decoded during seek. */
    uint64_t num_frames_decoded;

    SeekContext()
        : use_seek(false), seek_frame(0), mode(NEAREST_FUTURE_KEY_FRAME), crit(BY_NUMBER),
        out_frame_pts(0), out_frame_duration(0), num_frames_decoded(0U)
    {
    }

    SeekContext(uint64_t frame_id)
        : use_seek(true), seek_frame(frame_id), mode(NEAREST_FUTURE_KEY_FRAME),
        crit(BY_NUMBER), out_frame_pts(0), out_frame_duration(0),
        num_frames_decoded(0U)
    {
    }


    SeekContext& operator=(const SeekContext& other)
    {
        use_seek = other.use_seek;
        seek_frame = other.seek_frame;
        mode = other.mode;
        crit = other.crit;
        out_frame_pts = other.out_frame_pts;
        out_frame_duration = other.out_frame_duration;
        num_frames_decoded = other.num_frames_decoded;
        return *this;
    }
};

// Use this macro if ffmpeg API returns AVERROR_xxx on error
#define FFMPEG_API_CALL( ffmpegAPI )                                                                                 \
    do                                                                                                               \
    {                                                                                                                \
        int errorCode = ffmpegAPI;                                                                                   \
        if( errorCode < 0)                                                                                           \
        {                                                                                                            \
            char temp[256];                                                                                          \
            av_strerror(errorCode, temp, 256);                                                                       \
            std::ostringstream errorLog;                                                                             \
            errorLog << #ffmpegAPI << " returned error \" " << temp << "\"";                                         \
            throw PyNvVCException<PyNvVCGenericError>::makePyNvVCException(errorLog.str(), errorCode, __FUNCTION__, __FILE__, __LINE__); \
        } \
    } while (0)

/**
* @brief libavformat wrapper class. Retrieves the elementary encoded stream from the container format.
*/
cudaVideoChromaFormat FFmpeg2NvChromaFormat(AVPixelFormat id);
class FFmpegDemuxer {
private:
    AVFormatContext *fmtc = NULL;
    AVIOContext *avioc = NULL;
    AVPacket* pkt = NULL; /*!< AVPacket stores compressed data typically exported by demuxers and then passed as input to decoders */
    AVPacket* pktFiltered = NULL;
    AVBSFContext *bsfc = NULL;
    AVCodec* codec = NULL;
    AVCodecContext* codecContext = NULL;

    int iVideoStream;
    int iAudioStream;
    bool bMp4H264, bMp4HEVC, bMp4MPEG4, is_seekable;
    AVCodecID eVideoCodec;
    AVPixelFormat eChromaFormat;
    int nWidth, nHeight, nBitDepth, nBPP, nChromaHeight;
    double timeBase = 0.0;
    int64_t userTimeScale = 0; 
    double framerate = 0.0;
    double avg_framerate = 0.0;
    int64_t nBitrate = 0;
    double nDuration = 0.0;
    int64_t nNumFramesfromStream = 0;
    AVColorSpace color_space = AVCOL_SPC_UNSPECIFIED;
    AVColorRange color_range = AVCOL_RANGE_UNSPECIFIED;
    uint64_t packet_duration = 0;
    uint8_t *pDataWithHeader = NULL;
    unsigned int frameCount = 0;
    std::vector<uint32_t> keyFrameIndices;

public:
    class DataProvider {
    public:
        virtual ~DataProvider() {}
        virtual int GetData(uint8_t *pBuf, int nBuf) = 0;
    };

    class PyByteArrayProvider : public FFmpegDemuxer::DataProvider
    {
    private:

        std::function<int(py::bytearray)> callback;
        int mBytesReadTillNow = 0;

    public:

        PyByteArrayProvider(std::function<int(py::bytearray)> _callback)
        {
            callback = _callback;
        }

        virtual int GetData(uint8_t* pBuf, int nBuf)
        {
            auto store = py::bytearray((const char*)pBuf, nBuf);
            int bytesCopied = callback(store);
            py::buffer_info info(py::buffer(store).request());
            uint8_t* srcBufferPtr = reinterpret_cast<uint8_t*>(info.ptr);
            if (bytesCopied == 0)
            {
                return AVERROR_EOF;
            }
            /* copy internal buffer data to buf */
            memcpy(pBuf, srcBufferPtr, bytesCopied);
            mBytesReadTillNow += bytesCopied;
            return bytesCopied;
            
        }
    };

private:
    /**
    *   @brief  Private constructor to initialize libavformat resources.
    *   @param  fmtc - Pointer to AVFormatContext allocated inside avformat_open_input()
    */
    FFmpegDemuxer(AVFormatContext *fmtc, int64_t timeScale = 1000 /*Hz*/) : fmtc(fmtc) {
        if (!fmtc) {
            throw std::runtime_error("No AVFormatContext provided");
        }
        const char* envLevel = std::getenv("LOGGER_LEVEL");
        
        if (envLevel != NULL)
        {
            std::string level(envLevel);
            std::transform(level.begin(), level.end(), level.begin(), ::toupper);

            if (level == "TRACE")
            {
                av_log_set_level(AV_LOG_VERBOSE);
            }
            else if (level == "DEBUG")
            {
                av_log_set_level(AV_LOG_DEBUG);
            }
            else if (level == "INFO")
            {
                av_log_set_level(AV_LOG_INFO);
            }
            else if (level == "WARN")
            {
                av_log_set_level(AV_LOG_WARNING);
            }
            else if (level == "ERROR")
            {
                av_log_set_level(AV_LOG_ERROR);
            }
            else if (level == "FATAL")
            {
                av_log_set_level(AV_LOG_FATAL);
            }
            
        }
        else
        {
            av_log_set_level(AV_LOG_QUIET);
        }
        
        
        // Allocate the AVPackets and initialize to default values
        pkt = av_packet_alloc();
        pktFiltered = av_packet_alloc();
        if (!pkt || !pktFiltered) {
            throw std::runtime_error("AVPacket allocation failed");
        }

        LOG(DEBUG) << "Media format: " << fmtc->iformat->long_name << " (" << fmtc->iformat->name << ")";

        FFMPEG_API_CALL(avformat_find_stream_info(fmtc, NULL));
        iVideoStream = av_find_best_stream(fmtc, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
        if (iVideoStream < 0) {
            av_packet_free(&pkt);
            av_packet_free(&pktFiltered);
            throw std::runtime_error("Could not find stream in input file");
        }

        iAudioStream = av_find_best_stream(fmtc, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);

        eVideoCodec = fmtc->streams[iVideoStream]->codecpar->codec_id;
        nWidth = fmtc->streams[iVideoStream]->codecpar->width;
        nHeight = fmtc->streams[iVideoStream]->codecpar->height;
        eChromaFormat = (AVPixelFormat)fmtc->streams[iVideoStream]->codecpar->format;
        AVRational rTimeBase = fmtc->streams[iVideoStream]->time_base;
        timeBase = av_q2d(rTimeBase);
        userTimeScale = timeScale;
        framerate = (double)fmtc->streams[iVideoStream]->r_frame_rate.num /
            (double)fmtc->streams[iVideoStream]->r_frame_rate.den;
        avg_framerate = (double)fmtc->streams[iVideoStream]->avg_frame_rate.num /
            (double)fmtc->streams[iVideoStream]->avg_frame_rate.den;
        // Set bit depth, chroma height, bits per pixel based on eChromaFormat of input
        // eChromaFormat = (AVPixelFormat)fmtc->streams[iVideoStream]->codecpar->format;
        nBitrate = (AVPixelFormat)fmtc->streams[iVideoStream]->codecpar->bit_rate;
        nDuration = (AVPixelFormat)fmtc->streams[iVideoStream]->duration * timeBase;
        nNumFramesfromStream = (AVPixelFormat)fmtc->streams[iVideoStream]->nb_frames;
        color_space = fmtc->streams[iVideoStream]->codecpar->color_space;
        color_range = fmtc->streams[iVideoStream]->codecpar->color_range;
        switch (eChromaFormat)
        {
        case AV_PIX_FMT_YUV420P10LE:
        case AV_PIX_FMT_GRAY10LE:   // monochrome is treated as 420 with chroma filled with 0x0
            nBitDepth = 10;
            nChromaHeight = (nHeight + 1) >> 1;
            nBPP = 2;
            break;
        case AV_PIX_FMT_YUV420P12LE:
            nBitDepth = 12;
            nChromaHeight = (nHeight + 1) >> 1;
            nBPP = 2;
            break;
        case AV_PIX_FMT_YUV444P10LE:
            nBitDepth = 10;
            nChromaHeight = nHeight << 1;
            nBPP = 2;
            break;
        case AV_PIX_FMT_YUV444P12LE:
            nBitDepth = 12;
            nChromaHeight = nHeight << 1;
            nBPP = 2;
            break;
        case AV_PIX_FMT_YUV444P:
            nBitDepth = 8;
            nChromaHeight = nHeight << 1;
            nBPP = 1;
            break;
        case AV_PIX_FMT_YUV422P10LE:
            nBitDepth = 10;
            nChromaHeight = nHeight;
            nBPP = 2;
            break;
        case AV_PIX_FMT_YUV422P12LE:
            nBitDepth = 12;
            nChromaHeight = nHeight;
            nBPP = 2;
            break;
        case AV_PIX_FMT_YUV422P:
            nBitDepth = 8;
            nChromaHeight = nHeight;
            nBPP = 1;
            break;
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
        case AV_PIX_FMT_YUVJ422P:   // jpeg decoder output is subsampled to NV12 for 422/444 so treat it as 420
        case AV_PIX_FMT_YUVJ444P:   // jpeg decoder output is subsampled to NV12 for 422/444 so treat it as 420
        case AV_PIX_FMT_GRAY8:      // monochrome is treated as 420 with chroma filled with 0x0
            nBitDepth = 8;
            nChromaHeight = (nHeight + 1) >> 1;
            nBPP = 1;
            break;
        default:
            LOG(DEBUG) << "ChromaFormat not recognized. Assuming 420";
            eChromaFormat = AV_PIX_FMT_YUV420P;
            nBitDepth = 8;
            nChromaHeight = (nHeight + 1) >> 1;
            nBPP = 1;
        }

        bMp4H264 = eVideoCodec == AV_CODEC_ID_H264 && (
                !strcmp(fmtc->iformat->long_name, "QuickTime / MOV") 
                || !strcmp(fmtc->iformat->long_name, "FLV (Flash Video)") 
                || !strcmp(fmtc->iformat->long_name, "Matroska / WebM")
            );
        bMp4HEVC = eVideoCodec == AV_CODEC_ID_HEVC && (
                !strcmp(fmtc->iformat->long_name, "QuickTime / MOV")
                || !strcmp(fmtc->iformat->long_name, "FLV (Flash Video)")
                || !strcmp(fmtc->iformat->long_name, "Matroska / WebM")
            );

        bMp4MPEG4 = eVideoCodec == AV_CODEC_ID_MPEG4 && (
                !strcmp(fmtc->iformat->long_name, "QuickTime / MOV")
                || !strcmp(fmtc->iformat->long_name, "FLV (Flash Video)")
                || !strcmp(fmtc->iformat->long_name, "Matroska / WebM")
            );

        // Initialize bitstream filter and its required resources
        if (bMp4H264) {
            const AVBitStreamFilter *bsf = av_bsf_get_by_name("h264_mp4toannexb");
            if (!bsf) {
                av_packet_free(&pkt);
                av_packet_free(&pktFiltered);
                throw std::runtime_error("av_bsf_get_by_name() failed");
            }
            FFMPEG_API_CALL(av_bsf_alloc(bsf, &bsfc));
            avcodec_parameters_copy(bsfc->par_in, fmtc->streams[iVideoStream]->codecpar);
            FFMPEG_API_CALL(av_bsf_init(bsfc));
        }
        if (bMp4HEVC) {
            const AVBitStreamFilter *bsf = av_bsf_get_by_name("hevc_mp4toannexb");
            if (!bsf) {
                av_packet_free(&pkt);
                av_packet_free(&pktFiltered);
                throw std::runtime_error("av_bsf_get_by_name() failed");
            }
            FFMPEG_API_CALL(av_bsf_alloc(bsf, &bsfc));
            avcodec_parameters_copy(bsfc->par_in, fmtc->streams[iVideoStream]->codecpar);
            FFMPEG_API_CALL(av_bsf_init(bsfc));
        }

        bool seekable_format = (strcmp(fmtc->iformat->name, "hevc") != 0 &&
                                strcmp(fmtc->iformat->name, "h264") != 0);
        is_seekable = fmtc->pb->seekable && seekable_format;
    }


    AVFormatContext *CreateFormatContext(DataProvider *pDataProvider) {

        AVFormatContext *ctx = NULL;
        if (!(ctx = avformat_alloc_context())) {
            throw std::runtime_error("avformat_alloc_context() failed");
        }

        uint8_t *avioc_buffer = NULL;
        int avioc_buffer_size =  8 * 1024 * 1024;
        avioc_buffer = (uint8_t *)av_malloc(avioc_buffer_size);
        if (!avioc_buffer) {
            throw std::runtime_error("av_malloc() failed");
        }
        avioc = avio_alloc_context(avioc_buffer, avioc_buffer_size,
            0, pDataProvider, &ReadPacket, NULL, NULL);
        if (!avioc) {
            throw std::runtime_error("avio_alloc_context() failed");
        }
        ctx->pb = avioc;
        is_seekable = ctx->pb->seekable;

        ctx->flags = AVFMT_FLAG_CUSTOM_IO;

        FFMPEG_API_CALL(avformat_open_input(&ctx, NULL, NULL, NULL));
        return ctx;
    }

    /**
    *   @brief  Allocate and return AVFormatContext*.
    *   @param  szFilePath - Filepath pointing to input stream.
    *   @return Pointer to AVFormatContext
    */
     AVFormatContext *CreateFormatContext(const char *szFilePath) {
        avformat_network_init();

        AVFormatContext *ctx = NULL;
        FFMPEG_API_CALL(avformat_open_input(&ctx, szFilePath, NULL, NULL));
        return ctx;
    }


public:
    // Make the timescale constructor explicit to avoid ambiguity
    explicit FFmpegDemuxer(const char *szFilePath, int64_t timescale = 1000 /*Hz*/) 
        : FFmpegDemuxer(CreateFormatContext(szFilePath), timescale) {}
    
    explicit FFmpegDemuxer(DataProvider *pDataProvider) 
        : FFmpegDemuxer(CreateFormatContext(pDataProvider)) {avioc = fmtc->pb;}
    ~FFmpegDemuxer() {

        if (!fmtc) {
            // Should we throw error from destructor??
            return;
        }

        if (pkt) {
            av_packet_free(&pkt);
        }
        if (pktFiltered) {
            av_packet_free(&pktFiltered);
        }

        if (bsfc) {
            av_bsf_free(&bsfc);
        }

        avformat_close_input(&fmtc);
        
        if (avioc) {
            av_freep(&avioc->buffer);
            av_freep(&avioc);
        }

        if (pDataWithHeader) {
            av_free(pDataWithHeader);
        }
    }
    AVFormatContext* GetAVFormatContext() {
        return fmtc;
    }
    AVCodecID GetVideoCodec() {
        return eVideoCodec;
    }
    AVPixelFormat GetChromaFormat() {
        return eChromaFormat;
    }
    int GetWidth() {
        return nWidth;
    }
    int GetHeight() {
        return nHeight;
    }
    int GetBitDepth() {
        return nBitDepth;
    }
    int GetFrameSize() {
        return nWidth * (nHeight + nChromaHeight) * nBPP;
    }

    double GetFrameRate()
    {
        return get_fps();
    }

    AVPixelFormat GetPixelFormat() const { return eChromaFormat; }

    AVColorSpace GetColorSpace() const { return color_space; }

    AVColorRange GetColorRange() const { return color_range; }

    int64_t GetDuration() const { return fmtc->duration; }

    bool IsVFR() const { 
        return framerate != avg_framerate; 
    }
    int64_t TsFromTime(double ts_sec)
    {
        /* Internal timestamp representation is integer, so multiply to AV_TIME_BASE
         * and switch to fixed point precision arithmetics; */
        auto const ts_tbu = llround(ts_sec * AV_TIME_BASE);


        // Rescale the timestamp to value represented in stream base units;
        AVRational factor;
        factor.num = 1;
        factor.den = AV_TIME_BASE;
        
        return av_rescale_q(ts_tbu, factor, fmtc->streams[iVideoStream]->time_base);
    }

    int64_t TsFromFrameNumber(int64_t frame_num)
    {
        auto const ts_sec = (double)frame_num / framerate;
        return TsFromTime(ts_sec);
    }

    double r2d(AVRational r) const
    {
        return r.num == 0 || r.den == 0 ? 0. : (double)r.num / (double)r.den;
    }
    double get_fps() const
    {

        double eps_zero = 0.000025;

        double fps = r2d(fmtc->streams[iVideoStream]->avg_frame_rate);

        if (fps < eps_zero)
        {
            fps = r2d(av_guess_frame_rate(fmtc, fmtc->streams[iVideoStream], NULL));
        }
        if (fps < eps_zero)
        {
            fps = 1.0 / r2d(fmtc->streams[iVideoStream]->time_base);
        }

        return fps;
    }
    double get_audio_fps() const
    {
        double fps = r2d(fmtc->streams[iAudioStream]->avg_frame_rate);
        return fps;
    }
    double dts_to_sec(int64_t dts) const
    {
        return (double)(dts - fmtc->streams[iVideoStream]->start_time) *
            r2d(fmtc->streams[iVideoStream]->time_base);
    }
    int64_t dts_to_frame_number(int64_t dts)
    {
        std::string container = GetContainerName();
        double sec;
        
        if (container == "flv" || container == "mov") {
            // FLV and MOV use direct timebase conversion without start_time adjustment
            sec = (double)(dts) * timeBase;
        } else {
            // Other containers need start_time adjustment
            sec = dts_to_sec(dts);
        }
        
        return (int64_t)(get_fps() * sec + 0.5);
    }

    AVStream* GetVideoStream()
    {
        if (fmtc != NULL)
        {
            return fmtc->streams[iVideoStream];
        }
        else
        {
            PYNVVC_THROW_ERROR("AVFormatContext is NULL", CUDA_ERROR_NOT_SUPPORTED);
        }
        
    }

    std::string GetContainerName() {
        if (!fmtc || !fmtc->iformat) {
            return "unknown";
        }

        // Check if it's a mov/mp4 container
        if (strcmp(fmtc->iformat->name, "mov,mp4,m4a,3gp,3g2,mj2") == 0) {
            // Check major_brand metadata to differentiate
            AVDictionaryEntry* tag = av_dict_get(fmtc->metadata, "major_brand", NULL, 0);
            if (tag) {
                if (strcmp(tag->value, "qt  ") == 0) {
                    return "mov";
                } else if (strcmp(tag->value, "mp42") == 0 || 
                          strcmp(tag->value, "isom") == 0 ||
                          strcmp(tag->value, "mp41") == 0) {
                    return "mp4";
                }
            }
            
            // Fallback to extension check if metadata not available
            const char* filename = fmtc->url;
            const char* ext = strrchr(filename, '.');
            if (ext) {
                if (strcasecmp(ext, ".mp4") == 0) {
                    return "mp4";
                } else if (strcasecmp(ext, ".mov") == 0) {
                    return "mov";
                }
            }
        }

        // For other formats, return the format name
        return fmtc->iformat->name;
    }

    AVStream* GetAudioStream()
    {
        if (fmtc != NULL)
        {
            return fmtc->streams[iAudioStream];
        }
        else
        {
            PYNVVC_THROW_ERROR("AVFormatContext is NULL", CUDA_ERROR_NOT_SUPPORTED);
        }

    }

    int GetVideoStreamId()
    {
        return iVideoStream;
    }

    int GetAudioStreamId()
    {
        return iAudioStream;
    }

    bool Demux(uint8_t** ppVideo, int* pnVideoBytes, int64_t& pts, int64_t& dts, uint64_t& duration, uint64_t& pos, bool& isKeyFrame) {

        NVTX_SCOPED_RANGE("demux")
        if (!fmtc) {
            return false;
        }

        *pnVideoBytes = 0;

        if (pkt->data) {
            av_packet_unref(pkt);
        }

        int e = 0;
        while ((e = av_read_frame(fmtc, pkt)) >= 0 && pkt->stream_index != iVideoStream) {
            av_packet_unref(pkt);
        }
        if (e < 0) {
            return false;
        }

        if (bMp4H264 || bMp4HEVC) {
            if (pktFiltered->data) {
                av_packet_unref(pktFiltered);
            }
            ck(av_bsf_send_packet(bsfc, pkt));
            ck(av_bsf_receive_packet(bsfc, pktFiltered));
            *ppVideo = pktFiltered->data;
            *pnVideoBytes = pktFiltered->size;
            pts = (int64_t)(pktFiltered->pts);
            dts = (int64_t)(pktFiltered->dts);
            duration = (uint64_t)pktFiltered->duration;
            packet_duration = (uint64_t)pktFiltered->duration;
            pos = (uint64_t)pktFiltered->pos;
            isKeyFrame = pktFiltered->flags & AV_PKT_FLAG_KEY ? true : false;

        }
        else {

            if (bMp4MPEG4 && (frameCount == 0)) {

                int extraDataSize = fmtc->streams[iVideoStream]->codecpar->extradata_size;

                if (extraDataSize > 0) {

                    // extradata contains start codes 00 00 01. Subtract its size
                    pDataWithHeader = (uint8_t*)av_malloc(extraDataSize + pkt->size - 3 * sizeof(uint8_t));

                    if (!pDataWithHeader) {
                        PYNVVC_THROW_ERROR("av_malloc() failed", CUDA_ERROR_NOT_SUPPORTED);
                    }

                    memcpy(pDataWithHeader, fmtc->streams[iVideoStream]->codecpar->extradata, extraDataSize);
                    memcpy(pDataWithHeader + extraDataSize, pkt->data + 3, pkt->size - 3 * sizeof(uint8_t));

                    *ppVideo = pDataWithHeader;
                    *pnVideoBytes = extraDataSize + pkt->size - 3 * sizeof(uint8_t);
                }

            }
            else {
                *ppVideo = pkt->data;
                *pnVideoBytes = pkt->size;
            }

            pts = (int64_t)(pkt->pts);
            dts = (int64_t)(pkt->dts);
            duration = (uint64_t)pkt->duration;
            packet_duration = (uint64_t)pkt->duration;
            pos = (uint64_t)pkt->pos;
            isKeyFrame = pkt->flags & AV_PKT_FLAG_KEY;
        }

        frameCount++;

        return true;
    }

    bool DemuxA(uint8_t** ppVideo, int* pnVideoBytes, int64_t& pts, int64_t& dts, uint64_t& duration, uint64_t& pos, bool& isKeyFrame) {

        NVTX_SCOPED_RANGE("demux")
            if (!fmtc) {
                return false;
            }

        *pnVideoBytes = 0;

        if (pkt->data) {
            av_packet_unref(pkt);
        }

        int e = 0;
        while ((e = av_read_frame(fmtc, pkt)) >= 0 && pkt->stream_index != iAudioStream) {
            av_packet_unref(pkt);
        }
        if (e < 0) {
            return false;
        }

        *ppVideo = pkt->data;
        *pnVideoBytes = pkt->size;
        pts = pkt->pts;
        dts = pkt->dts;
        

        frameCount++;

        return true;
    }

    bool DemuxNoSkipAudio(uint8_t **ppVideo, int *pnVideoBytes, int64_t &pts, int64_t &dts, uint64_t &duration, uint64_t &pos, bool &isKeyFrame, int* isVideoPacket = NULL, int* streamIndex = NULL ) {
       
        NVTX_SCOPED_RANGE("demux")
        if (!fmtc) {
            return false;
        }

        *pnVideoBytes = 0;

        if (pkt->data) {
            av_packet_unref(pkt);
        }

        if (av_read_frame(fmtc, pkt) < 0) {
            if (isVideoPacket)
                *isVideoPacket = 1;
            if (streamIndex)
                *streamIndex = iVideoStream;
            return false;
        }
        
        
        if (pkt->stream_index == iVideoStream)
        {
            if (bMp4H264 || bMp4HEVC) {
                if (pktFiltered->data) {
                    av_packet_unref(pktFiltered);
                }
                ck(av_bsf_send_packet(bsfc, pkt));
                ck(av_bsf_receive_packet(bsfc, pktFiltered));
                *ppVideo = pktFiltered->data;
                *pnVideoBytes = pktFiltered->size;
                pts = (int64_t)(pktFiltered->pts);
                dts = (int64_t)(pktFiltered->dts);
                duration = (uint64_t)pktFiltered->duration;
                packet_duration = (uint64_t)pktFiltered->duration;
                pos = (uint64_t)pktFiltered->pos;
                isKeyFrame = pktFiltered->flags & AV_PKT_FLAG_KEY ? true : false;

            }
            else {

                if (bMp4MPEG4 && (frameCount == 0)) {

                    int extraDataSize = fmtc->streams[iVideoStream]->codecpar->extradata_size;

                    if (extraDataSize > 0) {

                        // extradata contains start codes 00 00 01. Subtract its size
                        pDataWithHeader = (uint8_t*)av_malloc(extraDataSize + pkt->size - 3 * sizeof(uint8_t));

                        if (!pDataWithHeader) {
                            PYNVVC_THROW_ERROR("av_malloc() failed", CUDA_ERROR_NOT_SUPPORTED);
                        }

                        memcpy(pDataWithHeader, fmtc->streams[iVideoStream]->codecpar->extradata, extraDataSize);
                        memcpy(pDataWithHeader + extraDataSize, pkt->data + 3, pkt->size - 3 * sizeof(uint8_t));

                        *ppVideo = pDataWithHeader;
                        *pnVideoBytes = extraDataSize + pkt->size - 3 * sizeof(uint8_t);
                    }

                }
                else 
                {
                    *ppVideo = pkt->data;
                    *pnVideoBytes = pkt->size;
                }
                pts = (int64_t)(pkt->pts);
                dts = (int64_t)(pkt->dts);
                duration = (uint64_t)pkt->duration;
                packet_duration = (uint64_t)pkt->duration;
                pos = (uint64_t)pkt->pos;
                isKeyFrame = pkt->flags & AV_PKT_FLAG_KEY;
            }
            if (isVideoPacket)
            {
                *isVideoPacket = 1;
            }
        }
        else
        {
            *ppVideo = pkt->data;
            *pnVideoBytes = pkt->size;
            pts = pkt->pts;
            dts = pkt->dts;
            if (isVideoPacket)
            {
                *isVideoPacket = 0;
            }
        }
        frameCount++;
        if (streamIndex)
        {
            *streamIndex = pkt->stream_index;
        }
        return true;
    }

    // Check if frame satisfies seek conditions;
    int is_seek_done(int64_t decodedFramePTS, int64_t target_frameIndex) {
        int64_t target_pts = 0;
        int eps = packet_duration / 2;
        target_pts = FrameToPts(fmtc->streams[iVideoStream], MAX(0, target_frameIndex));
        int64_t curr_frameIndex = dts_to_frame_number(decodedFramePTS);
        int seek_status = 0;

        if (decodedFramePTS == target_pts) {
            seek_status = 0;
        }
        else if (std::abs(decodedFramePTS - target_pts) <= eps)
        {
            seek_status = 0;
        }
        else if (decodedFramePTS > target_pts) {
            seek_status = 1;
        }
        else {
            seek_status = -1;
        };

        return seek_status;
    };

    bool IsSeekable()
    {
        return is_seekable;
    }

    int64_t FrameToPts(AVStream* pavStream, int frame) const
    {
        return (int64_t(frame) * pavStream->r_frame_rate.den * pavStream -> time_base.den) /
            (int64_t(pavStream->r_frame_rate.num) *
            pavStream->time_base.num);
    }
    
    bool Seek(uint32_t frameIdx)
    {
        if (!is_seekable) {
            PYNVVC_THROW_ERROR("Seek isn't supported for this input.", CUDA_ERROR_NOT_SUPPORTED);
            return false;
        }
        const AVIndexEntry* entry0 = avformat_index_get_entry(fmtc->streams[iVideoStream], 0);
        int64_t pts_offset = entry0->timestamp;
        int64_t iSeekTargetPTS = 0;
        std::string container = GetContainerName();
        if (container == "mov" ||
            container == "flv" ||
           container == "matroska,webm")
        {
            iSeekTargetPTS = FrameToPts(fmtc->streams[iVideoStream], frameIdx);
        }
        else
        {
            const AVIndexEntry* keyframeIndexEntry = avformat_index_get_entry(fmtc->streams[iVideoStream], frameIdx);
            iSeekTargetPTS = keyframeIndexEntry->timestamp;
        }
        iSeekTargetPTS = iSeekTargetPTS - pts_offset;
        
        int rv = av_seek_frame(
            fmtc, iVideoStream, iSeekTargetPTS, AVSEEK_FLAG_BACKWARD);
        if (rv < 0)
        {
            PYNVVC_THROW_ERROR("Failed to seek.", CUDA_ERROR_NOT_SUPPORTED);
        }
        
        return true;
    }

    bool Seek(SeekContext& seekCtx, uint8_t** ppVideo, int* pnVideoBytes)
    {

        if (!is_seekable) {
            cerr << "Seek isn't supported for this input." << endl;
            return false;
        }

        /* This will seek to nearest I-frame */
        auto seek_for_nearest_iframe = [&](PacketData& pkt_data, SeekContext& seek_ctx) {

            int frameIndex = seek_ctx.seek_frame;
            const AVIndexEntry* keyframeIndexEntry = avformat_index_get_entry(fmtc->streams[iVideoStream], frameIndex);
            int64_t iSeekTargetPTS = keyframeIndexEntry->timestamp;
            int rv = av_seek_frame(
                fmtc, iVideoStream, iSeekTargetPTS, AVSEEK_FLAG_BACKWARD);
            if (rv < 0)
                throw std::runtime_error("Failed to seek");

            };

        switch (seekCtx.mode) {
        case EXACT_FRAME:
        {
            //jump to the key frame just previous to target timestamp
            PacketData nearest_key_pkt = { 0 };
            seek_for_nearest_iframe(nearest_key_pkt, seekCtx);
        }
        break;
        case PREV_KEY_FRAME:

            break;
        case NEAREST_FUTURE_KEY_FRAME:

            break;
        default:
            throw runtime_error("Unsupported seek mode");
            break;
        }

        return true;
    }
    
    bool SeekAudioStream(uint32_t vframeIdx, uint32_t aframeIdx)
    {
        if (!is_seekable) {
            PYNVVC_THROW_ERROR("Seek isn't supported for this input.", CUDA_ERROR_NOT_SUPPORTED);
        }
        const AVIndexEntry* keyframeIndexEntry = avformat_index_get_entry(fmtc->streams[iVideoStream], vframeIdx);
        int64_t iSeekTargetPTS = keyframeIndexEntry->timestamp;
        int rv = av_seek_frame(
            fmtc, iVideoStream, iSeekTargetPTS, AVSEEK_FLAG_ANY);
        if (rv < 0)
        {
            PYNVVC_THROW_ERROR("Failed to seek.", CUDA_ERROR_NOT_SUPPORTED);
        }
        /*auto keyframeIndexEntry = avformat_index_get_entry(fmtc->streams[iAudioStream], aframeIdx);
        auto iSeekTargetPTS = keyframeIndexEntry->timestamp;
        auto rv = av_seek_frame(
            fmtc, iAudioStream, iSeekTargetPTS, AVSEEK_FLAG_BACKWARD);
        if (rv < 0)
        {
            PYNVVC_THROW_ERROR("Failed to seek.", CUDA_ERROR_NOT_SUPPORTED);
        }*/

        return true;
    }

    StreamMetadata GetStreamMetadata()
    {
        StreamMetadata streamMetadata = {};
        streamMetadata.width = nWidth;
        streamMetadata.height = nHeight;
        streamMetadata.averageFPS = avg_framerate;
        streamMetadata.bitrate = nBitrate;
        streamMetadata.duration = nDuration;
        streamMetadata.numFrames = nNumFramesfromStream;
        streamMetadata.codecName = avcodec_get_name(eVideoCodec);
        return streamMetadata;
    }

    void GetScannedStreamMetadata(std::promise<ScannedStreamMetadata>& scannedStreamMetadataPromise)
    {
        
        if (!is_seekable)
        {
            LOG(ERROR) << "This stream is not seekable. Not scanning for stream data\n";
            scannedStreamMetadataPromise.set_value(ScannedStreamMetadata());
            return;
        }
        AVPacket* avPacket;
        try
        {
            ScannedStreamMetadata scannedStreamMetadata = {};
            scannedStreamMetadata.width = nWidth;
            scannedStreamMetadata.height = nHeight;
            scannedStreamMetadata.averageFPS = avg_framerate;
            scannedStreamMetadata.bitrate = nBitrate;
            scannedStreamMetadata.duration = nDuration;
            scannedStreamMetadata.numFrames = 0;
            scannedStreamMetadata.codecName = avcodec_get_name(eVideoCodec);
            avPacket = av_packet_alloc();
            if (!avPacket) {
                PYNVVC_THROW_ERROR("AVPacket allocation failed.", CUDA_ERROR_NOT_SUPPORTED);
            }

            std::vector<PacketInfo> packetInfo;
            while (av_read_frame(fmtc, avPacket) >= 0) 
            {
                if (avPacket->flags & AV_PKT_FLAG_DISCARD ||
                    avPacket->stream_index != iVideoStream) 
                {
                    continue;
                }
                PacketInfo pi;
                pi.packetSize = avPacket->size;
                pi.isKeyFrame = false;
                pi.pts = avPacket->pts;
                pi.dts = avPacket->dts;
                if (avPacket->flags & AV_PKT_FLAG_KEY)
                {
                    pi.isKeyFrame = true;
                }
                packetInfo.push_back(pi);
                av_packet_unref(avPacket);
                ++scannedStreamMetadata.numFrames;
            }
            av_packet_free(&avPacket);
            avPacket = nullptr;
            
            // Sort in pts order
            std::sort(packetInfo.begin(), packetInfo.end(),
                [](const PacketInfo& pi1, const PacketInfo& pi2) {
                return pi1.pts < pi2.pts;
            });

            auto index = 0;
            for (auto& v : packetInfo)
            {
                scannedStreamMetadata.packetSize.push_back(v.packetSize);
                if (v.isKeyFrame)
                {
                    keyFrameIndices.push_back(index);
                }
                scannedStreamMetadata.pts.push_back(v.pts);
                scannedStreamMetadata.dts.push_back(v.dts);
                ++index;
            }
            scannedStreamMetadata.keyFrameIndices = keyFrameIndices;
            if (scannedStreamMetadata.duration == 0.0)
            {
                scannedStreamMetadata.duration = (scannedStreamMetadata.pts.back() - scannedStreamMetadata.pts.front()) * timeBase;

            }
            scannedStreamMetadataPromise.set_value(scannedStreamMetadata);
            // reset the demuxer pointer
            if (av_seek_frame(fmtc, -1, 0, AVSEEK_FLAG_BACKWARD) < 0)
            {
                PYNVVC_THROW_ERROR("Resetting the demuxer to original position failed.", CUDA_ERROR_NOT_SUPPORTED);
            }
        }
        catch(...)
        {
            if (avPacket != nullptr)
            {
                av_packet_free(&avPacket);
            }
            if (av_seek_frame(fmtc, -1, 0, AVSEEK_FLAG_BACKWARD) < 0)
            {
                PYNVVC_THROW_ERROR("Resetting the demuxer to original position failed.", CUDA_ERROR_NOT_SUPPORTED);
            }
            scannedStreamMetadataPromise.set_exception(std::current_exception());
        }       
    }


    static int ReadPacket(void *opaque, uint8_t *pBuf, int nBuf) {
        return ((DataProvider *)opaque)->GetData(pBuf, nBuf);
    }

    // Add a public method to access the container name if needed
    std::string GetContainerFormat() {
        return GetContainerName();
    }

};

#ifndef DEMUX_ONLY
inline cudaVideoCodec FFmpeg2NvCodecId(AVCodecID id) {
    switch (id) {
    case AV_CODEC_ID_MPEG1VIDEO : return cudaVideoCodec_MPEG1;
    case AV_CODEC_ID_MPEG2VIDEO : return cudaVideoCodec_MPEG2;
    case AV_CODEC_ID_MPEG4      : return cudaVideoCodec_MPEG4;
    case AV_CODEC_ID_WMV3       :
    case AV_CODEC_ID_VC1        : return cudaVideoCodec_VC1;
    case AV_CODEC_ID_H264       : return cudaVideoCodec_H264;
    case AV_CODEC_ID_HEVC       : return cudaVideoCodec_HEVC;
    case AV_CODEC_ID_VP8        : return cudaVideoCodec_VP8;
    case AV_CODEC_ID_VP9        : return cudaVideoCodec_VP9;
    case AV_CODEC_ID_MJPEG      : return cudaVideoCodec_JPEG;
    case AV_CODEC_ID_AV1        : return cudaVideoCodec_AV1;
    default                     : return cudaVideoCodec_NumCodecs;
    }
}
#endif

inline cudaVideoChromaFormat FFmpeg2NvChromaFormat(AVPixelFormat id) {
    switch(id) {
        case AV_PIX_FMT_GRAY10LE:
        case AV_PIX_FMT_GRAY8:
            return cudaVideoChromaFormat_Monochrome;
            
        case AV_PIX_FMT_YUV420P10LE:
        case AV_PIX_FMT_YUV420P12LE:
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
            return cudaVideoChromaFormat_420;

        case AV_PIX_FMT_YUV444P10LE:
        case AV_PIX_FMT_YUV444P12LE:
        case AV_PIX_FMT_YUV444P:
        case AV_PIX_FMT_YUVJ444P:       // NVDEC output is 420 for JPEG444 but return proper chromaformat to enduser
            return cudaVideoChromaFormat_444;

        case AV_PIX_FMT_YUV422P10LE:
        case AV_PIX_FMT_YUV422P12LE:
        case AV_PIX_FMT_YUV422P:
        case AV_PIX_FMT_YUVJ422P:       // NVDEC output is 420 for JPEG422 but return proper chromaformat to enduser
            return cudaVideoChromaFormat_422;

        default:
            // We should define cudaVideoChromaFormat_Unknown 
            return cudaVideoChromaFormat_420;
    }
}

