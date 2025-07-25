/*
 * This copyright notice applies to this file only
 *
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "SimpleTranscoder.hpp"
SimpleTranscoder::SimpleTranscoder()
{
    
}
SimpleTranscoder::SimpleTranscoder(
    const std::string& encSource,
    const std::string& muxedDst,
    uint32_t gpuId,
    size_t cudaContext,
    size_t cudaStream,
    std::map<std::string, std::string> kwargs
):numb(0)
{
    py::gil_scoped_release r;
    std::map<std::string, std::string> options = kwargs;
    mDemuxer.reset(new FFmpegDemuxer(encSource.c_str()));
    if (mDemuxer->GetAudioStreamId() < 0)
    {
        throw std::runtime_error("No audio stream found in the input file. Please provide an input file with audio stream.");
    }
    mSimpleDecoder.reset(new SimpleDecoder(encSource, gpuId, cudaContext, cudaStream,true));
    CUcontext cudacontext = (CUcontext)cudaContext;
    CUstream cudastream = (CUstream)cudaStream;
    NV_ENC_INITIALIZE_PARAMS params = { NV_ENC_INITIALIZE_PARAMS_VER };
    NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
    params.encodeConfig = &encodeConfig;
    NV_ENC_BUFFER_FORMAT bufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
    int width, height;
    switch (mSimpleDecoder->GetDecoderCommonInstance()->GetDecoder()->GetOutputFormat())
    {
        case cudaVideoSurfaceFormat_NV12:
        {
            bufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
        }
        break;        
        case cudaVideoSurfaceFormat_P016:
        {
            bufferFormat = NV_ENC_BUFFER_FORMAT_YUV420_10BIT;
        }
        break;
#if CHECK_API_VERSION(13,0)        
        case cudaVideoSurfaceFormat_NV16:
        {
            bufferFormat = NV_ENC_BUFFER_FORMAT_NV16;
        }
        break;

        case cudaVideoSurfaceFormat_P216:
        {
            bufferFormat = NV_ENC_BUFFER_FORMAT_P210;
        }
        break;
#endif
        case cudaVideoSurfaceFormat_YUV444:
        {
            bufferFormat = NV_ENC_BUFFER_FORMAT_YUV444;
        }
        break;
        case cudaVideoSurfaceFormat_YUV444_16Bit:
        {
            bufferFormat = NV_ENC_BUFFER_FORMAT_YUV444_10BIT;
        }
        break;

    }
    width = mSimpleDecoder->GetDecoderCommonInstance()->GetDemuxer()->GetWidth();
    height = mSimpleDecoder->GetDecoderCommonInstance()->GetDemuxer()->GetHeight();
    if (options["codec"] == "av1")
    {
        mEncoderCuda.reset(new NvEncoderCuda(
            mSimpleDecoder->GetDecoderCommonInstance()->GetCUContext(),
            mSimpleDecoder->GetDecoderCommonInstance()->GetCUStream(),
            width,
            height,
            bufferFormat,3,false,false,false,false
        ));
    }
    else
    {
        mEncoderCuda.reset(new NvEncoderCuda(
            mSimpleDecoder->GetDecoderCommonInstance()->GetCUContext(),
            mSimpleDecoder->GetDecoderCommonInstance()->GetCUStream(),
            width,
            height,
            bufferFormat
        ));
    }
    
    
    string res = "";
    res.append(std::to_string(width));
    res.append("x");
    res.append(std::to_string(height));
    options.insert(make_pair("s", res));
    string fps = std::to_string(mSimpleDecoder->GetDecoderCommonInstance()->GetDemuxer()->GetFrameRate());
    options.insert(make_pair("fps", fps));

    if (options.find("codec") == options.end()) {
        mCodec = "h264";
    }
    else {
        mCodec = options["codec"];
    }


    NvEncoderClInterface cliInterface(options);
    cliInterface.SetupInitParams(params, false, mEncoderCuda->GetApi(), mEncoderCuda->GetEncoder(), false);
    mEncoderCuda->CreateDefaultEncoderParams(&params, params.encodeGUID, params.presetGUID,
        params.tuningInfo);
    params.bufferFormat = bufferFormat;
    mEncoderCuda->CreateEncoder(&params);
    
    mEncSource = encSource;
    mMuxedDst = muxedDst;
    MEDIA_FORMAT mediaFormat = GetMediaFormat(mMuxedDst);
    
    numb = params.encodeConfig->frameIntervalP;
    if (numb >= 1)
    {
        numb = numb - 1;
    }
    py::gil_scoped_acquire a;
}

void SimpleTranscoder::TranscodeWithMux()
{
    
    int nBytes = 0, nFrameReturned = 0, nFrame = 0, isVideoPacket = 0, streamIndex = -1;
    int64_t pts = 0 , dts = 0;
    uint64_t duration = 0;
    uint64_t pos = 0;
    bool keyFrame = false;
    uint8_t* pData = NULL, * pFrame = NULL;
    bool bOut10 = false;
    CUdeviceptr dpFrame = 0;
    NV_ENC_PIC_PARAMS picParams = { NV_ENC_PIC_PARAMS_VER };
    std::vector<int64_t> vDts, vPts;
    std::vector<unsigned char> vSeqParams;

    mEncoderCuda->GetSequenceParams(vSeqParams);
    FFmpegDemuxer* demuxer = new FFmpegDemuxer(mEncSource.c_str());
    MEDIA_FORMAT mediaFormat = GetMediaFormat(mMuxedDst);
    std::unique_ptr<FFmpegMuxer> mMuxer;
    mMuxer.reset(new FFmpegMuxer(mMuxedDst.c_str(),
        mediaFormat,
        demuxer->GetAVFormatContext(),
        mSimpleDecoder->GetDecoderCommonInstance()->GetDemuxer()->GetVideoCodec(),
        mSimpleDecoder->GetDecoderCommonInstance()->GetDemuxer()->GetWidth(),
        mSimpleDecoder->GetDecoderCommonInstance()->GetDemuxer()->GetHeight(),
        vSeqParams.data(), 
        vSeqParams.size()));
    try
    {
        do {
            mSimpleDecoder->GetDecoderCommonInstance()->GetDemuxer()->DemuxNoSkipAudio(&pData, &nBytes, pts, dts, duration, pos, keyFrame, &isVideoPacket, &streamIndex);
            if (isVideoPacket == 0)
            {
                mMuxer->Mux(pData, nBytes, pts, dts, duration,streamIndex);
                continue;
            }
            
            nFrameReturned = mSimpleDecoder->GetDecoderCommonInstance()->GetDecoder()->Decode(pData, nBytes, 0, pts);
            for (int i = 0; i < nFrameReturned; i++)
            {
                pFrame = mSimpleDecoder->GetDecoderCommonInstance()->GetDecoder()->GetFrame(&pts);
                vPts.push_back(pts);
                vDts.push_back(pts);
                

                std::vector<NvEncOutputFrame> vPacket;
                const NvEncInputFrame* encoderInputFrame = mEncoderCuda->GetNextInputFrame();

                picParams.inputTimeStamp = pts;

                NvEncoderCuda::CopyToDeviceFrame(mSimpleDecoder->GetDecoderCommonInstance()->GetCUContext(),
                    pFrame,
                    mSimpleDecoder->GetDecoderCommonInstance()->GetDecoder()->GetDeviceFramePitch(),
                    (CUdeviceptr)encoderInputFrame->inputPtr,
                    encoderInputFrame->pitch,
                    mEncoderCuda->GetEncodeWidth(),
                    mEncoderCuda->GetEncodeHeight(),
                    CU_MEMORYTYPE_DEVICE,
                    encoderInputFrame->bufferFormat,
                    encoderInputFrame->chromaOffsets,
                    encoderInputFrame->numChromaPlanes);
                mEncoderCuda->EncodeFrame(vPacket, &picParams);
          
                for (int i = 0; i < (int)vPacket.size(); i++)
                {
                    mMuxer->Mux(reinterpret_cast<unsigned char*>(vPacket[i].frame.data()), vPacket[i].frame.size(), vDts[vPacket[i].timeStamp], vPts.front(), duration, streamIndex, vPacket[i].pictureType == NV_ENC_PIC_TYPE_IDR, numb);
                    vPts.erase(vPts.begin());
                    nFrame++;
                }
            }
        } while (nBytes);

        std::vector<NvEncOutputFrame> vPacket;
        mEncoderCuda->EndEncode(vPacket);
        for (int i = 0; i < (int)vPacket.size(); i++)
        {
            mMuxer->Mux(reinterpret_cast<unsigned char*>(vPacket[i].frame.data()), vPacket[i].frame.size(), vDts[vPacket[i].timeStamp], vPts.front(), duration, streamIndex);
            vPts.erase(vPts.begin());
            nFrame++;
        }
    }
    catch (const std::exception& ex)
    {
        LOG(ERROR) << ex.what();
    }
}

/*
    Basic idea behing cutting segment, re-encoding that segment and muxing it is as follows:
    Function takes in start and end time stamps in seconds.
    These represent the time where the cut should begin and end. Frame corresponding to start timestamp is accessed and encoded as IDR.
    We use SimpleDecoder's Indexing API to fetch this frame. SimpleDecoder also stores the list of frames just after the first frame in pending queue.
    We encode this queue separately.Then native decoder loop starts, demuxer gives audio and video packets on each call.
    video packets are re-encoded and audio packets are copied as it is to muxed output.
    For e.g. 
    Input is like IBBBPBBBPBBBP.....
    After decode, output is IPBBBPBBB (Display order)
    after re-encode output is IBBBPBBBPBBBP (Decode order)
    For muxing , we encode in Decode order but PTS are arranged in Display order

*/

void SimpleTranscoder::SegmentedTranscodeWithMux(float start_ts, float end_ts)
{
    
    int nBytes = 0, nAudioBytes = 0, nFrameReturned = 0, nFrame = 0, isVideoPacket = 0, streamIndex = -1, isVideoPacket0 = 0, streamIndex0 = -1;
    int64_t pts = 0, dts = 0;
    uint64_t duration = 0;
    uint64_t pos = 0;
    int64_t audio_pts = 0, audio_dts = 0;
    uint64_t audio_duration = 0;
    uint64_t audio_pos = 0;
    bool audio_keyFrame = false;
    bool keyFrame = false;
    uint8_t* pData = NULL, * pFrame = NULL;
    uint8_t* pAudioData = NULL, * pAudioFrame = NULL;
    bool bOut10 = false;
    CUdeviceptr dpFrame = 0;
    std::vector<int64_t> vDts, vPts;
    int64_t prev_vpts = -1, prev_vdts = -1;
    int64_t apts = 0, adts = 0;
    std::vector<unsigned char> vSeqParams;

    try
    {
        std::filesystem::path file = mMuxedDst;
        std::unique_ptr<FFmpegMuxer> pMuxer;
        std::string updated_filename("");
        updated_filename.append(file.replace_extension().string());
        file = mMuxedDst;
        updated_filename.append("_");
        updated_filename.append(std::to_string(start_ts));
        updated_filename.append("_");
        updated_filename.append(std::to_string(end_ts));
        updated_filename.append(file.extension().string());
        mEncoderCuda->GetSequenceParams(vSeqParams);
        MEDIA_FORMAT mediaFormat = GetMediaFormat(mMuxedDst);
        

        mEncoderCuda->ResetCounter();
        double fps = mSimpleDecoder->GetDecoderCommonInstance()->GetDemuxer()->GetFrameRate();
        double stream_duration = mSimpleDecoder->GetDecoderCommonInstance()->GetDemuxer()->GetStreamMetadata().duration;
        int frames = fps * stream_duration;

        std::vector<NvEncOutputFrame> vPacket;

        
        int start_index = llround(start_ts * frames / stream_duration);
        int end_index = llround(end_ts * frames / stream_duration);
        std::vector<uint32_t> idxs;
        idxs.push_back(start_index);
        int64_t videopkt_duration = frames / stream_duration;
        
        std::vector<DecodedFrame> first = mSimpleDecoder->GetBatchFramesByIndex(idxs);
        //we will be missing audio packets from the "start_index" to actual position of demuxer when the first decoded frame got returned
        py::gil_scoped_release r;
        AVCodecID codecID;
        if (mCodec == "h264")
        {
            codecID = AV_CODEC_ID_H264;
        }
        else if (mCodec == "hevc")
        {
            codecID = AV_CODEC_ID_HEVC;
        }
        else if (mCodec == "av1")
        {
            codecID = AV_CODEC_ID_AV1;
        }
        pMuxer.reset(new FFmpegMuxer(updated_filename.c_str(),
            mediaFormat,
            mSimpleDecoder->GetDecoderCommonInstance()->GetDemuxer()->GetAVFormatContext(),
            codecID,
            mSimpleDecoder->GetDecoderCommonInstance()->GetDemuxer()->GetWidth(),
            mSimpleDecoder->GetDecoderCommonInstance()->GetDemuxer()->GetHeight(),
            vSeqParams.data(),
            vSeqParams.size()));

        int64_t video_stream_pts_offset = first[0].timestamp;

        //PTS and DTS needs to be always in increasing order
        auto MuxVideoPacketsWithAdjPTS = [&](int64_t _duration) {

            for (int i = 0; i < (int)vPacket.size(); i++)
            {
                int64_t adj_pts = vDts[vPacket[i].timeStamp] - video_stream_pts_offset ;
                //encoder returns the position of frame before encod, i.e display order, since vDts has PTS in display order, we use that position to get correct PTS
                int64_t adj_dts = vPts.front() - video_stream_pts_offset;
                prev_vdts = adj_dts;
                pMuxer->Mux(reinterpret_cast<unsigned char*>(vPacket[i].frame.data()), vPacket[i].frame.size(), adj_pts, adj_dts, _duration,streamIndex, vPacket[i].pictureType == NV_ENC_PIC_TYPE_IDR, numb);
                vPts.erase(vPts.begin());
            }

        };

        auto EncodeVideoPackets = [&](NV_ENC_PIC_PARAMS& picParam, CUdeviceptr SrcDevice) {

            const NvEncInputFrame* encoderInputFrame = mEncoderCuda->GetNextInputFrame();
            NvEncoderCuda::CopyToDeviceFrame(mSimpleDecoder->GetDecoderCommonInstance()->GetCUContext(),
                (void*)SrcDevice,
                mSimpleDecoder->GetDecoderCommonInstance()->GetDecoder()->GetDeviceFramePitch(),
                (CUdeviceptr)encoderInputFrame->inputPtr,
                encoderInputFrame->pitch,
                mEncoderCuda->GetEncodeWidth(),
                mEncoderCuda->GetEncodeHeight(),
                CU_MEMORYTYPE_DEVICE,
                encoderInputFrame->bufferFormat,
                encoderInputFrame->chromaOffsets,
                encoderInputFrame->numChromaPlanes);
            mEncoderCuda->EncodeFrame(vPacket, &picParam);

        };
        

        //encode the IDR here
        NV_ENC_PIC_PARAMS picParam = { 0 };
        picParam.inputTimeStamp = first[0].timestamp;
        picParam.encodePicFlags = NV_ENC_PIC_FLAG_OUTPUT_SPSPPS | NV_ENC_PIC_FLAG_FORCEIDR;
        EncodeVideoPackets(picParam, first[0].views[0].data);
        
        vPts.push_back(first[0].timestamp);
        vDts.push_back(first[0].timestamp);
        
        //encode the frames from pending queue just after the IDR
        std::vector<DecodedFrame> pending = mSimpleDecoder->GetDecoderCommonInstance()->GetPtrToSeekUtils()->GetPendingFrames();

        for (int i = 0; i < pending.size(); i++)
        {
            NV_ENC_PIC_PARAMS picParam = { 0 };
            picParam.inputTimeStamp = pending[i].timestamp;
            EncodeVideoPackets(picParam, pending[0].views[0].data);
            vPts.push_back(pending[i].timestamp);
            vDts.push_back(pending[i].timestamp);
        }

        //mux the video packets
        MuxVideoPacketsWithAdjPTS(videopkt_duration);

        int framesize = mSimpleDecoder->GetDecoderCommonInstance()->GetDecoder()->GetFrameSize();
        int audio_frames_count = mDemuxer->GetAudioStream()->nb_frames;
        double audio_stream_duration = av_q2d(mDemuxer->GetAudioStream()->time_base) * mDemuxer->GetAudioStream()->duration;
        int audio_start_index = llround(start_ts * audio_frames_count / audio_stream_duration);
        int target_audio_pts = audio_start_index * (mDemuxer->GetAudioStream()->duration / audio_frames_count);
        int duration0 = mDemuxer->GetAudioStream()->duration / audio_frames_count;
        //plan is to reset demuxer to using any frame index, mux all packets till it matches dts of decoded frame
        
        bool bTest = false;
        vPacket.clear();
        float end_ts_audio = (stream_duration / frames) * (start_index + pending.size() + 1) ;// int start_index = llround(start_ts * frames / stream_duration);
        
        do {
            
            mSimpleDecoder->GetDecoderCommonInstance()->GetDemuxer()->DemuxNoSkipAudio(&pData, &nBytes, pts, dts, duration, pos, keyFrame, &isVideoPacket, &streamIndex);

            if (isVideoPacket == 0)
            {
                if (!bTest)
                {
                    //here we start native decode and copy all audio packets from IDR till the last pendig frame
                    mDemuxer->SeekAudioStream(start_index, audio_start_index);

                    do {
                        mDemuxer->DemuxNoSkipAudio(&pAudioData, &nAudioBytes, audio_pts, audio_dts, audio_duration, audio_pos, audio_keyFrame, &isVideoPacket, &streamIndex);
                        if (isVideoPacket == 0)
                        {
                            int pkt_index = audio_pts / duration0;
                            double pkt_duration = pkt_index * (audio_stream_duration / audio_frames_count);
                            
                            if (pkt_duration < start_ts)
                            {
                                continue;
                            }
                            else if (pkt_duration >= end_ts_audio)
                            {
                                break;
                            }
                            pMuxer->Mux(pAudioData, nAudioBytes, apts, adts, audio_duration,streamIndex);
                            apts = apts + duration0;
                            adts = adts + duration0;
                        }

                    } while (1);
                    bTest = true;
                }
                
                pMuxer->Mux(pData, nBytes, apts, adts, audio_duration, streamIndex);
                apts = apts + duration0;
                adts = adts + duration0;
                continue;
            }
            else
            {
                nFrame++;
            }
            //decode, re-encode and then mux
            nFrameReturned = mSimpleDecoder->GetDecoderCommonInstance()->GetDecoder()->Decode(pData, nBytes, 0, pts);
          
            for (int i = 0; i < nFrameReturned; i++)
            {
                pFrame = mSimpleDecoder->GetDecoderCommonInstance()->GetDecoder()->GetFrame(&pts);
               
                vPts.push_back(pts);
                vDts.push_back(pts);

                NV_ENC_PIC_PARAMS picParam = { 0 };
                picParam.inputTimeStamp = pts;
                EncodeVideoPackets(picParam, (CUdeviceptr)pFrame);
                MuxVideoPacketsWithAdjPTS(duration);
                vPacket.clear();
            }
            if (nFrame >= (end_index - start_index))
            {
                break;
            }
        } while (nBytes);

        vPacket.clear();
        //flush the encoder queue and mux separately
        mEncoderCuda->EndEncode(vPacket);
        MuxVideoPacketsWithAdjPTS(videopkt_duration);
            
    }
    catch (const std::exception& ex)
    {
        LOG(ERROR) << ex.what();
    }
    py::gil_scoped_acquire a;
}

SimpleTranscoder::~SimpleTranscoder()
{
    py::gil_scoped_release r;
    mEncoderCuda.reset();
    py::gil_scoped_acquire a;
}

