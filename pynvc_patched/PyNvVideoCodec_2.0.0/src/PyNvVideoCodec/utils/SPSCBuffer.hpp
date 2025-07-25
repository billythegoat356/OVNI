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


#include <atomic>
#include <condition_variable>
#include <mutex>
#include <sstream>
#include <vector>

#include "Logger.h"

template<typename T>
class SPSCBuffer {
public:
    SPSCBuffer() : mBuffer(0), mHead(0), mTail(0), mCapacity(0), mCount(0), mDrain(false) {}
    SPSCBuffer(size_t size) : mBuffer(size), mHead(0), mTail(0), mCapacity(size), mCount(0), mDrain(false) {}

    // Push an entry into the mBuffer, wait if necessary
    bool PushEntry(const T& entry) {
        std::unique_lock<std::mutex> lock(mMutx);
        // Wait until there is space available in the mBuffer
        mCV.wait(lock, [this]() {
            return mCount < mCapacity;
        });

        mBuffer[mHead] = entry;
        mHead = (mHead + 1) % mCapacity;
        ++mCount;

        // Notify app thread that new data is available
        mCV.notify_one();
        return true;
    }

    void PushDone()
    {
        std::unique_lock<std::mutex> lock(mMutx);
        LOG(DEBUG) << "Push completed\n";
        mDrain = true;
        mCV.notify_one();
    }

    // Pop exactly batchSize entries, wait if necessary
    std::vector<T> PopEntries(size_t batchSize) {
        std::unique_lock<std::mutex> lock(mMutx);
        if (batchSize > mCapacity) 
        {
            // LOG error or throw
            std::stringstream ss;
            ss << "Got invalid value for batchSize. Got "
               << batchSize
               << " Max allowed value is "
               << mCapacity;
            throw std::runtime_error(ss.str());
            return {};
        }

        if (batchSize == 0)
        {
            batchSize = mCount;
        }

        // Wait until there are at least batchSize entries available
        mCV.wait(lock, [this, &batchSize]() {
            return ((mCount >= batchSize) || mDrain);
        });

        // If we are done decoding and the requested size is greater than available number of
        // elements then set the requested size to available number and drain them all.
        // Next call to pop will return zero elements and this can be used as a signal
        // to end calls to get_batch_frames
        if (mDrain && mCount < batchSize)
        {
            LOG(DEBUG) << "Drain with batchSize: " << batchSize << " count: " << mCount << "\n";
            batchSize = mCount;
        }

        std::vector<T> entries;
        entries.reserve(batchSize);

        // Pop exactly batchSize entries from the mBuffer
        for (size_t i = 0; i < batchSize; ++i) {
            entries.push_back(mBuffer[mTail]);
            mTail = (mTail + 1) % mCapacity;
            --mCount;
        }
        // Notify decoder thread that space is now available
        mCV.notify_one();

        return entries;
    }

    void Clear()
    {
        mHead = mTail = mCount = 0;
        mDrain = false;
    }

private:
    std::vector<T> mBuffer;
    // Push at mHead
    uint32_t mHead;
    // Pop at tail
    uint32_t mTail;
    // Total size of the mBuffer
    const uint32_t mCapacity;
    // Current number of elements in mBuffer
    uint32_t mCount;
    // A signal to indicate push is done and the buffer can be drained on next pop call
    bool mDrain;
    // Mutex for sync
    std::mutex mMutx;
    // Condition variable for waiting and notification 
    std::condition_variable mCV;
};