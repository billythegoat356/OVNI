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

#pragma once

#include <iostream>
#include <list>
#include <optional>
#include <type_traits>
#include <unordered_map>

#include "cuviddec.h"
#include "HashUtils.hpp"


// std::hash specialization for cudaVideoCodec and cudaVideoChromaFormat
namespace std 
{
    template <>
    struct hash<cudaVideoCodec> 
    {
        size_t operator()(const cudaVideoCodec& e) const
        {
            return std::hash<std::underlying_type<cudaVideoCodec>::type>()(
                static_cast<std::underlying_type<cudaVideoCodec>::type>(e));
        }
    };
}

namespace std 
{
    template <>
    struct hash<cudaVideoChromaFormat> 
    {
        size_t operator()(const cudaVideoChromaFormat& e) const 
        {
            return std::hash<std::underlying_type<cudaVideoChromaFormat>::type>()(
                static_cast<std::underlying_type<cudaVideoChromaFormat>::type>(e));
        }
    };
}


// Decoder caching with LRU policy
template<typename Key, typename Value>
class DecoderCache {
public:
    DecoderCache(uint32_t capacity = 4) : mCapacity(capacity) 
    {
        if (capacity < 1)
        {
            LOG(INFO) << "Cache size needs to be atleast 1. Received: " << capacity << "\n";
            mCapacity = 1;
            LOG(INFO) << "Setting cache size to: " << mCapacity << "\n";
        }
    }

    std::optional<Value> GetDecoder(const Key& key)
    {
        auto it = mCacheMap.find(key);
        if (it == mCacheMap.end())
        {
            return std::nullopt;
        }
        // Move the accessed item to the front of the list
        mCacheItems.splice(mCacheItems.begin(), mCacheItems, it->second);
        return it->second->second;
    }

    std::optional<Value> PushDecoder(const Key& key, const Value& value)
    {
        std::optional<Value> decoder { std::nullopt };
        auto it = mCacheMap.find(key);
        if (it != mCacheMap.end())
        {
            // Update the value and move to front
            it->second->second = value;
            mCacheItems.splice(mCacheItems.begin(), mCacheItems, it->second);
        }
        else
        {
            if (mCacheItems.size() == mCapacity)
            {
                // Remove the least recently used item
                auto last = mCacheItems.back();
                mCacheMap.erase(last.first);
                decoder = mCacheItems.back().second;
                mCacheItems.pop_back();
            }
            // Insert the new item at the front
            mCacheItems.emplace_front(key, value);
            mCacheMap[key] = mCacheItems.begin();
        }
        return decoder;
    }

    std::optional<Value> RemoveElement()
    {
        if (mCacheItems.empty())
        {
            return std::nullopt;
        }
        auto last = mCacheItems.back();
        mCacheMap.erase(last.first);
        auto decoder = mCacheItems.back().second;
        mCacheItems.pop_back();

        return decoder;
    }

private:
    int mCapacity;
    using KeyValuePair = std::pair<Key, Value>;
    std::list<KeyValuePair> mCacheItems;
    std::unordered_map<Key, typename std::list<KeyValuePair>::iterator, TupleHash> mCacheMap;
};