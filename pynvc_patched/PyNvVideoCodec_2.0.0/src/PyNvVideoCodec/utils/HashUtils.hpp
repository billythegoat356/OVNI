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

#include <tuple>
#include <functional>

// Replicating boost::hash_combine.
// Refer: https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3876.pdf
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}


template <typename Tuple, std::size_t Index = 0>
inline typename std::enable_if<Index == std::tuple_size<Tuple>::value, void>::type
hash_tuple(std::size_t& seed, const Tuple& tuple) {}

template <typename Tuple, std::size_t Index = 0>
inline typename std::enable_if<Index < std::tuple_size<Tuple>::value, void>::type
hash_tuple(std::size_t& seed, const Tuple& tuple)
{
    hash_combine(seed, std::get<Index>(tuple));
    hash_tuple<Tuple, Index + 1>(seed, tuple);
}

struct TupleHash
{
    template <typename... Args>
    std::size_t operator()(const std::tuple<Args...>& t) const
    {
        std::size_t seed = 0;
        hash_tuple(seed, t);
        return seed;
    }
};
