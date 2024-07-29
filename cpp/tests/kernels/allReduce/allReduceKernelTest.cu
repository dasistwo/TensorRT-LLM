/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/kernels/customAllReduceKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/ipcUtils.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace mpi = tensorrt_llm::mpi;
namespace tr = tensorrt_llm::runtime;
using namespace tensorrt_llm::kernels;

void simple_assert(bool flag)
{
    if (!flag)
    {
        throw std::runtime_error("assert failed");
    }
}

struct CudaBuffer
{
    void* _data;
    int _size;

    CudaBuffer(int size_in_bytes)
        : _size(size_in_bytes)
    {
        cudaMalloc(&_data, _size);
    }

    template <typename T = void>
    T* data()
    {
        return reinterpret_cast<T*>(_data);
    }

    void copy_to(void* dst)
    {
        cudaMemcpy(dst, _data, _size, cudaMemcpyDeviceToHost);
    }

    void copy_from(void* src)
    {
        cudaMemcpy(_data, src, _size, cudaMemcpyHostToDevice);
    }

    ~CudaBuffer()
    {
        cudaFree(_data);
    }
};

template <typename T>
float compare(int rank, void* _pa, void* _pb, int size, float scale)
{
    auto pa = reinterpret_cast<T*>(_pa);
    auto pb = reinterpret_cast<T*>(_pb);
    float max_diff = 0.f, tot_diff = 0.f;
    float max_val = 0.f;
    int diff_cnt = 0;
    float threshold = 1e-7;
    for (int n = 0; n < size; ++n)
    {
        float va = static_cast<float>(pa[n]);
        float vb = static_cast<float>(pb[n]);
        max_val = std::max(max_val, vb);
        float diff = std::abs(va - vb);
        if (diff > threshold)
        {
            max_diff = std::max(max_diff, diff);
            tot_diff += diff;
            ++diff_cnt;
        }
    }
    float diff_thres = max_val * scale;
#if defined(ENABLE_BF16)
    if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        // bfloat16 has fewer mantissa digits than float16(10 bits for fp16 but only 7 bits for bf16), so the cumulative
        // error will be larger.
        diff_thres *= 3.f;
    }
    else
#endif
    {
        diff_thres *= 1.5f;
    }
    if (rank == 0)
    {
        printf("rank %d, max diff %f (diff threshold %f), avg diff %f, diff cnt %d/%d\n", rank, max_diff, diff_thres,
            tot_diff / std::max(diff_cnt, 1), diff_cnt, size);
    }
    return max_diff <= diff_thres;
}

template <typename T1, typename T2>
void random_fill(std::vector<T1>& vec, T2 minv, T2 maxv)
{
    std::mt19937 gen(20240410);
    std::uniform_real_distribution<float> dis(static_cast<float>(minv), static_cast<float>(maxv));
    for (auto& v : vec)
    {
        v = static_cast<T1>(dis(gen));
    }
}

std::string ar_info(AllReduceStrategyType runtime_strategy, AllReduceStrategyConfig config)
{
    std::string info;
    if (runtime_strategy == AllReduceStrategyType::ONESHOT)
    {
        info += "ONESHOT ";
    }
    else if (runtime_strategy == AllReduceStrategyType::TWOSHOT)
    {
        info += "TWOSHOT ";
    }
    if (config == AllReduceStrategyConfig::USE_MEMCPY)
    {
        info += "USE_MEMCPY";
    }
    else if (config == AllReduceStrategyConfig::PUSH_MODE)
    {
        info += "PUSH_MODE";
    }
    else
    {
        info += "NONE";
    }
    return info;
}

bool test(int token_num, int hidden_size, bool has_bias, bool has_affine, int warmup, int iter,
    AllReduceStrategyType runtime_strategy = AllReduceStrategyType::ONESHOT,
    AllReduceStrategyConfig config = AllReduceStrategyConfig(0), AllReduceFusionOp fusion_op = AllReduceFusionOp::NONE)
{
    std::srand(20240603);
    int message_size = token_num * hidden_size;
    int buffer_size = sizeof(half) * message_size;
    CudaBuffer in(buffer_size), out(buffer_size), residual(buffer_size), weight(hidden_size * sizeof(half)),
        inter(buffer_size), bias(hidden_size * sizeof(half));
    std::vector<half> input_buffer(message_size);
    std::vector<half> residual_buffer(message_size);
    std::vector<half> weight_buffer(hidden_size);
    std::vector<half> bias_buffer(hidden_size);
    std::vector<half> inter_buffer(message_size);
    std::vector<half> output_buffer(message_size);
    float eps = 1e-6;
    random_fill(residual_buffer, -1, 1);
    random_fill(weight_buffer, -1, 1);
    random_fill(bias_buffer, -1, 1);
    residual.copy_from(residual_buffer.data());
    weight.copy_from(weight_buffer.data());
    bias.copy_from(bias_buffer.data());
    auto& comm = mpi::MpiComm::world();
    auto world_size = comm.getSize();
    auto rank = comm.getRank();
    if (rank == 0)
    {
        std::string info = ar_info(runtime_strategy, config);
        if (fusion_op == AllReduceFusionOp::RESIDUAL_RMS_NORM)
        {
            printf(
                "Custom All Reduce with Residual Add and RMS Norm, %s, message size %d(token num %d, hidden size %d), "
                "has bias %d, has affine %d\n",
                info.c_str(), message_size, token_num, hidden_size, static_cast<int>(has_bias),
                static_cast<int>(has_affine));
        }
        else
        {
            printf("Custom All Reduce, %s, message size %d(token num %d, hidden size %d), has bias %d, has affine %d\n",
                info.c_str(), message_size, token_num, hidden_size, static_cast<int>(has_bias),
                static_cast<int>(has_affine));
        }
    }
    random_fill(input_buffer, -1 / world_size, 1 / world_size);
    in.copy_from(input_buffer.data());
    cudaSetDevice(rank);

    tr::WorldConfig world_config(world_size, 1, rank, world_size);
    auto p_s = std::make_shared<tr::CudaStream>();
    tr::BufferManager buf_mgr(p_s);
    tr::AllReduceBuffers buffers(1, 1, token_num, hidden_size, buf_mgr, world_config);

    AllReduceParams params;
    for (int i = 0; i < world_size; ++i)
    {
        params.peer_comm_buffer_ptrs[i] = buffers.mIpcMemoryHandles[0].getCommPtrs()[i];
    }
    for (int i = 0; i < world_size; ++i)
    {
        params.peer_barrier_ptrs_in[i] = reinterpret_cast<uint32_t*>(buffers.mIpcMemoryHandles[2].getCommPtrs()[i]);
    }
    for (int i = 0; i < world_size; ++i)
    {
        params.peer_barrier_ptrs_out[i] = reinterpret_cast<uint32_t*>(buffers.mIpcMemoryHandles[3].getCommPtrs()[i]);
    }
    params.barrier_flag = 0;
    params.ranks_per_node = world_size;
    params.local_rank = rank;
    params.local_output_buffer_ptr = out.data();
    params.local_input_buffer_ptr = in.data();
    params.elts_total = message_size;
    params.fusion_params.bias_buffer = has_bias ? bias.data() : nullptr;
    params.fusion_params.residual_buffer = residual.data();
    params.fusion_params.hidden_size = hidden_size;
    params.fusion_params.weight_buffer = has_affine ? weight.data() : nullptr;
    params.fusion_params.eps = eps;
    params.fusion_params.intermediate_buffer = inter.data();

    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    for (int i = 0; i < warmup; ++i)
    {
        params.barrier_flag += 1;
        customAllReduce(params, nvinfer1::DataType::kHALF, runtime_strategy, config, fusion_op, s);
    }
    cudaEventRecord(begin, s);
    for (int i = 0; i < iter; ++i)
    {
        params.barrier_flag += 1;
        customAllReduce(params, nvinfer1::DataType::kHALF, runtime_strategy, config, fusion_op, s);
    }
    cudaEventRecord(end, s);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, begin, end);
    time /= iter;
    std::vector<half> ref(message_size);
    for (int i = 0; i < ref.size(); ++i)
    {
        ref[i] = static_cast<float>(input_buffer[i]) * world_size;
    }
    out.copy_to(output_buffer.data());
    bool pass = true;
    if (fusion_op == AllReduceFusionOp::RESIDUAL_RMS_NORM)
    {
        inter.copy_to(inter_buffer.data());
        for (int i = 0; i < ref.size(); ++i)
        {
            ref[i] = static_cast<float>(ref[i])
                + (has_bias ? static_cast<float>(residual_buffer[i % hidden_size]) : 0.f)
                + static_cast<float>(residual_buffer[i]);
        }
        pass = pass && compare<half>(rank, inter_buffer.data(), ref.data(), ref.size(), 1e-2);
        for (int i = 0; i < token_num; ++i)
        {
            float sum = 0.f;
            for (int j = 0; j < hidden_size; ++j)
            {
                float v = static_cast<float>(ref[i * hidden_size + j]);
                sum += v * v;
            }
            float denom = std::sqrt((sum / hidden_size) + eps);
            for (int j = 0; j < hidden_size; ++j)
            {
                float v = static_cast<float>(ref[i * hidden_size + j]);
                ref[i * hidden_size + j] = v / denom * (has_affine ? static_cast<float>(weight_buffer[j]) : 1.f);
            }
        }
    }
    pass = pass && compare<half>(rank, output_buffer.data(), ref.data(), ref.size(), 1e-2);
    if (rank == 0)
        printf("duration %6.6fms\n", time);
    if (rank == 0 && pass)
    {
        printf("\033[32mPASS\033[0m\n");
    }
    else if (rank == 0 && !pass)
    {
        printf("\033[31mFAILED\033[0m\n");
    }
    cudaStreamDestroy(s);
    return pass;
}

TEST(Kernel, AllReduce)
{
    auto& comm = mpi::MpiComm::world();
    auto world_size = comm.getSize();
    if (world_size % 2)
        return;

    int warmup = 100, iter = 100;
    // clang-format off
    std::vector<AllReduceStrategyConfig> configs{
        AllReduceStrategyConfig(0),
        AllReduceStrategyConfig::USE_MEMCPY,
        AllReduceStrategyConfig::PUSH_MODE
    };
    std::vector<AllReduceFusionOp> ops{
        // AllReduceFusionOp::NONE,
        AllReduceFusionOp::RESIDUAL_RMS_NORM
    };
    // clang-format on
    bool pass = true;
    for (auto config : configs)
    {
        for (auto op : ops)
        {
            for (auto has_bias : {false, true})
            {
                for (auto has_affine : {false, true})
                {
                    pass = pass
                        && test(
                            1, 4096, has_bias, has_affine, warmup, iter, AllReduceStrategyType::ONESHOT, config, op);
                    pass = pass
                        && test(
                            1, 8192, has_bias, has_affine, warmup, iter, AllReduceStrategyType::ONESHOT, config, op);
                    pass = pass
                        && test(
                            10, 4096, has_bias, has_affine, warmup, iter, AllReduceStrategyType::ONESHOT, config, op);
                    pass = pass
                        && test(
                            10, 8192, has_bias, has_affine, warmup, iter, AllReduceStrategyType::ONESHOT, config, op);
                    pass = pass
                        && test(
                            1000, 4096, has_bias, has_affine, warmup, iter, AllReduceStrategyType::TWOSHOT, config, op);
                    pass = pass
                        && test(
                            1000, 8192, has_bias, has_affine, warmup, iter, AllReduceStrategyType::TWOSHOT, config, op);
                }
            }
        }
    }
    EXPECT_TRUE(pass);
}

TEST(Kernel, AllReduceOneShot)
{
    auto& comm = mpi::MpiComm::world();
    auto world_size = comm.getSize();
    if (world_size % 2)
        return;

    int warmup = 100, iter = 100;
    std::vector<int> candidate_bs{1, 2, 4, 8, 16, 32, 64, 128};
    std::vector<int> candidate_hidden{4096, 8192, 12288, 16384};
    bool pass = true;
    for (auto bs : candidate_bs)
    {
        for (auto hidden : candidate_hidden)
        {
            pass = pass
                && test(bs, hidden, false, true, warmup, iter, AllReduceStrategyType::ONESHOT,
                    AllReduceStrategyConfig(0), AllReduceFusionOp::RESIDUAL_RMS_NORM);
            pass = pass
                && test(bs, hidden, true, true, warmup, iter, AllReduceStrategyType::ONESHOT,
                    AllReduceStrategyConfig(0), AllReduceFusionOp::RESIDUAL_RMS_NORM);
            pass = pass
                && test(bs, hidden, false, false, warmup, iter, AllReduceStrategyType::ONESHOT,
                    AllReduceStrategyConfig(0), AllReduceFusionOp::RESIDUAL_RMS_NORM);
            pass = pass
                && test(bs, hidden, true, false, warmup, iter, AllReduceStrategyType::ONESHOT,
                    AllReduceStrategyConfig(0), AllReduceFusionOp::RESIDUAL_RMS_NORM);
        }
    }
    EXPECT_TRUE(pass);
}
