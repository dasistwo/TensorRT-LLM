/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Template for a pipelined StreamK GEMM kernel.
    based on cutlass_extensions/gemm/kernel/fpA_intB_gemm.h
    and cutlass/include/cutlass/gemm/kernel/gemm_universal_streamk.h
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/barrier.h"
#include "cutlass/block_striped.h"

#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass
{
namespace gemm
{
namespace kernel
{

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma_,          ///! Threadblock-scoped matrix multiply-accumulate
    typename Epilogue_,           ///! Epilogue
    typename ThreadblockSwizzle_, ///! Threadblock swizzling function
    typename KernelArch ///! The Architecture this kernel is compiled for. Used since SIMT kernels lose top-level
                         /// arch.
    >
struct GemmFpAIntBStreamK
{

    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using EpilogueOutputOp = typename Epilogue::OutputOp;
    using ThreadblockSwizzle = ThreadblockSwizzle_;

    using ElementA = typename Mma::IteratorA::Element;
    using LayoutA = typename Mma::IteratorA::Layout;
    using ElementB = typename Mma::IteratorB::Element;
    using LayoutB = typename Mma::IteratorB::Element;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC = typename Epilogue::OutputTileIterator::Layout;
    using ElementScale = ElementC;

    /// The per-thread tile of raw accumulators
    using AccumulatorTile = typename Mma::FragmentC;

    static ComplexTransform const kTransformA = Mma::kTransformA;
    static ComplexTransform const kTransformB = Mma::kTransformA;

    // Type definitions about the mainloop.
    using Operator = typename Mma::Operator;
    using OperatorClass = typename Mma::Operator::OperatorClass;
    using ThreadblockShape = typename Mma::Shape;
    using WarpShape = typename Mma::Operator::Shape;
    using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
    using ArchTag = typename Mma::ArchTag;

    static int const kStages = Mma::kStages;
    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    /// Warp count (concept: GemmShape)
    using WarpCount = typename Mma::WarpCount;
    static int const kThreadCount = 32 * WarpCount::kCount;

    static constexpr int kInterleave = Mma::IteratorB::Shape::kRow / Mma::Shape::kK;

    /// Workspace bytes per thread block
    static size_t const kWorkspaceBytesPerBlock =
      __NV_STD_MAX(
        kThreadCount * sizeof(AccumulatorTile),
        Epilogue::kWorkspaceBytesPerBlock);

    /// Block-striped reduction utility
    using BlockStripedReduceT = BlockStripedReduce<kThreadCount, AccumulatorTile>;

    /// Parameters structure
    struct Arguments
    {
        GemmUniversalMode mode = GemmUniversalMode::kGemm;

        cutlass::gemm::GemmCoord problem_size;
        int group_size = -1;
        typename Mma::IteratorA::TensorRef ref_A;
        typename Mma::IteratorB::TensorRef ref_B;
        typename Mma::IteratorScale::TensorRef ref_scale;
        typename Mma::IteratorScale::TensorRef ref_zero;
        typename Epilogue::OutputTileIterator::TensorRef ref_C;
        typename Epilogue::OutputTileIterator::TensorRef ref_D;

        // (mode == GemmUniversalMode::kGemm) the tile-splitting factor
        int batch_count = 1;

        typename EpilogueOutputOp::Params output_op;

        // For gather+scatter operations
        int const* gather_A_indices;
        int const* gather_B_indices;
        int const* scatter_D_indices;

        // Included so we can use Gemm Universal
        int batch_stride_D = 0;

        // The number of SMs that StreamK dispatch heuristics will attempt to
        // load-balance across (-1 defaults to device width, 1 implies classic
        // data-parallel scheduling)
        int avail_sms = -1;

        //
        // Methods
        //

        CUTLASS_HOST_DEVICE
        Arguments() {};

        CUTLASS_HOST_DEVICE
        Arguments(
            cutlass::gemm::GemmCoord const& problem_size,
            int const group_size,
            typename Mma::IteratorA::TensorRef ref_A,
            typename Mma::IteratorB::TensorRef ref_B,
            typename Mma::IteratorScale::TensorRef ref_scale,
            typename Mma::IteratorScale::TensorRef ref_zero,
            typename Epilogue::OutputTileIterator::TensorRef ref_C,
            typename Epilogue::OutputTileIterator::TensorRef ref_D,
            int split_k_factor,
            typename EpilogueOutputOp::Params output_op =
                typename EpilogueOutputOp::Params(),
            int const* gather_A_indices = nullptr,
            int const* gather_B_indices = nullptr,
            int const* scatter_D_indices = nullptr,
            int avail_sms = -1
        ):
            problem_size(problem_size),
            group_size(group_size),
            ref_A(ref_A),
            ref_B(ref_B),
            ref_scale(ref_scale),
            ref_zero(ref_zero),
            ref_C(ref_C),
            ref_D(ref_D),
            batch_count(split_k_factor),
            output_op(output_op),
            gather_A_indices(gather_A_indices),
            gather_B_indices(gather_B_indices),
            scatter_D_indices(scatter_D_indices),
            avail_sms(avail_sms)
        {
        }
    };

    /// Parameters structure
    struct Params
    {
        int group_size;
        typename Mma::IteratorA::Params params_A;
        typename Mma::IteratorA::TensorRef ref_A;
        typename Mma::IteratorB::Params params_B;
        typename Mma::IteratorB::TensorRef ref_B;
        typename Mma::IteratorScale::Params params_scale;
        typename Mma::IteratorScale::TensorRef ref_scale;
        typename Mma::IteratorScale::TensorRef ref_zero;
        typename Epilogue::OutputTileIterator::Params params_C;
        typename Epilogue::OutputTileIterator::TensorRef ref_C;
        typename Epilogue::OutputTileIterator::Params params_D;
        typename Epilogue::OutputTileIterator::TensorRef ref_D;
        typename EpilogueOutputOp::Params output_op;

        int64_t batch_stride_A;
        int64_t batch_stride_B;
        int64_t batch_stride_C;
        int64_t batch_stride_D;

        // For gather+scatter operations
        int const* gather_A_indices;
        int const* gather_B_indices;
        int const* scatter_D_indices;

        GemmUniversalMode mode = GemmUniversalMode::kGemm;
        ThreadblockSwizzle block_mapping;
        void* barrier_workspace = nullptr;
        void* partials_workspace = nullptr;

    protected:
        //
        // Host-only dispatch-utilities
        //

        /// Pad the given allocation size up to the nearest cache line
        static size_t cacheline_align_up(size_t size) {
            static const int CACHELINE_SIZE = 128;
            return (size + CACHELINE_SIZE - 1) / CACHELINE_SIZE * CACHELINE_SIZE;
        }

        /// Get the workspace size needed for barrier
        size_t get_barrier_workspace_size() const {
            // For atomic reduction, each SK-block needs a synchronization flag.
            // For parallel reduction, each reduction block needs its own
            // synchronization flag.
            int sk_blocks =
                block_mapping.sk_regions() * block_mapping.sk_blocks_per_region();
            int num_flags = fast_max(sk_blocks, block_mapping.reduction_blocks);

            return cacheline_align_up(sizeof(typename Barrier::T) * num_flags);
        }

        /// Get the workspace size needed for intermediate partial sums
        size_t get_partials_workspace_size() const {
            int sk_blocks =
                block_mapping.sk_regions() * block_mapping.sk_blocks_per_region();
            return cacheline_align_up(kWorkspaceBytesPerBlock * sk_blocks);
        }

    public:
        //
        // Methods
        //

        CUTLASS_HOST
        /// Default constructor
        Params() {};

        CUTLASS_HOST
        /// Constructor
        Params(
            Arguments const &args,  /// GEMM application arguments
            int device_sms,         /// Number of SMs on the device
            int sm_occupancy)       /// Kernel SM occupancy (in thread blocks)
        :
            group_size(args.group_size),
            params_A(args.ref_A.layout()),
            ref_A(args.ref_A),
            params_B(args.ref_B.layout()),
            ref_B(args.ref_B),
            params_scale(args.ref_scale.layout()),
            ref_scale(args.ref_scale),
            ref_zero(args.ref_zero),
            params_C(args.ref_C.layout()),
            ref_C(args.ref_C),
            params_D(args.ref_D.layout()),
            ref_D(args.ref_D),
            output_op(args.output_op),
            // batch_stride_A(args.problem_size.m() * args.problem_size.k()),
            // batch_stride_B(args.problem_size.n() * args.problem_size.k()),
            // batch_stride_C(args.problem_size.m() * args.problem_size.n()),
            // batch_stride_D(args.problem_size.m() * args.problem_size.n()),
            batch_stride_A(args.ref_A.stride()[0]),
            batch_stride_B(args.ref_B.stride()[0]),
            batch_stride_C(args.ref_C.stride()[0]),
            batch_stride_D(args.ref_D.stride()[0]),
            gather_A_indices(args.gather_A_indices),
            gather_B_indices(args.gather_B_indices),
            scatter_D_indices(args.scatter_D_indices),
            mode(args.mode),
            barrier_workspace(nullptr),
            partials_workspace(nullptr)
        {
          // Number of SMs to make available for StreamK decomposition
          int avail_sms = (args.avail_sms == -1)
                              ? device_sms
                              : fast_min(args.avail_sms, device_sms);

          // Initialize the block mapping structure
          block_mapping = ThreadblockSwizzle(
              args.mode,
              args.problem_size,
              {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
              args.batch_count,
              sm_occupancy,
              device_sms,
              avail_sms,
              sizeof(ElementA),
              sizeof(ElementB),
              sizeof(ElementC),
              Epilogue::kAccumulatorFragments);
        }

        /// Returns the workspace size (in bytes) needed for these parameters
        size_t get_workspace_size() const
        {
            return get_barrier_workspace_size() + get_partials_workspace_size();
        }

        /// Assign and initialize the specified workspace buffer.  Assumes
        /// the memory allocated to workspace is at least as large as
        /// get_workspace_size().
        Status init_workspace(void* workspace, cudaStream_t stream = nullptr)
        {
            uint8_t* ptr = static_cast<uint8_t*>(workspace);

            // Establish partials workspace
            partials_workspace = nullptr;
            size_t partials_workspace_bytes = get_partials_workspace_size();
            if (partials_workspace_bytes > 0)
            {
                if (!workspace)
                {
                return Status::kErrorWorkspaceNull;
                }
                partials_workspace = ptr;
                ptr += partials_workspace_bytes;
            }

            // Establish barrier workspace
            barrier_workspace = nullptr;
            size_t barrier_workspace_bytes = get_barrier_workspace_size();
            if (barrier_workspace_bytes > 0)
            {
                if (!workspace)
                {
                    return Status::kErrorWorkspaceNull;
                }
                barrier_workspace = ptr;
                ptr += barrier_workspace_bytes;
            }

            // Zero-initialize barrier workspace
            if (barrier_workspace)
            {
                size_t barrier_workspace_bytes = get_barrier_workspace_size();

                CUTLASS_TRACE_HOST("  Initialize " << barrier_workspace_bytes
                                                   << " barrier bytes");

                cudaError_t result = cudaMemsetAsync(
                    barrier_workspace, 0, barrier_workspace_bytes, stream);

                if (result != cudaSuccess)
                {
                    CUTLASS_TRACE_HOST("  cudaMemsetAsync() returned error "
                                        << cudaGetErrorString(result));
                    return Status::kErrorInternal;
                }
            }

            return Status::kSuccess;
        }

        /// Returns the GEMM volume in thread block tiles
        cutlass::gemm::GemmCoord get_tiled_shape() const
        {
            return block_mapping.tiled_shape();
        }

        /// Returns the total number of thread blocks to launch
        int get_grid_blocks() const
        {
            dim3 grid_dims = get_grid_dims();
            return grid_dims.x * grid_dims.y * grid_dims.z;
        }

        /// Returns the grid extents in thread blocks to launch
        dim3 get_grid_dims() const
        {
            return block_mapping.get_grid_dims();
        }
    };

    /// Tile work descriptor
    struct TileWorkDesc {
        /// The linear tile index
        int tile_idx;

        /// The location of this tile (in threadblock-tile coordinates) in the
        /// output matrix
        cutlass::gemm::GemmCoord tiled_coord;

        // The first global-scoped MAC-iteration this threadblock will perform for
        // this tile
        int iter_begin;

        // The starting index in the k-domain for MAC-iterations this threadblock
        // will perform for this tile
        int k_begin;

        // The ending index (one-past) in the k-domain for MAC-iterations this
        // threadblock will perform for this tile
        int k_end;

        /// The number of remaining MAC-iterations this threadblock will perform
        /// for this tile
        int k_iters_remaining;

        // Whether this block will perform the first iteration of this tile
        CUTLASS_DEVICE
        bool tile_started()
        {
            return (k_begin == 0);
        }

        // Whether this block will perform the last iteration of this tile
        CUTLASS_DEVICE
        bool tile_finished(Params const& params)
        {
            return (k_end == params.block_mapping.problem_size.k());
        }
    };

    /// Shared memory storage structure
    union SharedStorage
    {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };

protected:

    //
    // Data members
    //

    /// GEMM problem parameters
    Params params;

    /// Shared storage reference
    SharedStorage &shared_storage;

    /// ID within the threadblock
    int thread_idx;

    /// ID of warp
    int warp_idx;

    /// ID of each thread within a warp
    int lane_idx;

    /// Threadblock scoped epilogue
    Epilogue epilogue;

public:

    //
    // Methods
    //

    // Determines whether the GEMM problem size satisfies this kernel's
    // alignment requirements
    CUTLASS_HOST_DEVICE
    static Status can_implement(Arguments const& args)
    {
        static int const kAlignmentA
            = (platform::is_same<LayoutA, layout::ColumnMajorInterleaved<32>>::value) ? 32
            : (platform::is_same<LayoutA, layout::ColumnMajorInterleaved<64>>::value) ? 64
            : Mma::IteratorA::AccessType::kElements;
        static int const kAlignmentB
            = (platform::is_same<LayoutB, layout::RowMajorInterleaved<32>>::value) ? 32
            : (platform::is_same<LayoutB, layout::RowMajorInterleaved<64>>::value) ? 64
            : Mma::IteratorB::AccessType::kElements;

        static int const kAlignmentScale = Mma::IteratorScale::AccessType::kElements;

        static int const kAlignmentC 
            = (platform::is_same<LayoutC, layout::ColumnMajorInterleaved<32>>::value) ? 32
            : (platform::is_same<LayoutC, layout::ColumnMajorInterleaved<64>>::value) ? 64
            : Epilogue::OutputTileIterator::kElementsPerAccess;

        if (platform::is_same<LayoutB, layout::RowMajor>::value &&
                kInterleave == 1 ||
            platform::is_same<LayoutB, layout::ColumnMajor>::value &&
                kInterleave >= 1)
        {
            return Status::kErrorInvalidLayout;
        }

        if (!TensorRef_aligned(args.ref_A, kAlignmentA))
        {
            return Status::kErrorMisalignedOperand;
        }

        if (!TensorRef_aligned(args.ref_B, kAlignmentB))
        {
            return Status::kErrorMisalignedOperand;
        }

        if (!TensorRef_aligned(args.ref_scale, kAlignmentScale))
        {
            return Status::kErrorMisalignedOperand;
        }

        if (!TensorRef_aligned(args.ref_zero, kAlignmentScale))
        {
            return Status::kErrorMisalignedOperand;
        }

        if (!TensorRef_aligned(args.ref_C, kAlignmentC))
        {
            return Status::kErrorMisalignedOperand;
        }

        if (!TensorRef_aligned(args.ref_D, kAlignmentC))
        {
            return Status::kErrorMisalignedOperand;
        }

        if (!args.ref_scale.good())
        {
            return Status::kErrorNotSupported;
        }

        if constexpr (hasZero(Mma::QuantOp))
        {
            if (!args.ref_zero.good())
            {
                return Status::kErrorNotSupported;
            }
        }
        else
        {
            if (args.ref_zero.good())
            {
                return Status::kErrorNotSupported;
            }
        }

        if constexpr (isFinegrained(Mma::QuantOp))
        {
            if (args.group_size != 64 && args.group_size != 128)
            {
                return Status::kErrorNotSupported;
            }
        }

        return Status::kSuccess;
    }

    static size_t get_extra_workspace_size(Arguments const& args, cutlass::gemm::GemmCoord const& grid_tiled_shape)
    {
        return 0;
    }

protected:

    //
    // Device-only utility methods
    //

    CUTLASS_DEVICE
    void init_dp_tile_work(TileWorkDesc& tile_work, int tile_idx)
    {
        // The linear tile index
        tile_work.tile_idx = tile_idx;

        // The first global-scoped MAC-iteration this threadblock will perform for
        // this tile
        tile_work.iter_begin = tile_idx * params.block_mapping.iters_per_tile();

        // The number of MAC-iterations this threadblock will perform for this
        // tile
        tile_work.k_iters_remaining = params.block_mapping.iters_per_tile();

        // The starting index in the k-domain for MAC-iterations this threadblock
        // will perform for this tile
        tile_work.k_begin = 0;

        // The ending index (one-past) in the k-domain for MAC-iterations this
        // threadblock will perform for this tile
        tile_work.k_end = params.block_mapping.problem_size.k();

        // The location of this tile (in threadblock-tile coordinates) in the
        // output matrix
        tile_work.tiled_coord =
            params.block_mapping.get_tile_offset(tile_work.tile_idx);
    }

    CUTLASS_DEVICE
    void init_sk_tile_work(TileWorkDesc& tile_work, int tile_idx,
                           int block_iter_begin, int block_iter_end)
    {
        // The linear tile index
        tile_work.tile_idx = tile_idx;

        // The first global-scoped MAC-iteration for this tile
        int tile_iter_begin = tile_idx * params.block_mapping.iters_per_tile();

        // The first global-scoped MAC-iteration this threadblock will perform for
        // this tile
        tile_work.iter_begin = max(block_iter_begin, tile_iter_begin);

        // The first tile-scoped MAC-iteration this threadblock will perform for
        // this tile
        int k_iter_begin = tile_work.iter_begin - tile_iter_begin;

        // The last (one past) tile-scoped MAC-iteration this threadblock will
        // perform for this tile
        int k_iter_end = block_iter_end - tile_iter_begin;

        // The number of MAC-iterations this threadblock will perform for this
        // tile
        tile_work.k_iters_remaining = k_iter_end - k_iter_begin;

        // The starting index in the k-domain for MAC-iterations this threadblock
        // will perform for this tile
        tile_work.k_begin = k_iter_begin * Mma::Shape::kK;

        // The ending index (one-past) in the k-domain for MAC-iterations this
        // threadblock will perform for this tile
        tile_work.k_end =
            min(params.block_mapping.problem_size.k(),  // extent of k domain
                (k_iter_end * Mma::Shape::kK));  // extent of the threadblock's
                                                // global iteration assignment

        // The location of this tile (in threadblock-tile coordinates) in the
        // output matrix
        tile_work.tiled_coord =
            params.block_mapping.get_tile_offset(tile_work.tile_idx);
    }

    /// Share accumulators with peers
    CUTLASS_DEVICE
    void share_accumulators(AccumulatorTile const& accumulator_tile,
                            int block_idx, int first_block_idx)
    {
        AccumulatorTile* accum_tile_workspace =
            reinterpret_cast<AccumulatorTile*>(params.partials_workspace);

        int accum_tile_offset = first_block_idx * kThreadCount;

        if (block_idx == first_block_idx)
        {
            // First peer initializes the workspace partials
            BlockStripedReduceT::store(accum_tile_workspace + accum_tile_offset,
                                    accumulator_tile, thread_idx);
        }
        else
        {
            // Subsequent peers atomically accumulate into the workspace partials
            if (ThreadblockSwizzle::kReductionStrategy ==
                ThreadblockSwizzle::kAtomic)
            {
                // Non-deterministic reduction order: wait for the first peer to have
                // initialized the partials before we add to them
                Barrier::wait_lt(params.barrier_workspace, thread_idx,
                                first_block_idx, 1);
            }
            else
            {
                // Turnstile reduction order: wait until the previous peer has written
                int wait_count = block_idx - first_block_idx;
                Barrier::wait_eq(params.barrier_workspace, thread_idx,
                                first_block_idx, wait_count);
            }

            // Perform reduction in workspace
            BlockStripedReduceT::reduce(accum_tile_workspace + accum_tile_offset,
                                        accumulator_tile, thread_idx);
        }

        // Signal our arrival
        Barrier::arrive_inc(params.barrier_workspace, thread_idx,
                            first_block_idx);
    }

    /// Acquire accumulators from peers
    CUTLASS_DEVICE
    void acquire_accumulators(AccumulatorTile& accumulator_tile, int block_idx,
                              int first_block_idx)
    {
        AccumulatorTile* accum_tile_workspace =
            reinterpret_cast<AccumulatorTile*>(params.partials_workspace);

        // Wait for arrival
        int num_carry_in = block_idx - first_block_idx;
        Barrier::wait_eq_reset(params.barrier_workspace, thread_idx,
                                first_block_idx, num_carry_in);

        // Load and add peer-partials accumulator tile to local accumulator tile
        int accum_tile_offset = first_block_idx * kThreadCount;
        BlockStripedReduceT::load_add(accumulator_tile,
                                        accum_tile_workspace + accum_tile_offset,
                                        thread_idx);
    }

    /// Perform epilogue computations and output
    CUTLASS_DEVICE
    void do_epilogue(TileWorkDesc& tile_work,
                     AccumulatorTile& accumulator_tile)
    {
        // Location of this tile in item-coords
        MatrixCoord threadblock_item_begin(
            tile_work.tiled_coord.m() * Mma::Shape::kM,
            tile_work.tiled_coord.n() * Mma::Shape::kN
        );

        // Tile iterator loading from source tensor.
        typename Epilogue::OutputTileIterator iterator_C(
            params.params_C, params.ref_C.data(),
            params.block_mapping.problem_size.mn(), thread_idx,
            threadblock_item_begin);

        // Tile iterator writing to destination tensor.
        typename Epilogue::OutputTileIterator iterator_D(
            params.params_D, params.ref_D.data(),
            params.block_mapping.problem_size.mn(), thread_idx,
            threadblock_item_begin);

        // Execute the epilogue operator to update the destination tensor.
        epilogue(EpilogueOutputOp(params.output_op), iterator_D, accumulator_tile,
                iterator_C);
    }

    CUTLASS_DEVICE
    void separate_reduction(int reduce_idx) {
        int peer_idx_begin, peer_idx_last, reduce_tile_idx, reduce_fragment_idx;

        // Reduce by sk-tile (every tile contributed to by one or more blocks)
        reduce_tile_idx = reduce_idx / Epilogue::kAccumulatorFragments;
        reduce_fragment_idx = reduce_idx % Epilogue::kAccumulatorFragments;

        int iter_tile_first =
            reduce_tile_idx * params.block_mapping.iters_per_tile();
        int iter_tile_last =
            iter_tile_first + params.block_mapping.iters_per_tile() - 1;

        peer_idx_begin = params.block_mapping.get_sk_block_idx(iter_tile_first);
        peer_idx_last = params.block_mapping.get_sk_block_idx(iter_tile_last);

        // Wait for peers to complete
        int peer_idx_end = peer_idx_last + 1;
        int num_peers = peer_idx_end - peer_idx_begin;
        Barrier::wait_eq_reset(
            params.barrier_workspace, thread_idx,
            (reduce_tile_idx * Epilogue::kAccumulatorFragments) +
                reduce_fragment_idx,
            num_peers);

        /// The location of this tile (in threadblock-tile coordinates) in the
        /// output matrix
        GemmCoord tiled_coord =
            params.block_mapping.get_tile_offset(reduce_tile_idx);

        // Location of this tile in item-coords
        MatrixCoord threadblock_item_begin(
            tiled_coord.m() * Mma::Shape::kM,
            tiled_coord.n() * Mma::Shape::kN
        );

        // Tile iterator loading from source tensor.
        typename Epilogue::OutputTileIterator iterator_C(
            params.params_C, params.ref_C.data(),
            params.block_mapping.problem_size.mn(), thread_idx,
            threadblock_item_begin);

        // Tile iterator writing to destination tensor.
        typename Epilogue::OutputTileIterator iterator_D(
            params.params_D, params.ref_D.data(),
            params.block_mapping.problem_size.mn(), thread_idx,
            threadblock_item_begin);

        // Execute the epilogue operator to update the destination tensor.
        epilogue.reduce(peer_idx_begin, peer_idx_end, reduce_fragment_idx,
                        params.partials_workspace,
                        EpilogueOutputOp(params.output_op), iterator_D,
                        iterator_C);
    }

    CUTLASS_DEVICE
    void process_tile(TileWorkDesc tile_work, int block_idx,
                      int dp_start_block_idx, int block_iter_begin)
    {
        // Compute initial input iterators location in logical coordinates
        cutlass::MatrixCoord tb_extent_A  // Extent of tensor
        {
            params.block_mapping.problem_size.m(),
            tile_work.k_end,
            // params.problem_size.m() -> params.block_mapping.problem_size.m();
            // problem_size_k -> tile_work.k_end
        };

        MatrixCoord tb_offset_A  // Initial offset of threadblock
        {
            tile_work.tiled_coord.m() * Mma::Shape::kM,
            tile_work.k_begin,
            // threadblock_tile_offset.m() * Mma::Shape::kM -> tile_work.tiled_coord.m() * Mma::Shape::kM;
            // threadblock_tile_offset.k() * params.gemm_k_size,
        };

        MatrixCoord tb_extent_B // Extent of tensor
        {
            tile_work.k_end * kInterleave,
            params.block_mapping.problem_size.n() / kInterleave,
            // problem_size_k * kInterleave, -> tile_work.k_end 
            // params.problem_size.n() / kInterleave -> params.block_mapping.problem_size.n()
        };

        MatrixCoord tb_offset_B
        {
            tile_work.k_begin * kInterleave,
            tile_work.tiled_coord.n() * Mma::Shape::kN / kInterleave,
            // threadblock_tile_offset.k() * params.gemm_k_size * kInterleave -> tile_work.k_begin
            // threadblock_tile_offset.n() * Mma::Shape::kN / kInterleave -> tile_work.tiled_coord.n() * Mma::Shape::kN
        };

        // Construct iterators to A and B operands
        typename Mma::IteratorA iterator_A(params.params_A, params.ref_A.data(),
            tb_extent_A, thread_idx, tb_offset_A, params.gather_A_indices);

        typename Mma::IteratorB iterator_B(params.params_B, params.ref_B.data(),
            tb_extent_B, thread_idx, tb_offset_B, params.gather_B_indices);

        
        MatrixCoord tb_extent_scale
        {
            isFinegrained(Mma::QuantOp) ? tile_work.k_end / 64 : 1,
            params.block_mapping.problem_size.n(),
        };

        MatrixCoord tb_offset_scale
        {
            isFinegrained(Mma::QuantOp) ? tile_work.k_begin / 64 : 0,
            tile_work.tiled_coord.n() * Mma::Shape::kN,
        };

        // Construct iterators to scale and zero-point operands
        typename Mma::IteratorScale iterator_scale = initialize_scale<typename Mma::IteratorScale, Mma::QuantOp>(
            params.params_scale, params.ref_scale.data(), params.ref_zero.data(),
            tb_extent_scale, thread_idx, tb_offset_scale, params.group_size);

        // Initialize accumulators
        AccumulatorTile accumulator_tile;
        accumulator_tile.clear();

        // Initialize MMA abstraction
        Mma mma(
            shared_storage.main_loop,
            params.group_size,
            thread_idx,
            warp_idx,
            lane_idx);

        // Perform this tile's range of multiply-accumulate (MAC) iterations
        mma(tile_work.k_iters_remaining, accumulator_tile, iterator_A, iterator_B, iterator_scale, accumulator_tile);

        if ((ThreadblockSwizzle::kReductionStrategy == ThreadblockSwizzle::kAtomic) ||
            (params.block_mapping.reduction_blocks == 0) ||
            (block_idx >= dp_start_block_idx))
        {
            //
            // Cooperative SK peer reduction or DP block
            //

            int first_block_idx = params.block_mapping.get_first_block_idx(
                tile_work.tile_idx, block_idx);

            if (!tile_work.tile_finished(params))
            {
                // Non "finishing" SK blocks must share their partial accumulator sums
                // through global scratch workspace
                share_accumulators(accumulator_tile, block_idx, first_block_idx);
            }
            else
            {
                // DP blocks and "finishing" SK blocks must perform epilogue
                // operations and write the output tile
                if (!tile_work.tile_started())
                {
                    // A "finishing" SK block must first aggregate its accumulator
                    // partial sums with those shared by peer threadblocks
                    acquire_accumulators(accumulator_tile, block_idx, first_block_idx);
                }

                do_epilogue(tile_work, accumulator_tile);
            }
        }
        else
        {
            //
            // Separate peer reduction
            //

            // Share accumulator partial sums with peer threadblock(s) through
            // scratch workspace
            epilogue.share(block_idx, params.partials_workspace, accumulator_tile, tile_work.tile_started());

            // Signal arrival
            Barrier::arrive_range_inc(
                params.barrier_workspace, thread_idx,
                tile_work.tile_idx * Epilogue::kAccumulatorFragments,
                Epilogue::kAccumulatorFragments);
        }
    }

    // Initializes the fine grained scale+bias iterator. Needed since the fine grained iterator
    // has a different constructor signature than a regular cutlass iterator
    template <typename IteratorScale, WeightOnlyQuantOp op, std::enable_if_t<isFinegrained(op), bool> = true>
    CUTLASS_DEVICE static IteratorScale initialize_scale(typename IteratorScale::Params const& params,
        typename IteratorScale::Pointer pointer_scale, typename IteratorScale::Pointer pointer_zero,
        typename IteratorScale::TensorCoord extent, int thread_id,
        typename IteratorScale::TensorCoord const& threadblock_offset, int group_size)
    {

        return IteratorScale(params, pointer_scale, pointer_zero, extent, thread_id, threadblock_offset, group_size);
    }

    template <typename IteratorScale, WeightOnlyQuantOp op, std::enable_if_t<!isFinegrained(op), bool> = true>
    CUTLASS_DEVICE static IteratorScale initialize_scale(typename IteratorScale::Params const& params,
        typename IteratorScale::Pointer pointer_scale, typename IteratorScale::Pointer pointer_zero,
        typename IteratorScale::TensorCoord extent, int thread_id,
        typename IteratorScale::TensorCoord const& threadblock_offset, int group_size)
    {

        return IteratorScale(params, pointer_scale, extent, thread_id, threadblock_offset);
    }

    CUTLASS_DEVICE
    void gemm()
    {
        // Initialize block's iteration range
        int tile_idx = 0;
        int block_iter_begin = 0;
        int block_iters_remaining = 0;

        int block_idx = params.block_mapping.get_block_idx();

        int sk_padding_start_block_idx =  params.block_mapping.sk_regions() * params.block_mapping.sk_blocks_per_region();
        int dp_start_block_idx = params.block_mapping.sk_waves * params.block_mapping.avail_sms;
        int reduce_start_block_idx = dp_start_block_idx + params.block_mapping.dp_blocks;
        int grid_padding_start_block_idx = reduce_start_block_idx + params.block_mapping.reduction_blocks;

        // Initialize tile work descriptor
        TileWorkDesc tile_work;

        bool dp_block = (block_idx >= dp_start_block_idx) && (block_idx < reduce_start_block_idx);
        bool sk_block = (block_idx < sk_padding_start_block_idx);
        bool reduce_block = (block_idx >= reduce_start_block_idx) &&
                (block_idx < grid_padding_start_block_idx) &&
                (ThreadblockSwizzle::kReductionStrategy == ThreadblockSwizzle::kMixed);

        if (dp_block)
        {
            // This is a DP block
            int dp_block_idx = block_idx - dp_start_block_idx;
            int first_dp_tile = (params.block_mapping.cohort_raster) ? 0 : params.block_mapping.sk_tiles;

            // Blocks in first DP wave get configured number of tiles
            tile_idx = first_dp_tile + dp_block_idx;
            int tile_allottment = params.block_mapping.dp_first_wave_tiles;

            // Blocks in subsequent DP waves get 1 tile
            if (dp_block_idx >= params.block_mapping.avail_sms)
            {
                tile_allottment = 1;
                tile_idx += (params.block_mapping.dp_first_wave_tiles - 1) * params.block_mapping.avail_sms;
            }

            block_iters_remaining = params.block_mapping.iters_per_tile() * tile_allottment;

            init_dp_tile_work(tile_work, tile_idx);

            // DP blocks exit if out of bounds or overlap an SK tile (only possible during cohort rasterization, where dp_first_wave_tiles must be 1)
            if ((tile_idx < params.block_mapping.sk_tiles) ||
                (tile_work.tiled_coord.m() >= params.block_mapping.tiled_shape().m()) ||
                (tile_work.tiled_coord.n() >= params.block_mapping.tiled_shape().n()))
            {
                return;
            }
        }
        else if (sk_block)
        {
            // This is a SK block
            int block_iter_end;
            params.block_mapping.get_iter_extents(block_idx, block_iter_begin, block_iter_end);
            block_iters_remaining = block_iter_end - block_iter_begin;

            tile_idx = params.block_mapping.get_sk_tile_idx(block_iter_end - 1);
            init_sk_tile_work(tile_work, tile_idx, block_iter_begin, block_iter_begin + block_iters_remaining);
        }
        else
        {
            if (reduce_block)
            {
                // This is a reduction threadblock
                int reduce_block_idx = block_idx - reduce_start_block_idx;
                separate_reduction(reduce_block_idx);
            }
            return;
        }

        // Iteration-processing loop body
        CUTLASS_PRAGMA_NO_UNROLL
        while (true)
        {
            // Perform this block's share of work for this tile
            process_tile(
                tile_work,
                block_idx,
                dp_start_block_idx,
                block_iter_begin);

            block_iters_remaining -= tile_work.k_iters_remaining;

            if (block_iters_remaining == 0)
            {
                break;
            }

            // Continue to next tile
            __syncthreads();

            if (block_idx >= dp_start_block_idx)
            {
                // DP block consume their tiles at stride
                tile_idx += params.block_mapping.avail_sms;
                init_dp_tile_work(tile_work, tile_idx);
            }
            else
            {
                // SK blocks consume their tiles in backwards order
                tile_idx--;
                init_sk_tile_work(tile_work, tile_idx, block_iter_begin, block_iter_begin + block_iters_remaining);
            }
        }

    }

    template <typename CompilationArch>
    CUTLASS_DEVICE void run_kernel()
    {
        if constexpr (platform::is_same<KernelArch, CompilationArch>::value)
        {
            gemm();
        }
        else
        {
            CUTLASS_NOT_IMPLEMENTED();
        }
    }

public:
    //
    // Device-only API
    //

    // Factory invocation
    CUTLASS_DEVICE
    static void invoke(
        Params const &params,
        SharedStorage &shared_storage)
    {
        GemmFpAIntBStreamK op(params, shared_storage);
        op();
    }


    // Constructor
    CUTLASS_DEVICE
    GemmFpAIntBStreamK(
        Params const &params,
        SharedStorage &shared_storage)
        :
        params(params),
        shared_storage(shared_storage),
        thread_idx(threadIdx.x),
        warp_idx(__shfl_sync(0xffffffff, threadIdx.x / 32, 0)),   // broadcast the warp_id computed by lane 0 to ensure dependent code
        lane_idx(threadIdx.x % 32),
        epilogue(
            shared_storage.epilogue,
            thread_idx,
            warp_idx,
            lane_idx)
    {}

    /*
        To improve compilation speed, we do not compile the device operator if the CUDA_ARCH does not correspond
        to the ArchTag of the cutlass kernel operator.
      */
    /// Executes one GEMM
    CUTLASS_DEVICE
    void operator()()
    {
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 750)
        run_kernel<arch::Sm70>();
#elif (__CUDA_ARCH__ >= 750) && (__CUDA_ARCH__ < 800)
        run_kernel<arch::Sm75>();
#elif (__CUDA_ARCH__ >= 800) && (__CUDA_ARCH__ < 890)
        run_kernel<arch::Sm80>();
#elif (__CUDA_ARCH__ == 890)
        run_kernel<arch::Sm89>();
#elif (__CUDA_ARCH__ >= 900)
        CUTLASS_NOT_IMPLEMENTED(); // Don't compile these for Hopper or later. Use CUTLASS 3.x kernels.
#else
        static_assert(
            false, "Invalid architecture being compiled. Only Volta+ supported in weight-only quantization kernels.");
#endif
#else
        CUTLASS_NOT_IMPLEMENTED();
#endif
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass