/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is adapted from the upstream catccos MMAR device config.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include <cstdint>

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "catccos/catccos.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_swizzle.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/kernel/matmul_allreduce.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"

using half = __fp16;

constexpr uint32_t WORKSPACE_STAGES = 2;
constexpr uint32_t UB_STAGES = 2;

struct CocTilingParams {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t transA = 0;
    uint32_t transB = 0;
    uint32_t m0 = 0;
    uint32_t k0 = 0;
    uint32_t n0 = 0;
    uint32_t commTileM = 0;
    uint32_t commInterval = 0;
    uint32_t commNpuSplit = 0;
    uint32_t commDataSplit = 0;
    uint32_t commBlockM = 0;
    uint32_t rankSize = 0;
    uint32_t epSize = 0;
    uint32_t expertNum = 0;
    uint32_t topK = 1;
};

template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD,
    uint32_t M0_, uint32_t N0_, uint32_t K0_>
struct MatmulAllReduceConfig {
    using ArchTag = Catlass::Arch::AtlasA2;

    static constexpr bool ENABLE_UNIT_FLAG = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG>;

    using L1TileShape = Catlass::GemmShape<M0_, N0_, K0_>;
    using L0TileShape = Catlass::GemmShape<M0_, N0_, 64>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;
    using SymmetricType = DType;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, SymmetricType>;

    static constexpr bool IS_DYNAMIC = true;

    using BlockMmadScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<7, 1>;
    using BlockScheduler = Catccos::Comm::Block::BlockCommSwizzle<IS_DYNAMIC, void, 0, true>;

    using RemoteSrcType = SymmetricType;
    using RemoteDstType = DType;
    using CopyDirect = Catccos::detail::CopyDirect;
    using CopyTransport = Catccos::detail::CopyTransport;
    using TileRemoteCopy = Catccos::Comm::Tile::TileRemoteCopy<
        ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, CopyDirect::Get, CopyTransport::Mte>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using ReduceScatterDispatch = Catccos::Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using BlockReduceScatter = Catccos::Comm::Block::CommBlock<
        ReduceScatterDispatch, RemoteSrcType, RemoteDstType, void, TileRemoteCopy, TileScheduler>;

    using AllGatherDispatch = Catccos::Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
    using BlockAllGather = Catccos::Comm::Block::CommBlock<
        AllGatherDispatch, RemoteSrcType, RemoteDstType, void, TileRemoteCopy, TileScheduler>;

    using Kernel = Catccos::DGemm::Kernel::MatmulAllReduce<
        BlockMmad,
        BlockReduceScatter,
        BlockAllGather,
        BlockMmadScheduler,
        BlockScheduler,
        WORKSPACE_STAGES>;

    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementD, class LayoutD>
using MatmulAllReduceConfig_M0_128 = MatmulAllReduceConfig<
    ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, 128, 256, 256>;

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementD, class LayoutD>
using MatmulAllReduceConfig_M0_256 = MatmulAllReduceConfig<
    ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, 256, 128, 256>;
