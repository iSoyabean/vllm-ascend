#include "catccos/matmul_allreduce/matmul_allreduce_wrapper.h"

#include "catccos/matmul_allreduce/matmul_allreduce_device.h"

using namespace AscendC;
using namespace Catccos;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutD = Catlass::layout::RowMajor;
using ElementA = half;
using ElementB = half;
using ElementD = half;

namespace CatccosKernel {

void catccos_matmul_allreduce_wrapper(uint32_t block_dim,
                                      aclrtStream stream,
                                      uint64_t ffts_addr,
                                      uint8_t* a_ptr,
                                      uint8_t* b_ptr,
                                      uint8_t* output_ptr,
                                      uint8_t* symmetric_workspace,
                                      uint32_t m,
                                      uint32_t n,
                                      uint32_t k,
                                      int rank_id,
                                      int rank_size)
{
    using Config = MatmulAllReduceConfig_M0_128<ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD>;
    using DeviceOp = Config::Device;

    CocTilingParams coc_tiling;
    coc_tiling.m = m;
    coc_tiling.n = n;
    coc_tiling.k = k;
    coc_tiling.m0 = 128;
    coc_tiling.n0 = 256;
    coc_tiling.k0 = 256;
    coc_tiling.commTileM = 64;
    coc_tiling.commInterval = 3;
    coc_tiling.commNpuSplit = 1;
    coc_tiling.commDataSplit = 20;
    coc_tiling.commBlockM = 64;
    coc_tiling.rankSize = rank_size;

    Catlass::GemmCoord problem_shape{m, n, k};
    Catlass::MatrixCoord comm_core_split{coc_tiling.commDataSplit, coc_tiling.commNpuSplit};
    Catlass::MatrixCoord comm_block_shape{coc_tiling.commBlockM, coc_tiling.n0};
    Catlass::MatrixCoord comm_tile_shape{coc_tiling.commTileM / 2, coc_tiling.n0};

    DeviceOp::Arguments args{
        problem_shape,
        static_cast<uint32_t>(rank_id),
        static_cast<uint32_t>(rank_size),
        coc_tiling.commInterval,
        a_ptr,
        b_ptr,
        output_ptr,
        symmetric_workspace,
        comm_core_split,
        comm_block_shape,
        comm_tile_shape,
    };

    DeviceOp device_op;
    device_op.Initialize(args);
    device_op.Run(stream, block_dim, ffts_addr);
}

}  // namespace CatccosKernel
