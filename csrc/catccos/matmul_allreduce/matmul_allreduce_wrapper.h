#pragma once

#include <cstdint>
#include <acl/acl_rt.h>

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
                                      int rank_size);

}  // namespace CatccosKernel
