#pragma once

#include <torch/extension.h>
#include <torch/library.h>

namespace vllm_ascend {

int64_t catccos_init(int64_t rank_id,
                     int64_t rank_size,
                     int64_t local_mem_size,
                     c10::string_view ip_port);

at::Tensor catccos_matmul_allreduce(const at::Tensor& a,
                                    const at::Tensor& b,
                                    int64_t rank_size);

int64_t catccos_finalize();

}  // namespace vllm_ascend
