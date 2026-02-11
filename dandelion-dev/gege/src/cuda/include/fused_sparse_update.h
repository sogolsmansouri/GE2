#pragma once

#include <torch/torch.h>

// Fused sparse scatter-add update for embeddings and optimizer state.
// Returns true when the CUDA fused path is applied, false when caller should fallback.
bool gege_fused_sparse_add_cuda(torch::Tensor embedding_table,
                                torch::Tensor state_table,
                                torch::Tensor indices,
                                torch::Tensor embedding_delta,
                                torch::Tensor state_delta,
                                bool dedup_before_scatter,
                                int64_t *pre_nnz,
                                int64_t *post_nnz);
