#include "fused_sparse_update.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

namespace {

template <typename scalar_t>
__global__ void fused_sparse_add_atomic_kernel(const int64_t *indices,
                                               const scalar_t *embedding_delta,
                                               const scalar_t *state_delta,
                                               scalar_t *embedding_table,
                                               scalar_t *state_table,
                                               int64_t nnz,
                                               int64_t width,
                                               int64_t rows) {
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = nnz * width;
    if (linear >= total) {
        return;
    }

    int64_t i = linear / width;
    int64_t j = linear - i * width;
    int64_t row = indices[i];
    if (row < 0 || row >= rows) {
        return;
    }

    int64_t out = row * width + j;
    atomicAdd(&embedding_table[out], embedding_delta[linear]);
    atomicAdd(&state_table[out], state_delta[linear]);
}

template <typename scalar_t>
__global__ void fused_sparse_add_noatomic_kernel(const int64_t *indices,
                                                 const scalar_t *embedding_delta,
                                                 const scalar_t *state_delta,
                                                 scalar_t *embedding_table,
                                                 scalar_t *state_table,
                                                 int64_t nnz,
                                                 int64_t width,
                                                 int64_t rows) {
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = nnz * width;
    if (linear >= total) {
        return;
    }

    int64_t i = linear / width;
    int64_t j = linear - i * width;
    int64_t row = indices[i];
    if (row < 0 || row >= rows) {
        return;
    }

    int64_t out = row * width + j;
    embedding_table[out] += embedding_delta[linear];
    state_table[out] += state_delta[linear];
}

bool can_use_fused_path(const torch::Tensor &embedding_table,
                        const torch::Tensor &state_table,
                        const torch::Tensor &indices,
                        const torch::Tensor &embedding_delta,
                        const torch::Tensor &state_delta) {
    if (!embedding_table.defined() || !state_table.defined() || !indices.defined() || !embedding_delta.defined() || !state_delta.defined()) {
        return false;
    }
    if (!embedding_table.is_cuda() || !state_table.is_cuda() || !embedding_delta.is_cuda() || !state_delta.is_cuda()) {
        return false;
    }
    if (embedding_table.device() != state_table.device() || embedding_table.device() != embedding_delta.device() ||
        embedding_table.device() != state_delta.device()) {
        return false;
    }
    if (indices.dim() != 1 || embedding_table.dim() != 2 || state_table.dim() != 2 || embedding_delta.dim() != 2 || state_delta.dim() != 2) {
        return false;
    }
    if (indices.scalar_type() != torch::kInt64) {
        return false;
    }
    if (embedding_table.scalar_type() != embedding_delta.scalar_type() || embedding_table.scalar_type() != state_delta.scalar_type() ||
        embedding_table.scalar_type() != state_table.scalar_type()) {
        return false;
    }

    if (embedding_table.scalar_type() != torch::kFloat32 && embedding_table.scalar_type() != torch::kFloat64) {
        return false;
    }

    int64_t nnz = indices.size(0);
    int64_t width = embedding_table.size(1);
    if (state_table.size(1) != width || embedding_delta.size(0) != nnz || state_delta.size(0) != nnz ||
        embedding_delta.size(1) != width || state_delta.size(1) != width) {
        return false;
    }

    if (!embedding_table.is_contiguous() || !state_table.is_contiguous() || !embedding_delta.is_contiguous() || !state_delta.is_contiguous()) {
        return false;
    }

    return true;
}

} // namespace

bool gege_fused_sparse_add_cuda(torch::Tensor embedding_table,
                                torch::Tensor state_table,
                                torch::Tensor indices,
                                torch::Tensor embedding_delta,
                                torch::Tensor state_delta,
                                bool dedup_before_scatter,
                                int64_t *pre_nnz,
                                int64_t *post_nnz) {
    if (!can_use_fused_path(embedding_table, state_table, indices, embedding_delta, state_delta)) {
        return false;
    }

    auto idx = indices.to(embedding_table.device(), /*non_blocking=*/true).contiguous();
    auto emb_delta = embedding_delta.contiguous();
    auto st_delta = state_delta.contiguous();
    int64_t initial_nnz = idx.numel();
    int64_t reduced_nnz = initial_nnz;

    if (dedup_before_scatter && idx.numel() > 1) {
        // Collapse duplicate rows before scatter. This reduces write conflicts and
        // enables the no-atomic fused kernel path below.
        auto uniq = torch::_unique2(idx, /*sorted=*/true, /*return_inverse=*/true, /*return_counts=*/false);
        auto unique_idx = std::get<0>(uniq);
        auto inverse = std::get<1>(uniq);

        if (unique_idx.numel() < idx.numel()) {
            auto reduced_emb = torch::zeros({unique_idx.size(0), emb_delta.size(1)}, emb_delta.options());
            auto reduced_st = torch::zeros({unique_idx.size(0), st_delta.size(1)}, st_delta.options());
            reduced_emb.index_add_(0, inverse, emb_delta);
            reduced_st.index_add_(0, inverse, st_delta);
            idx = unique_idx.contiguous();
            emb_delta = reduced_emb.contiguous();
            st_delta = reduced_st.contiguous();
            reduced_nnz = idx.numel();
        }
    }

    if (pre_nnz != nullptr) {
        *pre_nnz = initial_nnz;
    }
    if (post_nnz != nullptr) {
        *post_nnz = reduced_nnz;
    }

    int64_t nnz = idx.size(0);
    if (nnz == 0) {
        return true;
    }

    c10::cuda::CUDAGuard device_guard(embedding_table.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int64_t width = embedding_table.size(1);
    int64_t rows = embedding_table.size(0);
    int64_t total = nnz * width;
    constexpr int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(embedding_table.scalar_type(), "gege_fused_sparse_add_cuda", [&] {
        if (dedup_before_scatter) {
            fused_sparse_add_noatomic_kernel<scalar_t><<<blocks, threads, 0, stream>>>(idx.data_ptr<int64_t>(),
                                                                                        emb_delta.data_ptr<scalar_t>(),
                                                                                        st_delta.data_ptr<scalar_t>(),
                                                                                        embedding_table.data_ptr<scalar_t>(),
                                                                                        state_table.data_ptr<scalar_t>(),
                                                                                        nnz,
                                                                                        width,
                                                                                        rows);
        } else {
            fused_sparse_add_atomic_kernel<scalar_t><<<blocks, threads, 0, stream>>>(idx.data_ptr<int64_t>(),
                                                                                      emb_delta.data_ptr<scalar_t>(),
                                                                                      st_delta.data_ptr<scalar_t>(),
                                                                                      embedding_table.data_ptr<scalar_t>(),
                                                                                      state_table.data_ptr<scalar_t>(),
                                                                                      nnz,
                                                                                      width,
                                                                                      rows);
        }
    });

    // Keep launch error checking compatible across Torch/CUDA header versions.
    const cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        return false;
    }
    return true;
}
