#include "storage/graph_storage.h"
#include <c10/cuda/CUDACachingAllocator.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <random>
#include <sstream>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>

#if defined(GEGE_CUDA) && __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define GEGE_HAS_NVTX 1
#elif defined(GEGE_CUDA) && __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#define GEGE_HAS_NVTX 1
#else
#define GEGE_HAS_NVTX 0
#endif

#include "configuration/util.h"
#include "data/ordering.h"
#include "reporting/logger.h"

namespace {
class NvtxRangeGuard {
   public:
    explicit NvtxRangeGuard(const char *name) {
#if GEGE_HAS_NVTX
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }

    ~NvtxRangeGuard() {
#if GEGE_HAS_NVTX
        nvtxRangePop();
#endif
    }
};

bool profile_timing_enabled() {
    static const bool enabled = []() {
        const char *env = std::getenv("GEGE_PROFILE_TIMING");
        if (env == nullptr) {
            return false;
        }
        return !(env[0] == '\0' || (env[0] == '0' && env[1] == '\0'));
    }();
    return enabled;
}

bool debug_subgraph_enabled() {
    static const bool enabled = []() {
        const char *env = std::getenv("GEGE_DEBUG_SUBGRAPH");
        if (env == nullptr) {
            return false;
        }
        return !(env[0] == '\0' || (env[0] == '0' && env[1] == '\0'));
    }();
    return enabled;
}

std::atomic<int64_t> subgraph_call_id_seq{0};

bool fused_sparse_update_enabled() {
    static const bool enabled = []() {
        const char *env = std::getenv("GEGE_ENABLE_FUSED_SPARSE_UPDATE");
        if (env == nullptr) {
            return false;
        }
        return !(env[0] == '\0' || (env[0] == '0' && env[1] == '\0'));
    }();
    return enabled;
}

int fused_sparse_debug_level() {
    static const int level = []() {
        const char *env = std::getenv("GEGE_FUSED_SPARSE_DEBUG");
        if (env == nullptr || env[0] == '\0') {
            return 0;
        }
        int v = std::atoi(env);
        return v > 0 ? v : 0;
    }();
    return level;
}

void fused_sparse_debug_once(const char *msg) {
    static std::atomic<int64_t> budget{0};
    if (fused_sparse_debug_level() <= 0) {
        return;
    }
    if (budget.fetch_add(1) < 16) {
        SPDLOG_INFO("FusedSparse debug: {}", msg);
    }
}
} // namespace

GraphModelStorage::GraphModelStorage(GraphModelStoragePtrs storage_ptrs, shared_ptr<StorageConfig> storage_config) {
    storage_ptrs_ = storage_ptrs;
    train_ = true;
    full_graph_evaluation_ = storage_config->full_graph_evaluation;

    prefetch_ = storage_config->prefetch;
    prefetch_complete_ = false;
    subgraph_lock_ = new std::mutex();
    subgraph_cv_ = new std::condition_variable();

    devices_ = devices_from_config(storage_config);
    active_edges_ = std::vector<EdgeList>(devices_.size());
    current_subgraph_states_ = std::vector<shared_ptr<InMemorySubgraphState>>(devices_.size());

    current_subgraph_state_ = nullptr;
    next_subgraph_state_ = nullptr;
    in_memory_embeddings_ = nullptr;
    in_memory_features_ = nullptr;

    num_nodes_ = storage_config->dataset->num_nodes;
    num_edges_ = storage_config->dataset->num_edges;

    if (full_graph_evaluation_) {
        if (storage_ptrs_.node_embeddings != nullptr) {
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
                string node_embedding_filename = storage_config->model_dir + PathConstants::embeddings_file + PathConstants::file_ext;
                in_memory_embeddings_ =
                    std::make_shared<InMemory>(node_embedding_filename, storage_ptrs_.node_embeddings->dim0_size_, storage_ptrs_.node_embeddings->dim1_size_,
                                               storage_ptrs_.node_embeddings->dtype_, torch::kCPU);
                
            }
        }

        if (storage_ptrs_.node_features != nullptr) {
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features)) {
                string node_feature_filename =
                    storage_config->dataset->dataset_dir + PathConstants::nodes_directory + PathConstants::features_file + PathConstants::file_ext;

                in_memory_features_ = std::make_shared<InMemory>(node_feature_filename, storage_ptrs_.node_features->dim0_size_,
                                                                 storage_ptrs_.node_features->dim1_size_, storage_ptrs_.node_features->dtype_, torch::kCPU);
            }
        }
    }
}

GraphModelStorage::GraphModelStorage(GraphModelStoragePtrs storage_ptrs, bool prefetch) {
    storage_ptrs_ = storage_ptrs;
    train_ = true;
    full_graph_evaluation_ = false;

    prefetch_ = prefetch;
    prefetch_complete_ = false;
    subgraph_lock_ = new std::mutex();
    subgraph_cv_ = new std::condition_variable();

    current_subgraph_state_ = nullptr;
    next_subgraph_state_ = nullptr;
    in_memory_embeddings_ = nullptr;
    in_memory_features_ = nullptr;

    if (storage_ptrs_.node_embeddings != nullptr) {
        num_nodes_ = storage_ptrs_.node_embeddings->getDim0();
    } else if (storage_ptrs_.node_features != nullptr) {
        num_nodes_ = storage_ptrs_.node_features->getDim0();
    } else {
        throw GegeRuntimeException("The input graph must have node features and/or node embeddings");
    }
    num_edges_ = storage_ptrs_.edges->getDim0();
}

GraphModelStorage::~GraphModelStorage() {
    // SPDLOG_INFO("destructing GraphModelStorage");
    unload(false);
    
    delete subgraph_lock_;
    delete subgraph_cv_;
}

void GraphModelStorage::_load(shared_ptr<Storage> storage) {
    if (storage != nullptr) {
        storage->load();
    }
}

void GraphModelStorage::_unload(shared_ptr<Storage> storage, bool write) {
    if (storage != nullptr) {
        storage->unload(write);
    }
}

void GraphModelStorage::load() {
    _load(storage_ptrs_.edges);
    _load(storage_ptrs_.train_edges);
    _load(storage_ptrs_.nodes);

    if (train_) {
        if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            //rePartition();
        }
        _load(storage_ptrs_.node_embeddings);
        _load(storage_ptrs_.node_optimizer_state);
        _load(storage_ptrs_.node_features);
    } else {
        if (storage_ptrs_.node_embeddings != nullptr) {
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings) && full_graph_evaluation_) {
                _load(in_memory_embeddings_);
            } else {
                _load(storage_ptrs_.node_embeddings);
            }
        }

        if (storage_ptrs_.node_features != nullptr) {
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features) && full_graph_evaluation_) {
                _load(in_memory_features_);
            } else {
                _load(storage_ptrs_.node_features);
            }
        }
    }

    _load(storage_ptrs_.encoded_nodes);
    _load(storage_ptrs_.node_labels);
    _load(storage_ptrs_.relation_features);
}

void GraphModelStorage::load_g() {
    if (train_) {
        _load(storage_ptrs_.node_embeddings_g);
        _load(storage_ptrs_.node_optimizer_state_g);
    }
}

void GraphModelStorage::unload(bool write) {
    _unload(storage_ptrs_.edges, false);
    _unload(storage_ptrs_.train_edges, false);
    _unload(storage_ptrs_.validation_edges, false);
    _unload(storage_ptrs_.test_edges, false);
    _unload(storage_ptrs_.nodes, false);
    _unload(storage_ptrs_.train_nodes, false);
    _unload(storage_ptrs_.valid_nodes, false);
    _unload(storage_ptrs_.test_nodes, false);
    _unload(storage_ptrs_.node_embeddings, write);
    _unload(storage_ptrs_.node_embeddings_g, write);
    _unload(storage_ptrs_.encoded_nodes, write);
    _unload(storage_ptrs_.node_optimizer_state, write);
    _unload(storage_ptrs_.node_optimizer_state_g, write);
    _unload(storage_ptrs_.node_features, false);
    _unload(storage_ptrs_.relation_features, false);

    _unload(in_memory_embeddings_, false);
    _unload(in_memory_features_, false);

    for (auto f_edges : storage_ptrs_.filter_edges) {
        _unload(f_edges, false);
    }
    for (int i = 0; i < active_edges_.size(); i ++)
        active_edges_[i] = torch::Tensor();
    active_nodes_ = torch::Tensor();
}

void GraphModelStorage::setEdgesStorage(shared_ptr<Storage> edge_storage) {
    storage_ptrs_.edges = edge_storage; 
}

void GraphModelStorage::setNodesStorage(shared_ptr<Storage> node_storage) { storage_ptrs_.nodes = node_storage; }

EdgeList GraphModelStorage::getEdges(Indices indices, int32_t device_idx) {
    if (active_edges_[device_idx].defined()) {
        return active_edges_[device_idx].index_select(0, indices);
    } else {
        return storage_ptrs_.edges->indexRead(indices);
    }
}

EdgeList GraphModelStorage::getEdgesRange(int64_t start, int64_t size, int device_idx) {
    if (active_edges_[device_idx].defined()) {
        return active_edges_[device_idx].narrow(0, start, size);
    } else {
        return storage_ptrs_.edges->range(start, size);
    }
}

void GraphModelStorage::shuffleEdges() { storage_ptrs_.edges->shuffle(); }

Indices GraphModelStorage::getRandomNodeIds(int64_t size) {
    torch::TensorOptions ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(storage_ptrs_.edges->device_);

    Indices ret;
    if (useInMemorySubGraph()) {
        if (storage_ptrs_.node_embeddings != nullptr) {
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings))
                ret = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->getRandomIds(size);
            if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings))
                ret = std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->getRandomIds(size);
        } else {
            ret = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->getRandomIds(size);
        }
    } else {
        ret = torch::randint(getNumNodesInMemory(), {size}, ind_opts);
    }

    return ret;
}

Indices GraphModelStorage::getNodeIdsRange(int64_t start, int64_t size) {
    if (active_nodes_.defined()) {
        return active_nodes_.narrow(0, start, size);
    } else {
        return storage_ptrs_.nodes->range(start, size).flatten(0, 1);
    }
}

torch::Tensor GraphModelStorage::getNodeEmbeddings(Indices indices, int32_t device_idx) {
    if (!train_ && (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings)) && full_graph_evaluation_) {
        if (in_memory_embeddings_ != nullptr) {
            return in_memory_embeddings_->indexRead(indices);
        } else {
            return torch::Tensor();
        }
    } else {
        if (storage_ptrs_.node_embeddings != nullptr) {
            if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
                return std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->indexRead(indices, device_idx);
            }
            else {
                return storage_ptrs_.node_embeddings->indexRead(indices);
            }
        } else {
            return torch::Tensor();
        }
    }
}

torch::Tensor GraphModelStorage::getNodeEmbeddingsG(Indices indices, int32_t device_idx) {
    if (storage_ptrs_.node_embeddings_g != nullptr) {
        if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings_g)) {
            return std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings_g)->indexRead(indices, device_idx);
        } else {
            return storage_ptrs_.node_embeddings_g->indexRead(indices);
        }
    } else {
        return torch::Tensor();
    }
}

torch::Tensor GraphModelStorage::getNodeEmbeddingsRange(int64_t start, int64_t size) {
    if (storage_ptrs_.node_embeddings != nullptr) {
        return storage_ptrs_.node_embeddings->range(start, size);
    } else {
        return torch::Tensor();
    }
}

torch::Tensor GraphModelStorage::getEncodedNodes(Indices indices) {
    if (storage_ptrs_.encoded_nodes != nullptr) {
        return storage_ptrs_.encoded_nodes->indexRead(indices);
    } else {
        return torch::Tensor();
    }
}

torch::Tensor GraphModelStorage::getEncodedNodesRange(int64_t start, int64_t size) {
    if (storage_ptrs_.encoded_nodes != nullptr) {
        return storage_ptrs_.encoded_nodes->range(start, size);
    } else {
        return torch::Tensor();
    }
}

torch::Tensor GraphModelStorage::getNodeFeatures(Indices indices) {
    if (!train_ && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features) && full_graph_evaluation_) {
        if (in_memory_features_ != nullptr) {
            return in_memory_features_->indexRead(indices);

        } else {
            return torch::Tensor();
        }
    } else {
        if (storage_ptrs_.node_features != nullptr) {
            return storage_ptrs_.node_features->indexRead(indices);
        } else {
            return torch::Tensor();
        }
    }
}

torch::Tensor GraphModelStorage::getNodeFeaturesRange(int64_t start, int64_t size) {
    if (storage_ptrs_.node_features != nullptr) {
        return storage_ptrs_.node_features->range(start, size);
    } else {
        return torch::Tensor();
    }
}

torch::Tensor GraphModelStorage::getNodeLabels(Indices indices) {
    if (storage_ptrs_.node_labels != nullptr) {
        return storage_ptrs_.node_labels->indexRead(indices);
    } else {
        return torch::Tensor();
    }
}

torch::Tensor GraphModelStorage::getNodeLabelsRange(int64_t start, int64_t size) {
    if (storage_ptrs_.node_labels != nullptr) {
        return storage_ptrs_.node_labels->range(start, size);
    } else {
        return torch::Tensor();
    }
}

void GraphModelStorage::updatePutNodeEmbeddings(Indices indices, torch::Tensor embeddings) { storage_ptrs_.node_embeddings->indexPut(indices, embeddings); }

void GraphModelStorage::updateAddNodeEmbeddings(Indices indices, torch::Tensor values, int32_t device_idx) { 
    // add multi-gpu training mode
    if (devices_.size() == 1) {
        storage_ptrs_.node_embeddings->indexAdd(indices, values); 
    } else {
        std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->indexAdd(indices, values, device_idx);
    }
}

bool GraphModelStorage::updateAddNodeEmbeddingsAndStateFused(Indices indices,
                                                             torch::Tensor embedding_values,
                                                             torch::Tensor state_values,
                                                             int32_t device_idx) {
#ifdef GEGE_CUDA
    if (!fused_sparse_update_enabled()) {
        fused_sparse_debug_once("GEGE_ENABLE_FUSED_SPARSE_UPDATE is disabled");
        return false;
    }
    if (storage_ptrs_.node_embeddings == nullptr || storage_ptrs_.node_optimizer_state == nullptr) {
        fused_sparse_debug_once("node_embeddings or node_optimizer_state is null");
        return false;
    }
    auto embeddings_storage = std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings);
    auto state_storage = std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_optimizer_state);
    if (!embeddings_storage || !state_storage) {
        fused_sparse_debug_once("node_embeddings/node_optimizer_state is not MemPartitionBufferStorage");
        return false;
    }
    bool fused_ok = embeddings_storage->indexAddFused(indices, embedding_values, state_values, state_storage, device_idx);
    static bool logged_enabled = false;
    if (fused_ok && !logged_enabled) {
        SPDLOG_INFO("Fused sparse embedding/state update enabled for MEM_PARTITION_BUFFER");
        logged_enabled = true;
    }
    return fused_ok;
#else
    (void)indices;
    (void)embedding_values;
    (void)state_values;
    (void)device_idx;
    return false;
#endif
}

void GraphModelStorage::updateAddNodeEmbeddingsG(Indices indices, torch::Tensor values, int32_t device_idx) {
    // add multi-gpu training mode
    if (devices_.size() == 1) {
        storage_ptrs_.node_embeddings_g->indexAdd(indices, values);
    } else {
        std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings_g)->indexAdd(indices, values, device_idx);
    }
}

void GraphModelStorage::updatePutEncodedNodes(Indices indices, torch::Tensor values) { storage_ptrs_.encoded_nodes->indexPut(indices, values); }

void GraphModelStorage::updatePutEncodedNodesRange(int64_t start, int64_t size, torch::Tensor values) {
    storage_ptrs_.encoded_nodes->rangePut(start, size, values);
}

OptimizerState GraphModelStorage::getNodeEmbeddingState(Indices indices, int32_t device_idx) {
    if (storage_ptrs_.node_optimizer_state != nullptr) {
        if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_optimizer_state)) {
            return std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_optimizer_state)->indexRead(indices, device_idx);
        }
        else {
            return storage_ptrs_.node_optimizer_state->indexRead(indices);
        }
    } else {
        return torch::Tensor();
    }
}

OptimizerState GraphModelStorage::getNodeEmbeddingStateG(Indices indices, int32_t device_idx) {
    if (storage_ptrs_.node_optimizer_state_g != nullptr) {
        if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings_g)) {
            return std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_optimizer_state_g)->indexRead(indices, device_idx);
        }
        else {
            return storage_ptrs_.node_optimizer_state_g->indexRead(indices);
        }
    } else {
        return torch::Tensor();
    }
}

OptimizerState GraphModelStorage::getNodeEmbeddingStateRange(int64_t start, int64_t size) {
    if (storage_ptrs_.node_optimizer_state != nullptr) {
        return storage_ptrs_.node_optimizer_state->range(start, size);
    } else {
        return torch::Tensor();
    }
}

void GraphModelStorage::updatePutNodeEmbeddingState(Indices indices, OptimizerState state) {
    if (storage_ptrs_.node_optimizer_state != nullptr) {
        storage_ptrs_.node_optimizer_state->indexPut(indices, state);
    }
}

void GraphModelStorage::updateAddNodeEmbeddingState(Indices indices, torch::Tensor values, int32_t device_idx) {
    // if (storage_ptrs_.node_optimizer_state != nullptr) {
    //     storage_ptrs_.node_optimizer_state->indexAdd(indices, values);
    // }
    // replace with multi-gpu mode
    // SPDLOG_INFO("updateAddNodeEmbeddingState");
    if (devices_.size() == 1) {
        storage_ptrs_.node_optimizer_state->indexAdd(indices, values); 
    } else {
        std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_optimizer_state)->indexAdd(indices, values, device_idx);
    }
}

void GraphModelStorage::updateAddNodeEmbeddingStateG(Indices indices, torch::Tensor values, int32_t device_idx) {
    // if (storage_ptrs_.node_optimizer_state != nullptr) {
    //     storage_ptrs_.node_optimizer_state->indexAdd(indices, values);
    // }
    // replace with multi-gpu mode
    if (devices_.size() == 1) {
        storage_ptrs_.node_optimizer_state_g->indexAdd(indices, values);
    } else {
        std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_optimizer_state_g)->indexAdd(indices, values, device_idx);
    }
}

bool GraphModelStorage::embeddingsOffDevice() {
    if (storage_ptrs_.node_embeddings != nullptr) {
        return storage_ptrs_.node_embeddings->device_ != torch::kCUDA;
    } else if (storage_ptrs_.node_features != nullptr) {
        return storage_ptrs_.node_features->device_ != torch::kCUDA;
    } else {
        return false;
    }
}

bool GraphModelStorage::embeddingsOffDeviceG() {
    if (storage_ptrs_.node_embeddings_g != nullptr) {
        return storage_ptrs_.node_embeddings_g->device_ != torch::kCUDA;
    } else {
        return false;
    }
}

void GraphModelStorage::initializeInMemorySubGraph(torch::Tensor buffer_state, torch::Device device, int32_t device_idx) {
    if (useInMemorySubGraph()) {
        current_subgraph_states_[device_idx] = nullptr;
        current_subgraph_state_ = std::make_shared<InMemorySubgraphState>();

        buffer_state = buffer_state.to(torch::kInt64);

        int buffer_size = buffer_state.size(0);
        int num_edge_buckets_in_mem = buffer_size * buffer_size;
        int num_partitions = getNumPartitions();

        torch::Tensor new_in_mem_partition_ids = buffer_state;
        auto new_in_mem_partition_ids_accessor = new_in_mem_partition_ids.accessor<int64_t, 1>();

        torch::Tensor in_mem_edge_bucket_ids = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);
        torch::Tensor in_mem_edge_bucket_sizes = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);
        torch::Tensor global_edge_bucket_starts = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);

        auto in_mem_edge_bucket_ids_accessor = in_mem_edge_bucket_ids.accessor<int64_t, 1>();
        auto in_mem_edge_bucket_sizes_accessor = in_mem_edge_bucket_sizes.accessor<int64_t, 1>();
        auto global_edge_bucket_starts_accessor = global_edge_bucket_starts.accessor<int64_t, 1>();

        // TODO we don't need to do this every time
        std::vector<int64_t> edge_bucket_sizes_ = storage_ptrs_.edges->getEdgeBucketSizes();
        torch::Tensor edge_bucket_sizes = torch::from_blob(edge_bucket_sizes_.data(), {(int)edge_bucket_sizes_.size()}, torch::kInt64);
        torch::Tensor edge_bucket_ends_disk = edge_bucket_sizes.cumsum(0);
        torch::Tensor edge_bucket_starts_disk = edge_bucket_ends_disk - edge_bucket_sizes;
        auto edge_bucket_sizes_accessor = edge_bucket_sizes.accessor<int64_t, 1>();
        auto edge_bucket_starts_disk_accessor = edge_bucket_starts_disk.accessor<int64_t, 1>();

#pragma omp parallel for
        for (int i = 0; i < buffer_size; i++) {
            for (int j = 0; j < buffer_size; j++) {
                int64_t edge_bucket_id = new_in_mem_partition_ids_accessor[i] * num_partitions + new_in_mem_partition_ids_accessor[j];
                int64_t edge_bucket_size = edge_bucket_sizes_accessor[edge_bucket_id];
                int64_t edge_bucket_start = edge_bucket_starts_disk_accessor[edge_bucket_id];

                int idx = i * buffer_size + j;
                in_mem_edge_bucket_ids_accessor[idx] = edge_bucket_id;
                in_mem_edge_bucket_sizes_accessor[idx] = edge_bucket_size;
                global_edge_bucket_starts_accessor[idx] = edge_bucket_start;
            }
        }

        torch::Tensor in_mem_edge_bucket_starts = in_mem_edge_bucket_sizes.cumsum(0);
        int64_t total_size = in_mem_edge_bucket_starts[-1].item<int64_t>();
        in_mem_edge_bucket_starts = in_mem_edge_bucket_starts - in_mem_edge_bucket_sizes;

        auto in_mem_edge_bucket_starts_accessor = in_mem_edge_bucket_starts.accessor<int64_t, 1>();
        current_subgraph_state_->all_in_memory_edges_ = torch::empty({total_size, storage_ptrs_.edges->dim1_size_}, torch::kInt64);

#pragma omp parallel for
        for (int i = 0; i < num_edge_buckets_in_mem; i++) {
            int64_t edge_bucket_size = in_mem_edge_bucket_sizes_accessor[i];
            int64_t edge_bucket_start = global_edge_bucket_starts_accessor[i];
            int64_t local_offset = in_mem_edge_bucket_starts_accessor[i];
            current_subgraph_state_->all_in_memory_edges_.narrow(0, local_offset, edge_bucket_size) =
                storage_ptrs_.edges->range(edge_bucket_start, edge_bucket_size);
        }


        if (storage_ptrs_.node_embeddings != nullptr) {
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
                current_subgraph_state_->global_to_local_index_map_ =
                    std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->getGlobalToLocalMap(true);
            }
            else if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
                current_subgraph_state_->global_to_local_index_map_ =
                    std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->getGlobalToLocalMap(true, device_idx);
            }
            else {
                SPDLOG_ERROR("Backend type not available for embeddings.");
                throw std::runtime_error("");
            }
        } else if (storage_ptrs_.node_features != nullptr) {
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features)) {
                current_subgraph_state_->global_to_local_index_map_ =
                    std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->getGlobalToLocalMap(true);
            }
            else if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_features)) {
                current_subgraph_state_->global_to_local_index_map_ =
                    std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_features)->getGlobalToLocalMap(true);
            }
            else {
                SPDLOG_ERROR("Backend type not available for features.");
                throw std::runtime_error("");
            }
        }
        
        torch::Tensor mapped_edges;
        torch::Tensor mapped_edges_dst_sort;
        if (storage_ptrs_.edges->dim1_size_ == 3) {
            mapped_edges =
                torch::stack({current_subgraph_state_->global_to_local_index_map_.index_select(0, current_subgraph_state_->all_in_memory_edges_.select(1, 0)),
                              current_subgraph_state_->all_in_memory_edges_.select(1, 1),
                              current_subgraph_state_->global_to_local_index_map_.index_select(0, current_subgraph_state_->all_in_memory_edges_.select(1, -1))})
                    .transpose(0, 1);
        } else if (storage_ptrs_.edges->dim1_size_ == 2) {
            mapped_edges =
                torch::stack({current_subgraph_state_->global_to_local_index_map_.index_select(0, current_subgraph_state_->all_in_memory_edges_.select(1, 0)),
                              current_subgraph_state_->global_to_local_index_map_.index_select(0, current_subgraph_state_->all_in_memory_edges_.select(1, -1))})
                    .transpose(0, 1);
        } else {
            SPDLOG_ERROR("Unexpected number of edge columns");
            std::runtime_error("Unexpected number of edge columns");
        }
        mapped_edges = mapped_edges.to(device);
        current_subgraph_state_->all_in_memory_mapped_edges_ = mapped_edges;

        mapped_edges          = merge_sorted_edge_buckets(mapped_edges, in_mem_edge_bucket_starts, buffer_size, true);
        mapped_edges_dst_sort = merge_sorted_edge_buckets(mapped_edges, in_mem_edge_bucket_starts, buffer_size, false);
        if (devices_.size() <= 1) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

        mapped_edges = mapped_edges.to(torch::kInt64);
        mapped_edges_dst_sort = mapped_edges_dst_sort.to(torch::kInt64);

        if (current_subgraph_state_->in_memory_subgraph_ != nullptr) {
            current_subgraph_state_->in_memory_subgraph_ = nullptr;
        }

        current_subgraph_state_->in_memory_subgraph_ = std::make_shared<GegeGraph>(mapped_edges, mapped_edges_dst_sort, getNumNodesInMemory(device_idx));
        current_subgraph_state_->in_memory_partition_ids_ = new_in_mem_partition_ids;
        current_subgraph_state_->in_memory_edge_bucket_ids_ = in_mem_edge_bucket_ids;
        current_subgraph_state_->in_memory_edge_bucket_sizes_ = in_mem_edge_bucket_sizes;
        current_subgraph_state_->in_memory_edge_bucket_starts_ = in_mem_edge_bucket_starts;
        if (prefetch_) {
            if (hasSwap()) {
                // update next_subgraph_state_ in background
                getNextSubGraph();
            }
        }
        current_subgraph_states_[device_idx] = current_subgraph_state_;
    } else {
        current_subgraph_state_ = std::make_shared<InMemorySubgraphState>();
        current_subgraph_states_[0] = current_subgraph_state_;
        EdgeList src_sort;
        EdgeList dst_sort;

        if (storage_ptrs_.train_edges != nullptr) {
            src_sort = storage_ptrs_.train_edges->range(0, storage_ptrs_.train_edges->getDim0()).to(torch::kInt64);
            dst_sort = storage_ptrs_.train_edges->range(0, storage_ptrs_.train_edges->getDim0()).to(torch::kInt64);
        } else {
            src_sort = storage_ptrs_.edges->range(0, storage_ptrs_.edges->getDim0()).to(torch::kInt64);
            dst_sort = storage_ptrs_.edges->range(0, storage_ptrs_.edges->getDim0()).to(torch::kInt64);
        }
        src_sort = src_sort.index_select(0, torch::argsort(src_sort.select(1, 0))).to(torch::kInt64);
        dst_sort = dst_sort.index_select(0, torch::argsort(dst_sort.select(1, -1))).to(torch::kInt64);

        current_subgraph_states_[0]->in_memory_subgraph_ = std::make_shared<GegeGraph>(src_sort, dst_sort, getNumNodesInMemory());
    }
}

void GraphModelStorage::updateInMemorySubGraph(int32_t device_idx) {
    if (prefetch_) {
        // wait until the prefetching has been completed
        std::unique_lock lock(*subgraph_lock_);
        subgraph_cv_->wait(lock, [this] { return prefetch_complete_ == true; });
        // need to wait for the subgraph to be prefetched to perform the swap, otherwise the prefetched buffer_index_map may be incorrect
        auto t1 = std::chrono::high_resolution_clock::now();
        performSwap();
        auto t2 = std::chrono::high_resolution_clock::now();
        SPDLOG_INFO("performSwap time {}", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
        // free previous subgraph
        current_subgraph_state_->in_memory_subgraph_ = nullptr;
        current_subgraph_state_ = nullptr;

        current_subgraph_state_ = next_subgraph_state_;
        next_subgraph_state_ = nullptr;
        prefetch_complete_ = false;

        if (hasSwap()) {
            // update next_subgraph_state_ in background
            getNextSubGraph();
        }
    } else {
        
        auto mem_buffer_guard = std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings);
        if (!mem_buffer_guard) {
            mem_buffer_guard = std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings_g);
        }
        // Coordinate swaps for multi-GPU MEM_PARTITION_BUFFER in one place (device 0),
        // regardless of P2P mode. Non-P2P path also swaps all devices per call.
        if (mem_buffer_guard && devices_.size() > 1) {
            {
                std::lock_guard<std::mutex> lk(p2p_update_mutex_);
                if (p2p_update_last_seen_.size() != devices_.size()) {
                    p2p_update_last_seen_.assign(devices_.size(), 0);
                }
            }
            if (device_idx != 0) {
                std::unique_lock<std::mutex> lk(p2p_update_mutex_);
                int last = p2p_update_last_seen_[device_idx];
                if (p2p_update_gen_ != last) {
                    p2p_update_last_seen_[device_idx] = p2p_update_gen_;
                    return;
                }
                p2p_update_cv_.wait(lk, [&]{ return p2p_update_gen_ != last; });
                p2p_update_last_seen_[device_idx] = p2p_update_gen_;
                return;
            }

            std::vector<std::pair<std::vector<int>, std::vector<int>>> swap_ids_per_device(devices_.size());
            for (int d = 0; d < devices_.size(); ++d) {
                swap_ids_per_device[d] = getNextSwapIds(d);
            }
            auto coordinator_start = std::chrono::steady_clock::now();
            auto swap_start = std::chrono::steady_clock::now();
            performSwap(0);
            auto swap_end = std::chrono::steady_clock::now();
            if (devices_.size() <= 1) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

            auto rebuild_start = std::chrono::steady_clock::now();
            std::vector<int64_t> rebuild_ms_per_device(devices_.size(), 0);
            for (int d = 0; d < devices_.size(); ++d) {
                auto rebuild_device_start = std::chrono::steady_clock::now();
                updateInMemorySubGraph_(current_subgraph_states_[d], swap_ids_per_device[d], d);
                auto rebuild_device_end = std::chrono::steady_clock::now();
                rebuild_ms_per_device[d] = std::chrono::duration_cast<std::chrono::milliseconds>(rebuild_device_end - rebuild_device_start).count();
            }
            auto rebuild_end = std::chrono::steady_clock::now();

            {
                std::lock_guard<std::mutex> lk(p2p_update_mutex_);
                p2p_update_gen_++;
                if (p2p_update_last_seen_.size() != devices_.size()) {
                    p2p_update_last_seen_.assign(devices_.size(), 0);
                }
                for (int d = 0; d < devices_.size(); ++d) {
                    p2p_update_last_seen_[d] = p2p_update_gen_;
                }
            }
            p2p_update_cv_.notify_all();
            if (profile_timing_enabled()) {
                auto coordinator_end = std::chrono::steady_clock::now();
                auto swap_ms = std::chrono::duration_cast<std::chrono::milliseconds>(swap_end - swap_start).count();
                auto rebuild_ms = std::chrono::duration_cast<std::chrono::milliseconds>(rebuild_end - rebuild_start).count();
                auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(coordinator_end - coordinator_start).count();
                std::ostringstream per_device_rebuild;
                for (int d = 0; d < static_cast<int>(rebuild_ms_per_device.size()); ++d) {
                    if (d > 0) {
                        per_device_rebuild << ", ";
                    }
                    per_device_rebuild << "d" << d << "=" << rebuild_ms_per_device[d] << "ms";
                }
                SPDLOG_INFO("Timing updateInMemorySubGraph coordinator: swap={}ms rebuild_total={}ms total={}ms [{}]",
                            swap_ms,
                            rebuild_ms,
                            total_ms,
                            per_device_rebuild.str());
            }
            return;
        }
std::pair<std::vector<int>, std::vector<int>> current_swap_ids = getNextSwapIds(device_idx);
        // SPDLOG_INFO("performSwap");
        auto call_start = std::chrono::steady_clock::now();
        auto swap_start = std::chrono::steady_clock::now();
        performSwap(device_idx);
        auto swap_end = std::chrono::steady_clock::now();
        // SPDLOG_INFO("performSwap time {}", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
        if (devices_.size() <= 1) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
        // SPDLOG_INFO("updateInMemorySubGraph_");
        auto rebuild_start = std::chrono::steady_clock::now();
        updateInMemorySubGraph_(current_subgraph_states_[device_idx], current_swap_ids, device_idx);
        auto rebuild_end = std::chrono::steady_clock::now();
        if (profile_timing_enabled()) {
            auto call_end = std::chrono::steady_clock::now();
            auto swap_ms = std::chrono::duration_cast<std::chrono::milliseconds>(swap_end - swap_start).count();
            auto rebuild_ms = std::chrono::duration_cast<std::chrono::milliseconds>(rebuild_end - rebuild_start).count();
            auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(call_end - call_start).count();
            SPDLOG_INFO("Timing updateInMemorySubGraph device={}: swap={}ms rebuild={}ms total={}ms",
                        device_idx,
                        swap_ms,
                        rebuild_ms,
                        total_ms);
        }
        // SPDLOG_INFO("updateInMemorySubGraph_ time {}", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
        // SPDLOG_INFO("performSwap time {}, updateInMemorySubGraph_ {}", duration, duration2);
        // for (int device_idx = 0; device_idx < current_subgraph_states_.size(); device_idx ++) {
        //     // SPDLOG_INFO("device_idx {}", device_idx);
        //     std::pair<std::vector<int>, std::vector<int>> current_swap_ids = getNextSwapIds(device_idx);
        //     performSwap(device_idx);
            
        //     std::vector<int> evict_partition_ids = std::get<0>(current_swap_ids);
        //     std::vector<int> admit_partition_ids = std::get<1>(current_swap_ids);

        //     // for(auto& ids : evict_partition_ids) {
        //     //     SPDLOG_INFO("evict_partition_ids {}", ids);
        //     // }

        //     // for(auto& ids : admit_partition_ids) {
        //     //     SPDLOG_INFO("admit_partition_ids {}", ids);
        //     // }
        //     // SPDLOG_INFO("start updateInMemorySubGraph_");
        //     updateInMemorySubGraph_(current_subgraph_states_[device_idx], current_swap_ids, device_idx);
        //     // SPDLOG_INFO("after updateInMemorySubGraph_");
        //     // SPDLOG_INFO("updateInMemorySubGraph_ finished {}", device_idx);
        // }
    }
}

void GraphModelStorage::getNextSubGraph() {
    std::pair<std::vector<int>, std::vector<int>> next_swap_ids = getNextSwapIds();
    if (debug_subgraph_enabled()) {
        SPDLOG_INFO("SUBGRAPH prefetch trigger: device={} evict_partitions={} admit_partitions={}",
                    0,
                    std::get<0>(next_swap_ids).size(),
                    std::get<1>(next_swap_ids).size());
    }
    next_subgraph_state_ = std::make_shared<InMemorySubgraphState>();
    next_subgraph_state_->in_memory_subgraph_ = nullptr;
    std::thread(&GraphModelStorage::updateInMemorySubGraph_, this, next_subgraph_state_, next_swap_ids, 0).detach();
}

void GraphModelStorage::updateInMemorySubGraph_(shared_ptr<InMemorySubgraphState> subgraph, std::pair<std::vector<int>, std::vector<int>> swap_ids, int32_t device_idx) {
    int64_t subgraph_call_id = subgraph_call_id_seq.fetch_add(1) + 1;
    bool debug_subgraph = debug_subgraph_enabled();
    auto total_start = std::chrono::steady_clock::now();
    if (prefetch_) {
        subgraph_lock_->lock();
    }

    std::vector<int> evict_partition_ids = std::get<0>(swap_ids);
    std::vector<int> admit_partition_ids = std::get<1>(swap_ids);

    torch::Tensor admit_ids_tensor = torch::tensor(admit_partition_ids, torch::kCPU);

    int buffer_size = current_subgraph_states_[device_idx]->in_memory_partition_ids_.size(0);
    int num_edge_buckets_in_mem = current_subgraph_states_[device_idx]->in_memory_edge_bucket_ids_.size(0);
    // SPDLOG_INFO("buffer_size {}, num_edge_buckets_in_mem {}", buffer_size, num_edge_buckets_in_mem);
    int num_partitions = getNumPartitions();
    int num_swap_partitions = evict_partition_ids.size();
    int num_remaining_partitions = buffer_size - num_swap_partitions;
    if (debug_subgraph) {
        SPDLOG_INFO("SUBGRAPH call={} device={} start evict_partitions={} admit_partitions={} buffer_size={} edge_buckets_in_mem={} num_partitions={}",
                    subgraph_call_id,
                    device_idx,
                    evict_partition_ids.size(),
                    admit_partition_ids.size(),
                    buffer_size,
                    num_edge_buckets_in_mem,
                    num_partitions);
    }

    auto bucket_partition_start = std::chrono::steady_clock::now();
    // get edge buckets that will be kept in memory
    torch::Tensor keep_mask;
    {
        NvtxRangeGuard nvtx_bucket_filter("subgraph.bucket_filter_masked_select");
        keep_mask = torch::ones({num_edge_buckets_in_mem}, torch::kBool);
    }
    auto accessor_keep_mask = keep_mask.accessor<bool, 1>();
    auto accessor_in_memory_edge_bucket_ids_ = current_subgraph_states_[device_idx]->in_memory_edge_bucket_ids_.accessor<int64_t, 1>();

#pragma omp parallel for
    for (int i = 0; i < num_edge_buckets_in_mem; i++) {
        int64_t edge_bucket_id = accessor_in_memory_edge_bucket_ids_[i];
        int64_t src_partition = edge_bucket_id / num_partitions;
        int64_t dst_partition = edge_bucket_id % num_partitions;

        for (int j = 0; j < num_swap_partitions; j++) {
            if (src_partition == evict_partition_ids[j] || dst_partition == evict_partition_ids[j]) {
                accessor_keep_mask[i] = false;
            }
        }
    }

    torch::Tensor in_mem_edge_bucket_ids;
    torch::Tensor in_mem_edge_bucket_sizes;
    torch::Tensor local_or_global_edge_bucket_starts;
    {
        NvtxRangeGuard nvtx_bucket_filter("subgraph.bucket_filter_masked_select");
        in_mem_edge_bucket_ids = current_subgraph_states_[device_idx]->in_memory_edge_bucket_ids_.masked_select(keep_mask);
        in_mem_edge_bucket_sizes = current_subgraph_states_[device_idx]->in_memory_edge_bucket_sizes_.masked_select(keep_mask);
        local_or_global_edge_bucket_starts = current_subgraph_states_[device_idx]->in_memory_edge_bucket_starts_.masked_select(keep_mask);
    }

    // get new in memory partition ids
    keep_mask = torch::ones({buffer_size}, torch::kBool);
    accessor_keep_mask = keep_mask.accessor<bool, 1>();
    auto accessor_in_memory_partition_ids_ = current_subgraph_states_[device_idx]->in_memory_partition_ids_.accessor<int64_t, 1>();

#pragma omp parallel for
    for (int i = 0; i < buffer_size; i++) {
        int64_t partition_id = accessor_in_memory_partition_ids_[i];

        for (int j = 0; j < num_swap_partitions; j++) {
            if (partition_id == evict_partition_ids[j]) {
                accessor_keep_mask[i] = false;
                break;
            }
        }
    }

    torch::Tensor old_in_mem_partition_ids;
    torch::Tensor new_in_mem_partition_ids;
    {
        NvtxRangeGuard nvtx_bucket_filter("subgraph.bucket_filter_masked_select");
        old_in_mem_partition_ids = current_subgraph_states_[device_idx]->in_memory_partition_ids_.masked_select(keep_mask);
        new_in_mem_partition_ids = current_subgraph_states_[device_idx]->in_memory_partition_ids_.masked_scatter(~keep_mask, admit_ids_tensor);
    }
    auto old_in_mem_partition_ids_accessor = old_in_mem_partition_ids.accessor<int64_t, 1>();
    auto new_in_mem_partition_ids_accessor = new_in_mem_partition_ids.accessor<int64_t, 1>();
    auto bucket_partition_end = std::chrono::steady_clock::now();
    if (debug_subgraph) {
        auto bucket_partition_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(bucket_partition_end - bucket_partition_start).count();
        SPDLOG_INFO("SUBGRAPH call={} stage=partition_filter kept_edge_buckets={} kept_partitions={} swapped_partitions={} ms={}",
                    subgraph_call_id,
                    in_mem_edge_bucket_ids.size(0),
                    old_in_mem_partition_ids.size(0),
                    num_swap_partitions,
                    bucket_partition_ms);
    }

    // get new incoming edge buckets
    auto bucket_meta_start = std::chrono::steady_clock::now();
    int num_new_edge_buckets = num_swap_partitions * (num_remaining_partitions + buffer_size);
    // SPDLOG_INFO("num_new_edge_buckets {}", num_new_edge_buckets);
    torch::Tensor new_edge_bucket_ids = torch::zeros({num_new_edge_buckets}, torch::kInt64);
    torch::Tensor new_edge_bucket_sizes = torch::zeros({num_new_edge_buckets}, torch::kInt64);
    torch::Tensor new_global_edge_bucket_starts = torch::zeros({num_new_edge_buckets}, torch::kInt64);

    auto new_edge_bucket_ids_accessor = new_edge_bucket_ids.accessor<int64_t, 1>();
    auto new_edge_bucket_sizes_accessor = new_edge_bucket_sizes.accessor<int64_t, 1>();
    auto new_global_edge_bucket_starts_accessor = new_global_edge_bucket_starts.accessor<int64_t, 1>();

    // TODO we don't need to do this every time
    std::vector<int64_t> edge_bucket_sizes_ = storage_ptrs_.edges->getEdgeBucketSizes();
    torch::Tensor edge_bucket_sizes = torch::from_blob(edge_bucket_sizes_.data(), {(int)edge_bucket_sizes_.size()}, torch::kInt64);
    torch::Tensor edge_bucket_ends_disk = edge_bucket_sizes.cumsum(0);
    torch::Tensor edge_bucket_starts_disk = edge_bucket_ends_disk - edge_bucket_sizes;
    auto edge_bucket_sizes_accessor = edge_bucket_sizes.accessor<int64_t, 1>();
    auto edge_bucket_starts_disk_accessor = edge_bucket_starts_disk.accessor<int64_t, 1>();

#pragma omp parallel for
    for (int i = 0; i < num_remaining_partitions; i++) {
        for (int j = 0; j < num_swap_partitions; j++) {
            int64_t edge_bucket_id = old_in_mem_partition_ids_accessor[i] * num_partitions + admit_partition_ids[j];
            int64_t edge_bucket_size = edge_bucket_sizes_accessor[edge_bucket_id];
            int64_t edge_bucket_start = edge_bucket_starts_disk_accessor[edge_bucket_id];

            int idx = i * num_swap_partitions + j;
            new_edge_bucket_ids_accessor[idx] = edge_bucket_id;
            new_edge_bucket_sizes_accessor[idx] = edge_bucket_size;
            new_global_edge_bucket_starts_accessor[idx] = edge_bucket_start;
        }
    }

    int offset = num_swap_partitions * num_remaining_partitions;

#pragma omp parallel for
    for (int i = 0; i < buffer_size; i++) {
        for (int j = 0; j < num_swap_partitions; j++) {
            int64_t edge_bucket_id = admit_partition_ids[j] * num_partitions + new_in_mem_partition_ids_accessor[i];
            int64_t edge_bucket_size = edge_bucket_sizes_accessor[edge_bucket_id];
            int64_t edge_bucket_start = edge_bucket_starts_disk_accessor[edge_bucket_id];

            int idx = offset + i * num_swap_partitions + j;
            new_edge_bucket_ids_accessor[idx] = edge_bucket_id;
            new_edge_bucket_sizes_accessor[idx] = edge_bucket_size;
            new_global_edge_bucket_starts_accessor[idx] = edge_bucket_start;
        }
    }

    // concatenate old and new
    in_mem_edge_bucket_ids = torch::cat({in_mem_edge_bucket_ids, new_edge_bucket_ids});
    in_mem_edge_bucket_sizes = torch::cat({in_mem_edge_bucket_sizes, new_edge_bucket_sizes});
    local_or_global_edge_bucket_starts = torch::cat({local_or_global_edge_bucket_starts, new_global_edge_bucket_starts});

    torch::Tensor in_mem_mask = torch::ones({num_edge_buckets_in_mem - num_new_edge_buckets}, torch::kBool);
    in_mem_mask = torch::cat({in_mem_mask, torch::zeros({num_new_edge_buckets}, torch::kBool)});

    // put the ids in the correct order so the mapped edges remain sorted
    torch::Tensor src_ids_order = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);
    auto src_ids_order_accessor = src_ids_order.accessor<int64_t, 1>();

#pragma omp parallel for
    for (int i = 0; i < buffer_size; i++) {
        for (int j = 0; j < buffer_size; j++) {
            int64_t edge_bucket_id = new_in_mem_partition_ids_accessor[i] * num_partitions + new_in_mem_partition_ids_accessor[j];

            int idx = i * buffer_size + j;
            src_ids_order_accessor[idx] = edge_bucket_id;
        }
    }
    auto bucket_meta_pre_sort_end = std::chrono::steady_clock::now();
    auto sort_stage_start = bucket_meta_pre_sort_end;
    auto sort_argsort_start = sort_stage_start;

    // TODO: all these argsorts can be done with one omp for loop, probably faster, same with masked_selects above
    torch::Tensor arg_sort;
    {
        NvtxRangeGuard nvtx_bucket_reorder("subgraph.bucket_reorder_argsort");
        arg_sort = torch::argsort(in_mem_edge_bucket_ids);
        arg_sort = (arg_sort.index_select(0, torch::argsort(torch::argsort(src_ids_order))));
    }
    auto sort_argsort_end = std::chrono::steady_clock::now();
    auto sort_reorder_start = sort_argsort_end;
    {
        NvtxRangeGuard nvtx_bucket_reorder("subgraph.bucket_reorder_argsort");
        in_mem_edge_bucket_ids = (in_mem_edge_bucket_ids.index_select(0, arg_sort));
        in_mem_edge_bucket_sizes = (in_mem_edge_bucket_sizes.index_select(0, arg_sort));
        local_or_global_edge_bucket_starts = (local_or_global_edge_bucket_starts.index_select(0, arg_sort));
        in_mem_mask = (in_mem_mask.index_select(0, arg_sort));
    }
    auto sort_stage_end = std::chrono::steady_clock::now();
    auto bucket_meta_end = std::chrono::steady_clock::now();
    if (debug_subgraph) {
        auto bucket_meta_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(bucket_meta_pre_sort_end - bucket_meta_start).count();
        auto sort_argsort_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(sort_argsort_end - sort_argsort_start).count();
        auto sort_reorder_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(sort_stage_end - sort_reorder_start).count();
        auto sort_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(sort_stage_end - sort_stage_start).count();
        SPDLOG_INFO("SUBGRAPH call={} stage=bucket_meta new_edge_buckets={} total_edge_buckets={} ms={}",
                    subgraph_call_id,
                    num_new_edge_buckets,
                    in_mem_edge_bucket_ids.size(0),
                    bucket_meta_ms);
        SPDLOG_INFO("SUBGRAPH call={} stage=sort_reorder meta_ids={} argsort_ms={} reorder_ms={} total_ms={}",
                    subgraph_call_id,
                    in_mem_edge_bucket_ids.size(0),
                    sort_argsort_ms,
                    sort_reorder_ms,
                    sort_total_ms);
    }

    // with everything in order grab the edge buckets
    auto edge_materialize_start = std::chrono::steady_clock::now();
    torch::Tensor in_mem_edge_bucket_starts = in_mem_edge_bucket_sizes.cumsum(0);
    int64_t total_size = in_mem_edge_bucket_starts[-1].item<int64_t>();
    in_mem_edge_bucket_starts = in_mem_edge_bucket_starts - in_mem_edge_bucket_sizes;

    auto in_mem_edge_bucket_sizes_accessor = in_mem_edge_bucket_sizes.accessor<int64_t, 1>();
    auto local_or_global_edge_bucket_starts_accessor = local_or_global_edge_bucket_starts.accessor<int64_t, 1>();
    auto in_mem_mask_accessor = in_mem_mask.accessor<bool, 1>();
    auto in_mem_edge_bucket_starts_accessor = in_mem_edge_bucket_starts.accessor<int64_t, 1>();

    torch::Tensor new_all_in_memory_edges = torch::empty({total_size, storage_ptrs_.edges->dim1_size_}, torch::kInt64);

// get the edges
    {
        NvtxRangeGuard nvtx_edge_materialize("subgraph.edge_materialize_loop");
#pragma omp parallel for
        for (int i = 0; i < num_edge_buckets_in_mem; i++) {
            int64_t edge_bucket_size = in_mem_edge_bucket_sizes_accessor[i];
            int64_t edge_bucket_start = local_or_global_edge_bucket_starts_accessor[i];
            bool in_mem = in_mem_mask_accessor[i];
            int64_t local_offset = in_mem_edge_bucket_starts_accessor[i];

            if (in_mem) {
                new_all_in_memory_edges.narrow(0, local_offset, edge_bucket_size) =
                    current_subgraph_states_[device_idx]->all_in_memory_edges_.narrow(0, edge_bucket_start, edge_bucket_size);
            } else {
                new_all_in_memory_edges.narrow(0, local_offset, edge_bucket_size) = storage_ptrs_.edges->range(edge_bucket_start, edge_bucket_size);
            }
        }
    }

    subgraph->all_in_memory_edges_ = new_all_in_memory_edges;
    auto edge_materialize_end = std::chrono::steady_clock::now();
    if (debug_subgraph) {
        auto edge_materialize_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(edge_materialize_end - edge_materialize_start).count();
        SPDLOG_INFO("SUBGRAPH call={} stage=edge_materialize total_edges={} edge_width={} ms={}",
                    subgraph_call_id,
                    total_size,
                    storage_ptrs_.edges->dim1_size_,
                    edge_materialize_ms);
    }

    auto map_graph_start = std::chrono::steady_clock::now();
    if (storage_ptrs_.node_embeddings != nullptr) {
        if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            subgraph->global_to_local_index_map_ =
                std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->getGlobalToLocalMap(!prefetch_);
        }
        else if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            subgraph->global_to_local_index_map_ =
                std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->getGlobalToLocalMap(!prefetch_, device_idx);
        }
    } else if (storage_ptrs_.node_features != nullptr) {
        subgraph->global_to_local_index_map_ = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->getGlobalToLocalMap(!prefetch_);
    }

    torch::Tensor mapped_edges;
    torch::Tensor mapped_edges_dst_sort;
    {
        NvtxRangeGuard nvtx_map("subgraph.global_to_local_index_select");
        if (storage_ptrs_.edges->dim1_size_ == 3) {
            mapped_edges = torch::stack({subgraph->global_to_local_index_map_.index_select(0, subgraph->all_in_memory_edges_.select(1, 0)),
                                         subgraph->all_in_memory_edges_.select(1, 1),
                                         subgraph->global_to_local_index_map_.index_select(0, subgraph->all_in_memory_edges_.select(1, -1))})
                               .transpose(0, 1);
            // mapped_edges = mapped_edges.to(devices_[device_idx]);
        } else if (storage_ptrs_.edges->dim1_size_ == 2) {
            mapped_edges = torch::stack({subgraph->global_to_local_index_map_.index_select(0, subgraph->all_in_memory_edges_.select(1, 0)),
                                         subgraph->global_to_local_index_map_.index_select(0, subgraph->all_in_memory_edges_.select(1, -1))})
                               .transpose(0, 1);
            // mapped_edges = mapped_edges.to(devices_[device_idx]);
        } else {
            // TODO use a function for logging errors and throwing expections
            SPDLOG_ERROR("Unexpected number of edge columns");
            std::runtime_error("Unexpected number of edge columns");
        }
    }

    mapped_edges = mapped_edges.to(devices_[device_idx]);
    subgraph->all_in_memory_mapped_edges_ = mapped_edges;
    subgraph->in_memory_subgraph_ = nullptr;
    if (devices_.size() <= 1) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    mapped_edges = merge_sorted_edge_buckets(mapped_edges, in_mem_edge_bucket_starts, buffer_size, true);
    mapped_edges_dst_sort = merge_sorted_edge_buckets(mapped_edges, in_mem_edge_bucket_starts, buffer_size, false);
    
    if (devices_.size() <= 1) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    mapped_edges = mapped_edges.to(torch::kInt64);
    mapped_edges_dst_sort = mapped_edges_dst_sort.to(torch::kInt64);

    if (subgraph->in_memory_subgraph_ != nullptr) {
        subgraph->in_memory_subgraph_ = nullptr;
    }

    subgraph->in_memory_subgraph_ = std::make_shared<GegeGraph>(mapped_edges, mapped_edges_dst_sort, getNumNodesInMemory(device_idx));
    auto map_graph_end = std::chrono::steady_clock::now();
    if (debug_subgraph) {
        auto map_graph_ms = std::chrono::duration_cast<std::chrono::milliseconds>(map_graph_end - map_graph_start).count();
        SPDLOG_INFO("SUBGRAPH call={} stage=map_graph mapped_edges_rows={} mapped_edges_cols={} dst_sort_rows={} ms={}",
                    subgraph_call_id,
                    mapped_edges.size(0),
                    mapped_edges.size(1),
                    mapped_edges_dst_sort.size(0),
                    map_graph_ms);
    }

    // update state
    subgraph->in_memory_partition_ids_ = new_in_mem_partition_ids;
    subgraph->in_memory_edge_bucket_ids_ = in_mem_edge_bucket_ids;
    subgraph->in_memory_edge_bucket_sizes_ = in_mem_edge_bucket_sizes;
    subgraph->in_memory_edge_bucket_starts_ = in_mem_edge_bucket_starts;

    if (prefetch_) {
        prefetch_complete_ = true;
        subgraph_lock_->unlock();
        subgraph_cv_->notify_all();
    }

    if (debug_subgraph) {
        auto total_end = std::chrono::steady_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
        SPDLOG_INFO("SUBGRAPH call={} device={} done total_ms={} final_edge_buckets={} final_edges_rows={} final_edges_cols={}",
                    subgraph_call_id,
                    device_idx,
                    total_ms,
                    in_mem_edge_bucket_ids.size(0),
                    subgraph->all_in_memory_edges_.size(0),
                    subgraph->all_in_memory_edges_.size(1));
    }

    if (profile_timing_enabled()) {
        auto total_end = std::chrono::steady_clock::now();
        auto bucket_partition_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(bucket_partition_end - bucket_partition_start).count();
        auto bucket_meta_ms = std::chrono::duration_cast<std::chrono::milliseconds>(bucket_meta_end - bucket_meta_start).count();
        auto edge_materialize_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(edge_materialize_end - edge_materialize_start).count();
        auto map_graph_ms = std::chrono::duration_cast<std::chrono::milliseconds>(map_graph_end - map_graph_start).count();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
        SPDLOG_INFO(
            "Timing updateInMemorySubGraph_: call={} device={} buckets_partition={}ms buckets_meta={}ms edge_materialize={}ms map_and_graph={}ms total={}ms",
            subgraph_call_id,
            device_idx,
            bucket_partition_ms,
            bucket_meta_ms,
            edge_materialize_ms,
            map_graph_ms,
            total_ms);
    }
}

EdgeList GraphModelStorage::merge_sorted_edge_buckets(EdgeList edges, torch::Tensor starts, int buffer_size, bool src) {
    NvtxRangeGuard nvtx_merge(src ? "subgraph.merge_sorted_edge_buckets.src" : "subgraph.merge_sorted_edge_buckets.dst");
    int sort_dim = 0;
    if (!src) {
        sort_dim = -1;
    }
    torch::Tensor result;
    {
        auto index = torch::argsort(edges.select(1, sort_dim));
        result = edges.index_select(0, index);
    }
    return result;
}

void GraphModelStorage::sortAllEdges(int32_t device_idx) {
    if (!useInMemorySubGraph()) {

        std::vector<EdgeList> additional_edges = {};

        if (storage_ptrs_.train_edges != nullptr) {
            storage_ptrs_.train_edges->load();
            additional_edges.emplace_back(storage_ptrs_.train_edges->range(0, storage_ptrs_.train_edges->getDim0()));
        }

        if (!train_) {
            if (storage_ptrs_.validation_edges != nullptr) {
                storage_ptrs_.validation_edges->load();
                additional_edges.emplace_back(storage_ptrs_.validation_edges->range(0, storage_ptrs_.validation_edges->getDim0()));
            }

            if (storage_ptrs_.test_edges != nullptr) {

                storage_ptrs_.test_edges->load();
                additional_edges.emplace_back(storage_ptrs_.test_edges->range(0, storage_ptrs_.test_edges->getDim0()));
            }
        } 

        for (auto f_edges : storage_ptrs_.filter_edges) {
            f_edges->load();
            additional_edges.emplace_back(f_edges->range(0, f_edges->getDim0()));
        }

        current_subgraph_states_[0]->in_memory_subgraph_->sortAllEdges(torch::cat(additional_edges));

        for (auto f_edges : storage_ptrs_.filter_edges) {
            f_edges->unload();
        }

    } else {
        current_subgraph_states_[device_idx]->in_memory_subgraph_->sortAllEdges(current_subgraph_states_[device_idx]->in_memory_subgraph_->src_sorted_edges_);
    }
}
