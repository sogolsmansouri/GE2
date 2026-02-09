#include "storage/graph_storage.h"
#include <c10/cuda/CUDACachingAllocator.h>

#include <algorithm>
#include <random>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>

#include "configuration/util.h"
#include "data/ordering.h"
#include "reporting/logger.h"

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
            auto t1 = std::chrono::high_resolution_clock::now();
            performSwap(0);
            auto t2 = std::chrono::high_resolution_clock::now();
            if (devices_.size() <= 1) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

            for (int d = 0; d < devices_.size(); ++d) {
                updateInMemorySubGraph_(current_subgraph_states_[d], swap_ids_per_device[d], d);
            }

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
            return;
        }
std::pair<std::vector<int>, std::vector<int>> current_swap_ids = getNextSwapIds(device_idx);
        // SPDLOG_INFO("performSwap");
        auto t1 = std::chrono::high_resolution_clock::now();
        performSwap(device_idx);
        auto t2 = std::chrono::high_resolution_clock::now();
        // SPDLOG_INFO("performSwap time {}", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
        if (devices_.size() <= 1) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
        // SPDLOG_INFO("updateInMemorySubGraph_");
        t1 = std::chrono::high_resolution_clock::now();
        updateInMemorySubGraph_(current_subgraph_states_[device_idx], current_swap_ids, device_idx);
        t2 = std::chrono::high_resolution_clock::now();
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
    next_subgraph_state_ = std::make_shared<InMemorySubgraphState>();
    next_subgraph_state_->in_memory_subgraph_ = nullptr;
    std::thread(&GraphModelStorage::updateInMemorySubGraph_, this, next_subgraph_state_, next_swap_ids, 0).detach();
}

void GraphModelStorage::updateInMemorySubGraph_(shared_ptr<InMemorySubgraphState> subgraph, std::pair<std::vector<int>, std::vector<int>> swap_ids, int32_t device_idx) {
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

    // get edge buckets that will be kept in memory
    torch::Tensor keep_mask = torch::ones({num_edge_buckets_in_mem}, torch::kBool);
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

    torch::Tensor in_mem_edge_bucket_ids = current_subgraph_states_[device_idx]->in_memory_edge_bucket_ids_.masked_select(keep_mask);
    torch::Tensor in_mem_edge_bucket_sizes = current_subgraph_states_[device_idx]->in_memory_edge_bucket_sizes_.masked_select(keep_mask);
    torch::Tensor local_or_global_edge_bucket_starts = current_subgraph_states_[device_idx]->in_memory_edge_bucket_starts_.masked_select(keep_mask);

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

    torch::Tensor old_in_mem_partition_ids = current_subgraph_states_[device_idx]->in_memory_partition_ids_.masked_select(keep_mask);
    torch::Tensor new_in_mem_partition_ids = current_subgraph_states_[device_idx]->in_memory_partition_ids_.masked_scatter(~keep_mask, admit_ids_tensor);
    auto old_in_mem_partition_ids_accessor = old_in_mem_partition_ids.accessor<int64_t, 1>();
    auto new_in_mem_partition_ids_accessor = new_in_mem_partition_ids.accessor<int64_t, 1>();

    // get new incoming edge buckets
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

    // TODO: all these argsorts can be done with one omp for loop, probably faster, same with masked_selects above
    torch::Tensor arg_sort = torch::argsort(in_mem_edge_bucket_ids);
    arg_sort = (arg_sort.index_select(0, torch::argsort(torch::argsort(src_ids_order))));
    in_mem_edge_bucket_ids = (in_mem_edge_bucket_ids.index_select(0, arg_sort));
    in_mem_edge_bucket_sizes = (in_mem_edge_bucket_sizes.index_select(0, arg_sort));
    local_or_global_edge_bucket_starts = (local_or_global_edge_bucket_starts.index_select(0, arg_sort));
    in_mem_mask = (in_mem_mask.index_select(0, arg_sort));

    // with everything in order grab the edge buckets
    torch::Tensor in_mem_edge_bucket_starts = in_mem_edge_bucket_sizes.cumsum(0);
    int64_t total_size = in_mem_edge_bucket_starts[-1].item<int64_t>();
    in_mem_edge_bucket_starts = in_mem_edge_bucket_starts - in_mem_edge_bucket_sizes;

    auto in_mem_edge_bucket_sizes_accessor = in_mem_edge_bucket_sizes.accessor<int64_t, 1>();
    auto local_or_global_edge_bucket_starts_accessor = local_or_global_edge_bucket_starts.accessor<int64_t, 1>();
    auto in_mem_mask_accessor = in_mem_mask.accessor<bool, 1>();
    auto in_mem_edge_bucket_starts_accessor = in_mem_edge_bucket_starts.accessor<int64_t, 1>();

    torch::Tensor new_all_in_memory_edges = torch::empty({total_size, storage_ptrs_.edges->dim1_size_}, torch::kInt64);

// get the edges
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

    subgraph->all_in_memory_edges_ = new_all_in_memory_edges;

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
}

EdgeList GraphModelStorage::merge_sorted_edge_buckets(EdgeList edges, torch::Tensor starts, int buffer_size, bool src) {
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
