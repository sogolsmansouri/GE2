#include "data/dataloader.h"

#include "common/util.h"
#include "data/ordering.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <algorithm>
#include <sstream>

DataLoader::DataLoader(shared_ptr<GraphModelStorage> graph_storage, LearningTask learning_task, shared_ptr<TrainingConfig> training_config,
                       shared_ptr<EvaluationConfig> evaluation_config, shared_ptr<EncoderConfig> encoder_config, vector<torch::Device> devices,
                       NegativeSamplingMethod nsm) {
    current_edge_ = 0;
    train_ = true;
    epochs_processed_ = 0;
    batches_processed_ = 0;
    false_negative_edges = 0;
    swap_tasks_completed = 0;
    loaded_subgraphs = 0;
    sampler_lock_ = new std::mutex();
    batch_lock_ = new std::mutex;
    batch_cv_ = new std::condition_variable;
    waiting_for_batches_ = false;

    single_dataset_ = false;
    async_barrier = 0;
    graph_storage_ = graph_storage;
    learning_task_ = learning_task;
    training_config_ = training_config;
    evaluation_config_ = evaluation_config;
    only_root_features_ = false;

    edge_sampler_ = std::make_shared<RandomEdgeSampler>(graph_storage_);

    devices_ = devices;
    activate_devices_ = 0;
    transfer_plan_num_devices_ = static_cast<int>(devices_.size());
    transfer_plan_mock_mode_ = false;

    negative_sampling_method_ = nsm;

    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        if (negative_sampling_method_ == NegativeSamplingMethod::RNS) {
            training_negative_sampler_ = std::make_shared<RNS>(
                training_config_->negative_sampling->num_chunks, training_config_->negative_sampling->negatives_per_positive,
                training_config_->negative_sampling->degree_fraction, training_config_->negative_sampling->filtered,
                training_config_->negative_sampling->local_filter_mode);
        } else if (negative_sampling_method_ == NegativeSamplingMethod::DNS) {
            training_negative_sampler_ = std::make_shared<DNS>(
                training_config_->negative_sampling->num_chunks, training_config_->negative_sampling->negatives_per_positive,
                training_config_->negative_sampling->degree_fraction, training_config_->negative_sampling->filtered,
                training_config_->negative_sampling->local_filter_mode);
        } else if (negative_sampling_method_ == NegativeSamplingMethod::GAN) {
            training_negative_sampler_ = std::make_shared<KBGAN>(
                training_config_->negative_sampling->num_chunks, training_config_->negative_sampling->negatives_per_positive,
                training_config_->negative_sampling->degree_fraction, training_config_->negative_sampling->filtered,
                training_config_->negative_sampling->local_filter_mode);
        } else {
            SPDLOG_INFO("NegativeSampling: not supported");
        }
        evaluation_negative_sampler_ = std::make_shared<CorruptNodeNegativeSampler>(
            evaluation_config_->negative_sampling->num_chunks, evaluation_config_->negative_sampling->negatives_per_positive,
            evaluation_config_->negative_sampling->degree_fraction, evaluation_config_->negative_sampling->filtered,
            evaluation_config_->negative_sampling->local_filter_mode);
    } else {
        training_negative_sampler_ = nullptr;
        evaluation_negative_sampler_ = nullptr;
    }

    if (encoder_config != nullptr) {
        if (!encoder_config->train_neighbor_sampling.empty()) {
            training_neighbor_sampler_ = std::make_shared<LayeredNeighborSampler>(graph_storage_, encoder_config->train_neighbor_sampling);

            if (!encoder_config->eval_neighbor_sampling.empty()) {
                evaluation_neighbor_sampler_ = std::make_shared<LayeredNeighborSampler>(graph_storage_, encoder_config->eval_neighbor_sampling);
            } else {
                evaluation_neighbor_sampler_ = training_neighbor_sampler_;
            }

        } else {
            training_neighbor_sampler_ = nullptr;
            evaluation_neighbor_sampler_ = nullptr;
        }
    } else {
        training_neighbor_sampler_ = nullptr;
        evaluation_neighbor_sampler_ = nullptr;
    }
}

DataLoader::DataLoader(shared_ptr<GraphModelStorage> graph_storage, LearningTask learning_task, int batch_size, shared_ptr<NegativeSampler> negative_sampler,
                       shared_ptr<NeighborSampler> neighbor_sampler, bool train) {
    current_edge_ = 0;
    train_ = train;
    epochs_processed_ = 0;
    false_negative_edges = 0;
    batches_processed_ = 0;
    loaded_subgraphs = 0;
    sampler_lock_ = new std::mutex();
    batch_lock_ = new std::mutex;
    batch_cv_ = new std::condition_variable;
    async_barrier = 0;
    waiting_for_batches_ = false;

    batch_size_ = batch_size;
    single_dataset_ = true;

    graph_storage_ = graph_storage;
    learning_task_ = learning_task;
    only_root_features_ = false;

    edge_sampler_ = std::make_shared<RandomEdgeSampler>(graph_storage_);
    negative_sampler_ = negative_sampler;
    neighbor_sampler_ = neighbor_sampler;

    training_negative_sampler_ = nullptr;
    evaluation_negative_sampler_ = nullptr;

    training_neighbor_sampler_ = nullptr;
    evaluation_neighbor_sampler_ = nullptr;

    transfer_plan_num_devices_ = 0;
    transfer_plan_mock_mode_ = false;

    loadStorage();
}

DataLoader::~DataLoader() {
    delete sampler_lock_;
    delete batch_lock_;
    delete batch_cv_;
}

void DataLoader::nextEpoch() {
    batch_id_offset_ = 0;
    total_batches_processed_ = 0;
    epochs_processed_++;
    loaded_subgraphs = 0;
    buffer_states_.clear();
    if (graph_storage_->useInMemorySubGraph()) {
        unloadStorage();
    }
}

void DataLoader::setActiveEdges(int32_t device_idx) {
    auto state = graph_storage_->current_subgraph_states_[device_idx];
    if (state != nullptr && state->all_in_memory_edges_.defined()) {
        SPDLOG_INFO("setActiveEdges: device={} in_mem_edges size={}x{}",
                    device_idx,
                    state->all_in_memory_edges_.size(0),
                    state->all_in_memory_edges_.size(1));
    } else {
        SPDLOG_INFO("setActiveEdges: device={} in_mem_edges undefined", device_idx);
    }

    EdgeList active_edges;

    if (graph_storage_->useInMemorySubGraph()) {
        torch::Tensor edge_bucket_ids = *edge_buckets_per_buffer_iterators_[device_idx];
        for (int i = 0; i < devices_.size(); i++) {
            if (edge_buckets_per_buffer_iterators_[device_idx] != edge_buckets_per_buffer_.end()) {
                edge_buckets_per_buffer_iterators_[device_idx]++;
            }
        }

        int num_partitions = graph_storage_->getNumPartitions();
        edge_bucket_ids = edge_bucket_ids.select(1, 0) * num_partitions + edge_bucket_ids.select(1, 1);
        torch::Tensor in_memory_edge_bucket_idx = torch::empty({edge_bucket_ids.size(0)}, edge_bucket_ids.options());
        torch::Tensor edge_bucket_sizes = torch::empty({edge_bucket_ids.size(0)}, edge_bucket_ids.options());

        auto edge_bucket_ids_accessor = edge_bucket_ids.accessor<int64_t, 1>();
        auto in_memory_edge_bucket_idx_accessor = in_memory_edge_bucket_idx.accessor<int64_t, 1>();
        auto edge_bucket_sizes_accessor = edge_bucket_sizes.accessor<int64_t, 1>();
        auto all_edge_bucket_sizes_accessor = graph_storage_->current_subgraph_states_[device_idx]->in_memory_edge_bucket_sizes_.accessor<int64_t, 1>();
        auto all_edge_bucket_starts_accessor = graph_storage_->current_subgraph_states_[device_idx]->in_memory_edge_bucket_starts_.accessor<int64_t, 1>();
        
        auto tup = torch::sort(graph_storage_->current_subgraph_states_[device_idx]->in_memory_edge_bucket_ids_);
        torch::Tensor sorted_in_memory_ids = std::get<0>(tup);
        torch::Tensor in_memory_id_indices = std::get<1>(tup);
        auto in_memory_id_indices_accessor = in_memory_id_indices.accessor<int64_t, 1>();

#pragma omp parallel for
        for (int i = 0; i < in_memory_edge_bucket_idx.size(0); i++) {
            int64_t edge_bucket_id = edge_bucket_ids_accessor[i];
            int64_t idx = torch::searchsorted(sorted_in_memory_ids, edge_bucket_id).item<int64_t>();
            idx = in_memory_id_indices_accessor[idx];
            int64_t edge_bucket_size = all_edge_bucket_sizes_accessor[idx];

            in_memory_edge_bucket_idx_accessor[i] = idx;
            edge_bucket_sizes_accessor[i] = edge_bucket_size;
        }

        torch::Tensor local_offsets = edge_bucket_sizes.cumsum(0);
        int64_t total_size = local_offsets[-1].item<int64_t>();
        local_offsets = local_offsets - edge_bucket_sizes;

        auto local_offsets_accessor = local_offsets.accessor<int64_t, 1>();

        active_edges = torch::empty({total_size, graph_storage_->storage_ptrs_.edges->dim1_size_},
                                    graph_storage_->current_subgraph_states_[device_idx]->all_in_memory_mapped_edges_.options());

#pragma omp parallel for
        for (int i = 0; i < in_memory_edge_bucket_idx.size(0); i++) {
            int64_t idx = in_memory_edge_bucket_idx_accessor[i];
            int64_t edge_bucket_size = edge_bucket_sizes_accessor[i];
            int64_t edge_bucket_start = all_edge_bucket_starts_accessor[idx];
            int64_t local_offset = local_offsets_accessor[i];

            active_edges.narrow(0, local_offset, edge_bucket_size) =
                graph_storage_->current_subgraph_states_[device_idx]->all_in_memory_mapped_edges_.narrow(0, edge_bucket_start, edge_bucket_size);
        }

    } else {
        active_edges = graph_storage_->storage_ptrs_.edges->range(0, graph_storage_->storage_ptrs_.edges->getDim0());
    }

    // Defensive guard: if subgraph mapping produced invalid endpoints, drop those
    // rows before batching so we do not propagate -1 node ids downstream.
    if (active_edges.defined() && active_edges.size(0) > 0 && active_edges.size(1) >= 2) {
        torch::Tensor invalid_src = active_edges.select(1, 0).lt(0);
        torch::Tensor invalid_dst = active_edges.select(1, -1).lt(0);
        torch::Tensor invalid_mask = invalid_src.logical_or(invalid_dst);
        int64_t invalid_edges = invalid_mask.sum().item<int64_t>();
        if (invalid_edges > 0) {
            torch::Tensor valid_idx = torch::nonzero(invalid_mask.logical_not()).flatten(0, 1);
            SPDLOG_WARN("setActiveEdges: device={} dropping {} / {} edges with unmapped endpoints", device_idx, invalid_edges, active_edges.size(0));
            if (valid_idx.numel() == 0) {
                throw std::runtime_error("No valid mapped edges remain in active subgraph");
            }
            active_edges = active_edges.index_select(0, valid_idx);
        }
    }

    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto perm = torch::randperm(active_edges.size(0), opts);
    perm = perm.to(active_edges.device());
    active_edges = (active_edges.index_select(0, perm));
    perm = torch::Tensor();
    graph_storage_->setActiveEdges(active_edges, device_idx);
}

void DataLoader::setActiveNodes() {
    torch::Tensor node_ids;

    if (graph_storage_->useInMemorySubGraph()) {
        node_ids = *node_ids_per_buffer_iterator_++;
    } else {
        node_ids = graph_storage_->storage_ptrs_.nodes->range(0, graph_storage_->storage_ptrs_.nodes->getDim0());
        if (node_ids.sizes().size() == 2) {
            node_ids = node_ids.flatten(0, 1);
        }
    }

    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(node_ids.device());
    node_ids = (node_ids.index_select(0, torch::randperm(node_ids.size(0), opts)));
    graph_storage_->setActiveNodes(node_ids);
}

void DataLoader::initializeBatches(bool prepare_encode, int32_t device_idx) {
    int64_t batch_id = 0;
    int64_t start_idx = 0;

    int64_t num_items;
    if (prepare_encode) {
        num_items = graph_storage_->getNumNodes();
    } else {
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            setActiveEdges(device_idx);
            num_items = graph_storage_->getNumActiveEdges(device_idx);
        } else {
            setActiveNodes();
            num_items = graph_storage_->getNumActiveNodes();
        }
    }

    int64_t batch_size = batch_size_;
    vector<shared_ptr<Batch>> batches;
    while (start_idx < num_items) {
        if (num_items - (start_idx + batch_size) < 0) {
            batch_size = num_items - start_idx;
        }
        shared_ptr<Batch> curr_batch = std::make_shared<Batch>(train_);
        curr_batch->batch_id_ = batch_id + batch_id_offset_;
        curr_batch->start_idx_ = start_idx;
        curr_batch->batch_size_ = batch_size;

        if (prepare_encode) {
            curr_batch->task_ = LearningTask::ENCODE;
        } else {
            curr_batch->task_ = learning_task_;
        }

        batches.emplace_back(curr_batch);
        batch_id++;
        start_idx += batch_size;
    }
    all_batches_[device_idx] = batches;
    batches_left_[device_idx] = batches.size();
    batch_iterators_[device_idx] = all_batches_[device_idx].begin();
}

void DataLoader::setBufferOrdering() {
    shared_ptr<PartitionBufferOptions> options;
    if (instance_of<Storage, PartitionBufferStorage>(graph_storage_->storage_ptrs_.node_embeddings)) {
        options = std::dynamic_pointer_cast<PartitionBufferStorage>(graph_storage_->storage_ptrs_.node_embeddings)->options_;
    } else if (instance_of<Storage, MemPartitionBufferStorage>(graph_storage_->storage_ptrs_.node_embeddings)) {
        options = std::dynamic_pointer_cast<MemPartitionBufferStorage>(graph_storage_->storage_ptrs_.node_embeddings)->options_;
    } else if (instance_of<Storage, PartitionBufferStorage>(graph_storage_->storage_ptrs_.node_features)) {
        options = std::dynamic_pointer_cast<PartitionBufferStorage>(graph_storage_->storage_ptrs_.node_features)->options_;
    }

    transfer_plan_num_devices_ = static_cast<int>(devices_.size());
    transfer_plan_mock_mode_ = false;
    if (options != nullptr && devices_.size() == 1 && options->mock_num_devices_for_schedule > 1) {
        transfer_plan_num_devices_ = options->mock_num_devices_for_schedule;
        transfer_plan_mock_mode_ = true;
    }

    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        if (graph_storage_->useInMemorySubGraph()) {
            auto tup = getEdgeBucketOrdering(options->edge_bucket_ordering, options->num_partitions, options->buffer_capacity, options->fine_to_coarse_ratio,
                                             options->num_cache_partitions, options->randomly_assign_edge_buckets,
                                             options->randomize_partition_ids, options->shuffle_buffer_states);
            buffer_states_ = std::get<0>(tup);
            SPDLOG_INFO("buffer_states_ sizes() {}", buffer_states_.size());
            edge_buckets_per_buffer_ = std::get<1>(tup);
            auto edge_buckets_per_buffer_iterator = edge_buckets_per_buffer_.begin();
            edge_buckets_per_buffer_iterators_.resize(devices_.size());
            for (int i = 0; i < devices_.size(); i ++) {
                edge_buckets_per_buffer_iterators_[i] = edge_buckets_per_buffer_iterator;
                edge_buckets_per_buffer_iterator++;
            }
            graph_storage_->setBufferOrdering(buffer_states_);
            buildTransferPlan();
        }
    } else {
        if (graph_storage_->useInMemorySubGraph()) {
            graph_storage_->storage_ptrs_.train_nodes->load();
            int64_t num_train_nodes = graph_storage_->storage_ptrs_.nodes->getDim0();
            auto tup = getNodePartitionOrdering(
                options->node_partition_ordering, graph_storage_->storage_ptrs_.train_nodes->range(0, num_train_nodes).flatten(0, 1),
                graph_storage_->getNumNodes(), options->num_partitions, options->buffer_capacity, options->fine_to_coarse_ratio, options->num_cache_partitions);
            buffer_states_ = std::get<0>(tup);
            node_ids_per_buffer_ = std::get<1>(tup);

            node_ids_per_buffer_iterator_ = node_ids_per_buffer_.begin();

            graph_storage_->setBufferOrdering(buffer_states_);
            buildTransferPlan();
        }
    }
}

void DataLoader::buildTransferPlan() {
    transfer_plan_per_round_.clear();
    transfer_round_logged_.clear();
    device_swap_rounds_.assign(devices_.size(), 0);

    if (buffer_states_.empty() || devices_.empty()) {
        return;
    }

    int num_devices = transfer_plan_num_devices_ > 0 ? transfer_plan_num_devices_ : static_cast<int>(devices_.size());
    if (num_devices <= 0) {
        return;
    }
    if (buffer_states_.size() % num_devices != 0) {
        SPDLOG_WARN("Cannot build transfer plan: {} buffer states not divisible by {} devices", buffer_states_.size(), num_devices);
        return;
    }

    int num_rounds = static_cast<int>(buffer_states_.size() / num_devices);
    transfer_plan_per_round_.assign(num_rounds, {});
    transfer_round_logged_.assign(num_rounds, false);

    if (num_devices == 1) {
        SPDLOG_INFO("Transfer-plan sanity: single GPU mode, no inter-GPU migrations ({} rounds)", num_rounds);
        return;
    }

    if (transfer_plan_mock_mode_) {
        SPDLOG_INFO("Transfer-plan sanity: mock {}-GPU schedule on 1 physical GPU (planner only)", num_devices);
    }

    std::vector<std::vector<std::vector<int>>> round_states(num_rounds, std::vector<std::vector<int>>(num_devices));
    int max_partition_id = -1;
    for (int round_id = 0; round_id < num_rounds; round_id++) {
        for (int device_idx = 0; device_idx < num_devices; device_idx++) {
            int state_idx = round_id * num_devices + device_idx;
            torch::Tensor state = buffer_states_[state_idx].to(torch::kCPU).to(torch::kInt64);
            auto state_accessor = state.accessor<int64_t, 1>();
            for (int i = 0; i < state.size(0); i++) {
                int part_id = static_cast<int>(state_accessor[i]);
                round_states[round_id][device_idx].emplace_back(part_id);
                max_partition_id = std::max(max_partition_id, part_id);
            }
        }
    }

    if (max_partition_id < 0) {
        SPDLOG_WARN("Cannot build transfer plan: no partitions found in buffer states");
        return;
    }

    std::vector<int> owner(max_partition_id + 1, -1);
    for (int device_idx = 0; device_idx < num_devices; device_idx++) {
        for (auto part_id : round_states[0][device_idx]) {
            if (owner[part_id] != -1 && owner[part_id] != device_idx) {
                SPDLOG_WARN("Partition {} appears on multiple devices in round 0: {} and {}", part_id, owner[part_id], device_idx);
            }
            owner[part_id] = device_idx;
        }
    }

    for (int round_id = 1; round_id < num_rounds; round_id++) {
        for (int device_idx = 0; device_idx < num_devices; device_idx++) {
            for (auto part_id : round_states[round_id][device_idx]) {
                int src_gpu = (part_id < owner.size()) ? owner[part_id] : -1;
                if (src_gpu == -1) {
                    SPDLOG_WARN("Round {} partition {} has unknown owner in transfer-plan build", round_id, part_id);
                    continue;
                }
                if (src_gpu != device_idx) {
                    transfer_plan_per_round_[round_id].push_back({round_id, part_id, src_gpu, device_idx});
                }
            }
        }
        // apply ownership updates for next round planning
        for (int device_idx = 0; device_idx < num_devices; device_idx++) {
            for (auto part_id : round_states[round_id][device_idx]) {
                owner[part_id] = device_idx;
            }
        }
    }

    SPDLOG_INFO("Transfer-plan sanity: {} rounds, {} devices", num_rounds, num_devices);
    for (int round_id = 0; round_id < num_rounds; round_id++) {
        std::ostringstream round_ss;
        for (int device_idx = 0; device_idx < num_devices; device_idx++) {
            if (device_idx != 0) {
                round_ss << " | ";
            }
            round_ss << "gpu" << device_idx << ":";
            round_ss << "[";
            for (size_t i = 0; i < round_states[round_id][device_idx].size(); i++) {
                if (i != 0) {
                    round_ss << ",";
                }
                round_ss << "p" << round_states[round_id][device_idx][i];
            }
            round_ss << "]";
        }
        SPDLOG_INFO("Round {} ownership {}", round_id, round_ss.str());
    }
    for (int round_id = 1; round_id < num_rounds; round_id++) {
        SPDLOG_INFO("Round {} planned inter-GPU partition migrations: {}", round_id, transfer_plan_per_round_[round_id].size());
        if (transfer_plan_mock_mode_) {
            std::ostringstream ops_ss;
            for (size_t i = 0; i < transfer_plan_per_round_[round_id].size(); i++) {
                const auto &op = transfer_plan_per_round_[round_id][i];
                if (i != 0) {
                    ops_ss << ", ";
                }
                ops_ss << "p" << op.partition_id << ":" << op.src_gpu << "->" << op.dst_gpu;
            }
            SPDLOG_INFO("Round {} mock transfer ops {}", round_id, ops_ss.str());
        }
    }
}

void DataLoader::logTransferPlanRound(int round_id) {
    if (round_id < 0 || round_id >= transfer_plan_per_round_.size()) {
        return;
    }
    if (transfer_round_logged_[round_id]) {
        return;
    }
    transfer_round_logged_[round_id] = true;

    const auto &ops = transfer_plan_per_round_[round_id];
    if (ops.empty()) {
        SPDLOG_INFO("Round {} transfer plan: no inter-GPU migrations", round_id);
        return;
    }

    std::ostringstream oss;
    for (size_t i = 0; i < ops.size(); i++) {
        const auto &op = ops[i];
        if (i != 0) {
            oss << ", ";
        }
        oss << "p" << op.partition_id << ":" << op.src_gpu << "->" << op.dst_gpu;
    }
    SPDLOG_INFO("Round {} transfer ops (embedding + optimizer-state): {}", round_id, oss.str());
}

void DataLoader::clearBatches() { all_batches_ = std::vector<std::vector<shared_ptr<Batch>>>(); }

shared_ptr<Batch> DataLoader::getNextBatch(int32_t device_idx) {
    // std::unique_lock batch_lock(*batch_lock_);
    // // batch_cv_->wait(batch_lock, [this] { return !waiting_for_batches_; });

    shared_ptr<Batch> batch;
    if (batch_iterators_[device_idx] != all_batches_[device_idx].end()) {
        batch = *batch_iterators_[device_idx];
        batch_iterators_[device_idx]++;

        // all_reads_[device_idx] = true;

        // check if all batches have been read
        if (batch_iterators_[device_idx] == all_batches_[device_idx].end()) {
            ++ async_barrier;
            // all_reads_[device_idx] = true;
            if (graph_storage_->useInMemorySubGraph()) {
                if (!graph_storage_->hasSwap(device_idx)) {
                    all_reads_[device_idx] = true;
                }
            } else {
                all_reads_[device_idx] = true;
            }
        }
    } else {
        batch = nullptr;
        if (graph_storage_->useInMemorySubGraph()) {
            if (graph_storage_->hasSwap(device_idx)) {
                // Training uses one consumer per configured device; synchronous evaluation consumes only device 0.
                int swap_participants = train_ ? static_cast<int>(all_batches_.size()) : 1;
                if (swap_participants <= 0) {
                    swap_participants = 1;
                }

                // wait for all batches to finish before swapping
                if (swap_tasks_completed == swap_participants) {
                    swap_tasks_completed = 0;
                }
                // batch_cv_->wait(batch_lock, [this, device_idx] { 
                //     return async_barrier.load() % all_batches_.size() == 0; });
                // batch_lock.unlock();
                while(async_barrier.load() % swap_participants != 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }

                // ensure all devices finished the current round before swapping
                while (activate_devices_.load() > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                int next_round_id = 0;
                if (device_idx < device_swap_rounds_.size()) {
                    next_round_id = device_swap_rounds_[device_idx] + 1;
                }
                // Log one round coordinator view for sanity checking swap planning.
                if (device_idx == 0 && devices_.size() > 1) {
                    logTransferPlanRound(next_round_id);
                }

                if (devices_.size() <= 1) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
                // SPDLOG_INFO("Swapping subgraph for device {}", device_idx);
                // auto t1 = std::chrono::high_resolution_clock::now();
                graph_storage_->updateInMemorySubGraph(device_idx);
                // SPDLOG_INFO("graph_storage_->updateInMemorySubGraph");

                if (devices_.size() <= 1) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
                // auto t11 = std::chrono::high_resolution_clock::now();
                // SPDLOG_INFO("Time to updateInMemorySubGraph for device {}: {} ms", device_idx, std::chrono::duration_cast<std::chrono::milliseconds>(t11 - t1).count());
                initializeBatches(false, device_idx);
                if (devices_.size() <= 1) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

                swap_tasks_completed ++;
                activate_devices_ ++;
                if (device_idx < device_swap_rounds_.size()) {
                    device_swap_rounds_[device_idx] = next_round_id;
                }

                while(swap_tasks_completed.load() != swap_participants) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                // auto t2 = std::chrono::high_resolution_clock::now();
                // SPDLOG_INFO("Finished swapping subgraph for device {}", device_idx);
                // SPDLOG_INFO("Time to swap subgraph for device {}: {} ms", device_idx, std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
                // if (device_idx == 0) {
                //     graph_storage_->updateInMemorySubGraph();
                //     for (int i = 0; i < devices_.size(); i ++) {
                //         initializeBatches(false, i);
                //     }
                //     swap_tasks_completed = true;
                // } else {
                //     batch_cv_->wait(batch_lock, [this, device_idx] {
                //         return swap_tasks_completed.load() == true;
                //     });
                // }
                batch = *batch_iterators_[device_idx];
                batch_iterators_[device_idx]++;

                // check if all batches have been read
                if (batch_iterators_[device_idx] == all_batches_[device_idx].end()) {
                    if (graph_storage_->useInMemorySubGraph()) {
                        if (!graph_storage_->hasSwap(device_idx)) {
                            all_reads_[device_idx] = true;
                        }
                    } else {
                        all_reads_[device_idx] = true;
                    }
                }
            } else {
                all_reads_[device_idx] = true;
            }
        } else {
            all_reads_[device_idx] = true;
        }
    }
    return batch;
}

bool DataLoader::hasNextBatch(int32_t device_idx) {
    batch_lock_->lock();
    bool ret = !all_reads_[device_idx];
    batch_lock_->unlock();
    return ret;
}

void DataLoader::finishedBatch(int32_t device_idx) {
    batch_lock_->lock();
    batches_left_[device_idx]--;
    if (batches_left_[device_idx] == 0) {
        activate_devices_ --;
    }
    total_batches_processed_++;
    batch_lock_->unlock();
    batch_cv_->notify_all();
}

shared_ptr<Batch> DataLoader::getBatch(at::optional<torch::Device> device, bool perform_map, int32_t device_idx) {
    shared_ptr<Batch> batch = getNextBatch(device_idx);
    
    if (batch == nullptr) {
        return batch;
    }
 
    if (batch->task_ == LearningTask::LINK_PREDICTION) {
        edgeSample(batch, device_idx);
    } else if (batch->task_ == LearningTask::NODE_CLASSIFICATION || batch->task_ == LearningTask::ENCODE) {
        nodeSample(batch, device_idx);
    }

    loadCPUParameters(batch);

    if (device.has_value()) {
        if (device.value().is_cuda()) {
            batch->to(device.value());
            loadGPUParameters(batch, device_idx);
            batch->dense_graph_.performMap();
        }
    }

    if (perform_map) {
        batch->dense_graph_.performMap();
    }

    return batch;
}

void DataLoader::edgeSample(shared_ptr<Batch> batch, int32_t device_idx) {

    if (!batch->edges_.defined()) {
        batch->edges_ = edge_sampler_->getEdges(batch, device_idx);
    }

    if (negative_sampler_ != nullptr) {
        negativeSample(batch, device_idx);
    }

    if (!train_) {
        int64_t edges_n = batch->edges_.defined() ? batch->edges_.size(0) : -1;
        int64_t src_neg_n = batch->src_neg_indices_.defined() ? batch->src_neg_indices_.numel() : -1;
        int64_t dst_neg_n = batch->dst_neg_indices_.defined() ? batch->dst_neg_indices_.numel() : -1;
        SPDLOG_INFO("eval negatives: edges={} src_neg_numel={} dst_neg_numel={}", edges_n, src_neg_n, dst_neg_n);
    }
    // std::cout << batch->src_neg_indices_ << std::endl;
    // std::cout << batch->edges_.sizes() << " " <<  batch->src_neg_indices_.sizes() << " " << batch->dst_neg_indices_.sizes() << std::endl;

    // int32_t false_negative_edges_src = 0;
    // int32_t false_negative_edges_dst = 0;
    // for (int32_t i = 0; i < 20; i++) {
    //     for (int32_t j = 0; j < batch->src_neg_indices_.size(1); j++) {
    //         if (batch->edges_[i][0].item<int64_t>() == batch->src_neg_indices_[i / 500][j].item<int64_t>()) {
    //             false_negative_edges_src ++;
    //             // std::cout << "[ " << i << " " << j << " " 
    //             //           << batch->edges_[i][0].item<int64_t>() << " " << batch->edges_[i][-1].item<int64_t>() << " "
    //             //           << batch->src_neg_indices_[i / 500][j].item<int64_t>() << " " << batch->dst_neg_indices_[i / 500][j].item<int64_t>() << " ]" << std::endl;

    //         }
    //         if (batch->edges_[i][-1].item<int32_t>() == batch->dst_neg_indices_[i / 500][j].item<int32_t>()) {
    //             false_negative_edges_dst++;
    //         }
    //     }
    // }
    // SPDLOG_INFO("false_negative_edges_src: {}", false_negative_edges_src);
    // SPDLOG_INFO("false_negative_edges_dst: {}", false_negative_edges_dst);
    std::vector<torch::Tensor> all_ids = {batch->edges_.select(1, 0), batch->edges_.select(1, -1)};

    if (batch->src_neg_indices_.defined()) {
        all_ids.emplace_back(batch->src_neg_indices_.flatten(0, 1));
    }

    if (batch->dst_neg_indices_.defined()) {
        all_ids.emplace_back(batch->dst_neg_indices_.flatten(0, 1));
    }

    torch::Tensor src_mapping;
    torch::Tensor dst_mapping;
    torch::Tensor src_neg_mapping;
    torch::Tensor dst_neg_mapping;

    std::vector<torch::Tensor> mapped_tensors;

    if (neighbor_sampler_ != nullptr) {
        // get unique nodes in edges and negatives
        batch->root_node_indices_ = std::get<0>(torch::_unique(torch::cat(all_ids)));

        // sample neighbors and get unique nodes
        batch->dense_graph_ = neighbor_sampler_->getNeighbors(batch->root_node_indices_, graph_storage_->current_subgraph_state_->in_memory_subgraph_);
        batch->unique_node_indices_ = batch->dense_graph_.getNodeIDs();

        // map edges and negatives to their corresponding index in unique_node_indices_
        auto tup = torch::sort(batch->unique_node_indices_);
        torch::Tensor sorted_map = std::get<0>(tup);
        torch::Tensor map_to_unsorted = std::get<1>(tup);

        mapped_tensors = apply_tensor_map(sorted_map, all_ids);

        int64_t num_nbrs_sampled = batch->dense_graph_.hop_offsets_[-2].item<int64_t>();

        src_mapping = map_to_unsorted.index_select(0, mapped_tensors[0]) - num_nbrs_sampled;
        dst_mapping = map_to_unsorted.index_select(0, mapped_tensors[1]) - num_nbrs_sampled;

        if (batch->src_neg_indices_.defined()) {
            src_neg_mapping = map_to_unsorted.index_select(0, mapped_tensors[2]).reshape(batch->src_neg_indices_.sizes()) - num_nbrs_sampled;
        }

        if (batch->dst_neg_indices_.defined()) {
            dst_neg_mapping = map_to_unsorted.index_select(0, mapped_tensors[3]).reshape(batch->dst_neg_indices_.sizes()) - num_nbrs_sampled;
        }
    } else {
        // map edges and negatives to their corresponding index in unique_node_indices_
        auto tup = map_tensors(all_ids);
    

        batch->unique_node_indices_ = std::get<0>(tup);

        if (batch->unique_node_indices_[0].item<int64_t>() == -1) {
            SPDLOG_ERROR("Node mapping is broken. Try repartition again.");
            throw std::runtime_error("");
        }


        mapped_tensors = std::get<1>(tup);

        src_mapping = mapped_tensors[0];
        dst_mapping = mapped_tensors[1];

        if (batch->src_neg_indices_.defined()) {
            src_neg_mapping = mapped_tensors[2].reshape(batch->src_neg_indices_.sizes());
        }

        if (batch->dst_neg_indices_.defined()) {
            dst_neg_mapping = mapped_tensors[3].reshape(batch->dst_neg_indices_.sizes());
        }
    }

    if (batch->edges_.size(1) == 2) {
        batch->edges_ = torch::stack({src_mapping, dst_mapping}).transpose(0, 1);
    } else if (batch->edges_.size(1) == 3) {
        batch->edges_ = torch::stack({src_mapping, batch->edges_.select(1, 1), dst_mapping}).transpose(0, 1);
    } else {
        throw TensorSizeMismatchException(batch->edges_, "Edge list must be a 3 or 2 column tensor");
    }

    batch->src_neg_indices_mapping_ = src_neg_mapping;
    batch->dst_neg_indices_mapping_ = dst_neg_mapping;
}

void DataLoader::nodeSample(shared_ptr<Batch> batch, int32_t device_idx) {
    if (batch->task_ == LearningTask::ENCODE) {
        torch::TensorOptions node_opts = torch::TensorOptions().dtype(torch::kInt64).device(graph_storage_->storage_ptrs_.edges->device_);
        batch->root_node_indices_ = torch::arange(batch->start_idx_, batch->start_idx_ + batch->batch_size_, node_opts);
    } else {
        batch->root_node_indices_ = graph_storage_->getNodeIdsRange(batch->start_idx_, batch->batch_size_).to(torch::kInt64);
    }

    if (graph_storage_->storage_ptrs_.node_labels != nullptr) {
        batch->node_labels_ = graph_storage_->getNodeLabels(batch->root_node_indices_).flatten(0, 1);
    }

    if (graph_storage_->current_subgraph_state_->global_to_local_index_map_.defined()) {
        batch->root_node_indices_ = graph_storage_->current_subgraph_state_->global_to_local_index_map_.index_select(0, batch->root_node_indices_);
    }

    if (neighbor_sampler_ != nullptr) {
        batch->dense_graph_ = neighbor_sampler_->getNeighbors(batch->root_node_indices_, graph_storage_->current_subgraph_state_->in_memory_subgraph_);
        batch->unique_node_indices_ = batch->dense_graph_.getNodeIDs();
    } else {
        batch->unique_node_indices_ = batch->root_node_indices_;
    }
}

void DataLoader::negativeSample(shared_ptr<Batch> batch, int32_t device_idx) {
    std::tie(batch->src_neg_indices_, batch->src_neg_filter_) =
        negative_sampler_->getNegatives(graph_storage_->current_subgraph_states_[device_idx]->in_memory_subgraph_, batch->edges_, true);

    std::tie(batch->dst_neg_indices_, batch->dst_neg_filter_) =
        negative_sampler_->getNegatives(graph_storage_->current_subgraph_states_[device_idx]->in_memory_subgraph_, batch->edges_, false);
}

void DataLoader::loadCPUParameters(shared_ptr<Batch> batch) {
    if (graph_storage_->storage_ptrs_.node_embeddings != nullptr) {
        if (graph_storage_->storage_ptrs_.node_embeddings->device_ != torch::kCUDA) {
            batch->node_embeddings_ = graph_storage_->getNodeEmbeddings(batch->unique_node_indices_);
            if (train_) {
                batch->node_embeddings_state_ = graph_storage_->getNodeEmbeddingState(batch->unique_node_indices_);
                if (negative_sampling_method_ == NegativeSamplingMethod::GAN) {
                    batch->node_embeddings_g_ = graph_storage_->getNodeEmbeddingsG(batch->unique_node_indices_);
                    batch->node_embeddings_state_g_ = graph_storage_->getNodeEmbeddingStateG(batch->unique_node_indices_);
                }
            }
        }
    }

    if (graph_storage_->storage_ptrs_.node_features != nullptr) {
        if (graph_storage_->storage_ptrs_.node_features->device_ != torch::kCUDA) {
            if (only_root_features_) {
                batch->node_features_ = graph_storage_->getNodeFeatures(batch->root_node_indices_);
            } else {
                batch->node_features_ = graph_storage_->getNodeFeatures(batch->unique_node_indices_);
            }
        }
    }

    batch->load_timestamp_ = timestamp_;
}

void DataLoader::loadGPUParameters(shared_ptr<Batch> batch, int32_t device_idx) {
    if (graph_storage_->storage_ptrs_.node_embeddings != nullptr) {
        if (graph_storage_->storage_ptrs_.node_embeddings->device_ == torch::kCUDA) {

            batch->node_embeddings_ = graph_storage_->getNodeEmbeddings(batch->unique_node_indices_, device_idx);

            if (train_) {
                batch->node_embeddings_state_ = graph_storage_->getNodeEmbeddingState(batch->unique_node_indices_, device_idx);
                if (negative_sampling_method_ == NegativeSamplingMethod::GAN) {
                    batch->node_embeddings_g_ = graph_storage_->getNodeEmbeddingsG(batch->unique_node_indices_, device_idx);
                    batch->node_embeddings_state_g_ = graph_storage_->getNodeEmbeddingStateG(batch->unique_node_indices_, device_idx);
                }
            }
        }
    }

    if (graph_storage_->storage_ptrs_.node_features != nullptr) {
        if (graph_storage_->storage_ptrs_.node_features->device_ == torch::kCUDA) {
            if (only_root_features_) {
                batch->node_features_ = graph_storage_->getNodeFeatures(batch->root_node_indices_);
            } else {
                batch->node_features_ = graph_storage_->getNodeFeatures(batch->unique_node_indices_);
            }
        }
    }
}

void DataLoader::updateEmbeddings(shared_ptr<Batch> batch, bool gpu, int32_t device_idx) {
    if (gpu) {
        if (graph_storage_->storage_ptrs_.node_embeddings->device_ == torch::kCUDA) {
            graph_storage_->updateAddNodeEmbeddings(batch->unique_node_indices_, batch->node_gradients_, device_idx);
            graph_storage_->updateAddNodeEmbeddingState(batch->unique_node_indices_, batch->node_state_update_, device_idx);
        }
    } else {
        batch->host_transfer_.synchronize();
        if (graph_storage_->storage_ptrs_.node_embeddings->device_ != torch::kCUDA) {
            graph_storage_->updateAddNodeEmbeddings(batch->unique_node_indices_, batch->node_gradients_);
            graph_storage_->updateAddNodeEmbeddingState(batch->unique_node_indices_, batch->node_state_update_);
        }
        batch->clear();
    }
}

void DataLoader::updateEmbeddingsG(shared_ptr<Batch> batch, bool gpu, int32_t device_idx) {
    if (gpu) {
        if (graph_storage_->storage_ptrs_.node_embeddings_g->device_ == torch::kCUDA) {
            graph_storage_->updateAddNodeEmbeddingsG(batch->unique_node_indices_, batch->node_gradients_g_, device_idx);
            graph_storage_->updateAddNodeEmbeddingStateG(batch->unique_node_indices_, batch->node_state_update_g_, device_idx);
        }
    } else {
        batch->host_transfer_.synchronize();
        if (graph_storage_->storage_ptrs_.node_embeddings_g->device_ != torch::kCUDA) {
            graph_storage_->updateAddNodeEmbeddingsG(batch->unique_node_indices_, batch->node_gradients_g_);
            graph_storage_->updateAddNodeEmbeddingStateG(batch->unique_node_indices_, batch->node_state_update_g_);
        }
        batch->clear();
    }
}

void DataLoader::loadStorage() {
    loaded_subgraphs = 0;
    async_barrier = 0;
    // Training may use one worker per configured GPU, while synchronous evaluation
    // consumes batches only on device 0. Reset this counter on every dataset switch.
    activate_devices_ = train_ ? static_cast<int64_t>(devices_.size()) : 1;
    setBufferOrdering();
    graph_storage_->load();
    if (negative_sampling_method_ == NegativeSamplingMethod::GAN) {
        graph_storage_->load_g();
    }

    batch_id_offset_ = 0;
    total_batches_processed_ = 0;

    all_batches_ = std::vector<std::vector<shared_ptr<Batch>>>(devices_.size());
    batches_left_ = std::vector<int32_t>(devices_.size());
    batch_iterators_ = std::vector<std::vector<shared_ptr<Batch>>::iterator>(devices_.size());
    all_reads_ = std::vector<bool>(devices_.size());

    for (int device_idx = 0; device_idx < devices_.size(); device_idx ++) {
        all_reads_[device_idx] = false;
    }

    if (!buffer_states_.empty()) {
        for(int device_idx = 0; device_idx < devices_.size(); device_idx ++) {
            graph_storage_->initializeInMemorySubGraph(buffer_states_[loaded_subgraphs ++], devices_[device_idx], device_idx);
        }
    } else {
        graph_storage_->initializeInMemorySubGraph(torch::empty({}));
    }

    if (negative_sampler_ != nullptr) {
        if (instance_of<NegativeSampler, CorruptNodeNegativeSampler>(negative_sampler_)) {
            if (std::dynamic_pointer_cast<CorruptNodeNegativeSampler>(negative_sampler_)->filtered_) {
                graph_storage_->sortAllEdges();
            }
        }
    }
}
