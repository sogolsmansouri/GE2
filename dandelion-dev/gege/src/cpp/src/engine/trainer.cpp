#include "engine/trainer.h"

#include "configuration/options.h"
#include "common/nvtx_range.h"
#include "common/runtime_profile.h"
#include "reporting/logger.h"
#include <c10/cuda/CUDACachingAllocator.h>

using std::get;
using std::tie;

namespace {

double ns_to_ms(int64_t ns) { return static_cast<double>(ns) / 1.0e6; }

double avg_ns_to_us(int64_t ns, int64_t calls) {
    if (calls <= 0) {
        return 0.0;
    }
    return static_cast<double>(ns) / static_cast<double>(calls) / 1.0e3;
}

void log_runtime_profile_epoch(int64_t epoch_number) {
    auto snap = runtime_profile::captureAndReset();
    if (snap.empty()) {
        return;
    }

    SPDLOG_INFO(
        "[runtime-prof][epoch {}] loadGPUParameters {:.2f} ms ({} calls, {:.2f} us/call), "
        "getNodeEmbeddings {:.2f} ms ({} calls, {:.2f} us/call), "
        "getNodeState {:.2f} ms ({} calls, {:.2f} us/call)",
        epoch_number,
        ns_to_ms(snap.load_gpu_params_ns),
        snap.load_gpu_params_calls,
        avg_ns_to_us(snap.load_gpu_params_ns, snap.load_gpu_params_calls),
        ns_to_ms(snap.storage_get_embeddings_ns),
        snap.storage_get_embeddings_calls,
        avg_ns_to_us(snap.storage_get_embeddings_ns, snap.storage_get_embeddings_calls),
        ns_to_ms(snap.storage_get_state_ns),
        snap.storage_get_state_calls,
        avg_ns_to_us(snap.storage_get_state_ns, snap.storage_get_state_calls));

    SPDLOG_INFO(
        "[runtime-prof][epoch {}] buffer.indexRead {:.2f} ms ({} calls, {:.2f} us/call), "
        "buffer.indexAdd {:.2f} ms ({} calls, {:.2f} us/call), "
        "updateEmbeddings(gpu) {:.2f} ms ({} calls, {:.2f} us/call), "
        "updateEmbeddings(host) {:.2f} ms ({} calls, {:.2f} us/call)",
        epoch_number,
        ns_to_ms(snap.buffer_index_read_ns),
        snap.buffer_index_read_calls,
        avg_ns_to_us(snap.buffer_index_read_ns, snap.buffer_index_read_calls),
        ns_to_ms(snap.buffer_index_add_ns),
        snap.buffer_index_add_calls,
        avg_ns_to_us(snap.buffer_index_add_ns, snap.buffer_index_add_calls),
        ns_to_ms(snap.update_embeddings_gpu_ns),
        snap.update_embeddings_gpu_calls,
        avg_ns_to_us(snap.update_embeddings_gpu_ns, snap.update_embeddings_gpu_calls),
        ns_to_ms(snap.update_embeddings_host_ns),
        snap.update_embeddings_host_calls,
        avg_ns_to_us(snap.update_embeddings_host_ns, snap.update_embeddings_host_calls));

    SPDLOG_INFO(
        "[runtime-prof][epoch {}] Batch::to {:.2f} ms ({} calls, {:.2f} us/call), "
        "Batch::embeddingsToHost {:.2f} ms ({} calls, {:.2f} us/call)",
        epoch_number,
        ns_to_ms(snap.batch_to_device_ns),
        snap.batch_to_device_calls,
        avg_ns_to_us(snap.batch_to_device_ns, snap.batch_to_device_calls),
        ns_to_ms(snap.batch_to_host_ns),
        snap.batch_to_host_calls,
        avg_ns_to_us(snap.batch_to_host_ns, snap.batch_to_host_calls));
}

} // namespace


SynchronousTrainer::SynchronousTrainer(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, int logs_per_epoch) {
    dataloader_ = dataloader;
    model_ = model;
    learning_task_ = dataloader_->learning_task_;

    std::string item_name;
    int64_t num_items = 0;
    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        item_name = "Edges";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        item_name = "Nodes";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
    }

    progress_reporter_ = std::make_shared<ProgressReporter>(item_name, num_items, logs_per_epoch);
}

void SynchronousTrainer::train(int num_epochs) {

    if (!dataloader_->single_dataset_) {
        dataloader_->setTrainSet();
    }
    dataloader_->initializeBatches(false);
    c10::cuda::CUDACachingAllocator::emptyCache();

    Timer timer = Timer(false);
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        timer.start();
        auto epoch_number = dataloader_->getEpochsProcessed() + 1;
        std::string epoch_range_name = "train_epoch_" + std::to_string(epoch_number);
        nvtx3::scoped_range epoch_range{epoch_range_name.c_str()};
        SPDLOG_INFO("################ Starting training epoch {} ################", epoch_number);
        while (dataloader_->hasNextBatch()) {
            // gets data and parameters for the next batch
            Timer timer0 = Timer(false);
            timer0.start();

            shared_ptr<Batch> batch = dataloader_->getBatch();

            if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                batch->to(model_->device_);
            } else {
                dataloader_->loadGPUParameters(batch);
            }

            if (batch->node_embeddings_.defined()) {
                batch->node_embeddings_.requires_grad_();
            }

            batch->dense_graph_.performMap();

            model_->train_batch(batch);


            
            if (batch->node_embeddings_.defined()) {
                if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                    batch->embeddingsToHost();
                } else {
                    dataloader_->updateEmbeddings(batch, true);
                }
                dataloader_->updateEmbeddings(batch, false);        
            }

            if (batch->node_embeddings_g_.defined()) {
                if (dataloader_->graph_storage_->embeddingsOffDeviceG()) {
                    batch->embeddingsToHostG();
                } else {
                    dataloader_->updateEmbeddingsG(batch, true);
                }
                dataloader_->updateEmbeddingsG(batch, false);        
            }

            batch->clear();
            // notify that the batch has been completed
            dataloader_->finishedBatch();

            // log progress
            progress_reporter_->addResult(batch->batch_size_);

        }
        SPDLOG_INFO("################ Finished training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        timer.stop();
        
        // notify that the epoch has been completed
        dataloader_->nextEpoch();
        progress_reporter_->clear();

        std::string item_name;
        int64_t num_items = 0;
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            item_name = "Edges";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            item_name = "Nodes";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
        }

        int64_t epoch_time = timer.getDuration();
        float items_per_second = (float)num_items / ((float)epoch_time / 1000);
        SPDLOG_INFO("Epoch Runtime: {}ms", epoch_time);
        SPDLOG_INFO("{} per Second: {}", item_name, items_per_second);
        log_runtime_profile_epoch(epoch_number);
    }
}


SynchronousMultiGPUTrainer::SynchronousMultiGPUTrainer(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, int logs_per_epoch) {
    dataloader_ = dataloader;
    model_ = model;
    learning_task_ = dataloader_->learning_task_;

    std::string item_name;
    int64_t num_items = 0;
    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        item_name = "Edges";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        item_name = "Nodes";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
    }

    progress_reporter_ = std::make_shared<ProgressReporter>(item_name, num_items, logs_per_epoch);
}



void SynchronousMultiGPUTrainer::train(int num_epochs) {
    if (!dataloader_->single_dataset_) {
	    dataloader_->setTrainSet();
    }

    dataloader_->activate_devices_ = model_->device_models_.size();

    for (int i = 0; i < model_->device_models_.size(); i ++) {
        dataloader_->initializeBatches(false, i);
    }
    c10::cuda::CUDACachingAllocator::emptyCache();

    Timer timer = Timer(false); 

    std::atomic<int64_t> need_sync = 0;
    std::atomic<bool> sync_finished = false;
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        timer.start();
        std::vector<std::thread> threads;

        auto epoch_number = dataloader_->getEpochsProcessed() + 1;
        std::string epoch_range_name = "train_epoch_" + std::to_string(epoch_number);
        nvtx3::scoped_range epoch_range{epoch_range_name.c_str()};
        SPDLOG_INFO("################ Starting training epoch {} ################", epoch_number);
        for (int32_t device_idx = 0; device_idx < model_->device_models_.size(); device_idx ++) {
            threads.emplace_back(std::thread([this, &need_sync, &sync_finished, device_idx] {
                while (dataloader_->hasNextBatch(device_idx)) {
                    // gets data and parameters for the next batch

                    shared_ptr<Batch> batch = dataloader_->getBatch(c10::nullopt, false, device_idx);
                    bool has_relation = (batch->edges_.size(1) == 3);
                    dataloader_->loadGPUParameters(batch, device_idx);

                    if (batch->node_embeddings_.defined()) {
                        batch->node_embeddings_.requires_grad_();
                    }

                    batch->dense_graph_.performMap();

                    model_->device_models_[device_idx]->train_batch(batch, false);

                    if (batch->node_embeddings_.defined()) {
                        if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                            batch->embeddingsToHost();
                        } else {
                            dataloader_->updateEmbeddings(batch, true, device_idx);
                        }
                        dataloader_->updateEmbeddings(batch, false, device_idx);
                    }

                    if (batch->node_embeddings_g_.defined()) {
                        if (dataloader_->graph_storage_->embeddingsOffDeviceG()) {
                            batch->embeddingsToHostG();
                        } else {
                            dataloader_->updateEmbeddingsG(batch, true, device_idx);
                        }
                        dataloader_->updateEmbeddingsG(batch, false, device_idx);
                    }


                    // if(has_relation) {
                    //     if (dataloader_->batches_left_[device_idx] == 1) {
                    //         sync_finished = false;
                    //         need_sync ++;

                    //         if (need_sync == dataloader_->activate_devices_) {
                    //             model_->all_reduce_rel();
                    //             sync_finished = true;
                    //             need_sync = 0;
                    //         }
                    //         while (!sync_finished) {
                    //             std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    //         }
                    //     }
                    // }

                    if(has_relation) {
                        // if ((batch->batch_id_ + 1) % 1 == 0 || dataloader_->batches_left_[device_idx] == 1) {
                        {
                            sync_finished = false;
                            need_sync ++;

                            if (need_sync == dataloader_->activate_devices_) {
                                model_->all_reduce();
                                sync_finished = true;
                                need_sync = 0;
                            }
                            while (!sync_finished) {
                                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            }
                        }
                    }
                    
                    batch->clear();
                    // notify that the batch has been completed
                    dataloader_->finishedBatch(device_idx);
                 }
            }));
        }
        for(auto &thread : threads){ thread.join(); }
        // if (model_->device_models_.size() > 1)
        //     model_->all_reduce();

        SPDLOG_INFO("################ Finished training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        timer.stop();
        // notify that the epoch has been completed
        dataloader_->nextEpoch();
        progress_reporter_->clear();

        std::string item_name;
        int64_t num_items = 0;
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            item_name = "Edges";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            item_name = "Nodes";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
        }

        int64_t epoch_time = timer.getDuration();
        float items_per_second = (float)num_items / ((float)epoch_time / 1000);
        SPDLOG_INFO("Epoch Runtime: {}ms", epoch_time);
        SPDLOG_INFO("{} per Second: {}", item_name, items_per_second);
        log_runtime_profile_epoch(epoch_number);
    }
}
