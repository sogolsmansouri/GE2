#include "engine/trainer.h"

#include "configuration/options.h"
#include "reporting/logger.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>

using std::get;
using std::tie;

namespace {
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
        SPDLOG_INFO("################ Starting training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
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

        dataloader_->graph_storage_->logEpochCommunicationStatsAndReset(dataloader_->getEpochsProcessed() + 1);
        
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
    std::atomic<int64_t> sync_generation = 0;
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        timer.start();
        std::vector<std::thread> threads;
        std::vector<int64_t> batches_processed(model_->device_models_.size(), 0);
        std::vector<int64_t> get_batch_ms(model_->device_models_.size(), 0);
        std::vector<int64_t> load_gpu_params_ms(model_->device_models_.size(), 0);
        std::vector<int64_t> train_batch_ms(model_->device_models_.size(), 0);
        std::vector<int64_t> update_embeddings_ms(model_->device_models_.size(), 0);
        std::vector<int64_t> sync_wait_ms(model_->device_models_.size(), 0);
        std::vector<int64_t> finish_batch_ms(model_->device_models_.size(), 0);

        SPDLOG_INFO("################ Starting training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        for (int32_t device_idx = 0; device_idx < model_->device_models_.size(); device_idx ++) {
            threads.emplace_back(std::thread([this, &need_sync, &sync_generation, &batches_processed, &get_batch_ms, &load_gpu_params_ms,
                                              &train_batch_ms, &update_embeddings_ms, &sync_wait_ms, &finish_batch_ms, device_idx] {
                while (dataloader_->hasNextBatch(device_idx)) {
                    // gets data and parameters for the next batch
                    auto t_get_batch_start = std::chrono::steady_clock::now();
                    shared_ptr<Batch> batch = dataloader_->getBatch(c10::nullopt, false, device_idx);
                    auto t_get_batch_end = std::chrono::steady_clock::now();
                    get_batch_ms[device_idx] +=
                        std::chrono::duration_cast<std::chrono::milliseconds>(t_get_batch_end - t_get_batch_start).count();
                    bool has_relation = (batch->edges_.size(1) == 3);
                    auto t_load_gpu_start = std::chrono::steady_clock::now();
                    dataloader_->loadGPUParameters(batch, device_idx);
                    auto t_load_gpu_end = std::chrono::steady_clock::now();
                    load_gpu_params_ms[device_idx] +=
                        std::chrono::duration_cast<std::chrono::milliseconds>(t_load_gpu_end - t_load_gpu_start).count();

                    if (batch->node_embeddings_.defined()) {
                        batch->node_embeddings_.requires_grad_();
                    }

                    batch->dense_graph_.performMap();

                    auto t_train_batch_start = std::chrono::steady_clock::now();
                    model_->device_models_[device_idx]->train_batch(batch, false);
                    auto t_train_batch_end = std::chrono::steady_clock::now();
                    train_batch_ms[device_idx] +=
                        std::chrono::duration_cast<std::chrono::milliseconds>(t_train_batch_end - t_train_batch_start).count();

                    auto t_update_embeddings_start = std::chrono::steady_clock::now();
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
                    auto t_update_embeddings_end = std::chrono::steady_clock::now();
                    update_embeddings_ms[device_idx] +=
                        std::chrono::duration_cast<std::chrono::milliseconds>(t_update_embeddings_end - t_update_embeddings_start).count();


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

                    if (has_relation) {
                        auto t_sync_start = std::chrono::steady_clock::now();
                        int64_t observed_generation = sync_generation.load();
                        int64_t arrivals = need_sync.fetch_add(1) + 1;

                        if (arrivals == dataloader_->activate_devices_) {
                            model_->all_reduce();
                            need_sync.store(0);
                            sync_generation.fetch_add(1);
                        } else {
                            while (sync_generation.load() == observed_generation) {
                                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            }
                        }
                        auto t_sync_end = std::chrono::steady_clock::now();
                        sync_wait_ms[device_idx] +=
                            std::chrono::duration_cast<std::chrono::milliseconds>(t_sync_end - t_sync_start).count();
                    }
                    
                    batch->clear();
                    // notify that the batch has been completed
                    auto t_finish_batch_start = std::chrono::steady_clock::now();
                    dataloader_->finishedBatch(device_idx);
                    auto t_finish_batch_end = std::chrono::steady_clock::now();
                    finish_batch_ms[device_idx] +=
                        std::chrono::duration_cast<std::chrono::milliseconds>(t_finish_batch_end - t_finish_batch_start).count();
                    batches_processed[device_idx]++;
                 }
            }));
        }
        for(auto &thread : threads){ thread.join(); }
        // if (model_->device_models_.size() > 1)
        //     model_->all_reduce();

        SPDLOG_INFO("################ Finished training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        timer.stop();

        dataloader_->graph_storage_->logEpochCommunicationStatsAndReset(dataloader_->getEpochsProcessed() + 1);
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

        if (profile_timing_enabled()) {
            int64_t total_batches = 0;
            int64_t total_get_batch_ms = 0;
            int64_t total_load_gpu_params_ms = 0;
            int64_t total_train_batch_ms = 0;
            int64_t total_update_embeddings_ms = 0;
            int64_t total_sync_wait_ms = 0;
            int64_t total_finish_batch_ms = 0;

            for (int32_t device_idx = 0; device_idx < model_->device_models_.size(); device_idx++) {
                total_batches += batches_processed[device_idx];
                total_get_batch_ms += get_batch_ms[device_idx];
                total_load_gpu_params_ms += load_gpu_params_ms[device_idx];
                total_train_batch_ms += train_batch_ms[device_idx];
                total_update_embeddings_ms += update_embeddings_ms[device_idx];
                total_sync_wait_ms += sync_wait_ms[device_idx];
                total_finish_batch_ms += finish_batch_ms[device_idx];

                double denom = std::max<int64_t>(1, batches_processed[device_idx]);
                SPDLOG_INFO(
                    "Timing epoch {} device {}: batches={} get_batch={}ms ({:.3f}ms/b) "
                    "load_gpu={}ms ({:.3f}ms/b) train_batch={}ms ({:.3f}ms/b) "
                    "update_embeddings={}ms ({:.3f}ms/b) sync_wait={}ms ({:.3f}ms/b) "
                    "finish_batch={}ms ({:.3f}ms/b)",
                    dataloader_->getEpochsProcessed(),
                    device_idx,
                    batches_processed[device_idx],
                    get_batch_ms[device_idx], (double)get_batch_ms[device_idx] / denom,
                    load_gpu_params_ms[device_idx], (double)load_gpu_params_ms[device_idx] / denom,
                    train_batch_ms[device_idx], (double)train_batch_ms[device_idx] / denom,
                    update_embeddings_ms[device_idx], (double)update_embeddings_ms[device_idx] / denom,
                    sync_wait_ms[device_idx], (double)sync_wait_ms[device_idx] / denom,
                    finish_batch_ms[device_idx], (double)finish_batch_ms[device_idx] / denom);
            }

            double total_denom = std::max<int64_t>(1, total_batches);
            SPDLOG_INFO(
                "Timing epoch {} totals: batches={} get_batch={}ms ({:.3f}ms/b) "
                "load_gpu={}ms ({:.3f}ms/b) train_batch={}ms ({:.3f}ms/b) "
                "update_embeddings={}ms ({:.3f}ms/b) sync_wait={}ms ({:.3f}ms/b) "
                "finish_batch={}ms ({:.3f}ms/b)",
                dataloader_->getEpochsProcessed(),
                total_batches,
                total_get_batch_ms, (double)total_get_batch_ms / total_denom,
                total_load_gpu_params_ms, (double)total_load_gpu_params_ms / total_denom,
                total_train_batch_ms, (double)total_train_batch_ms / total_denom,
                total_update_embeddings_ms, (double)total_update_embeddings_ms / total_denom,
                total_sync_wait_ms, (double)total_sync_wait_ms / total_denom,
                total_finish_batch_ms, (double)total_finish_batch_ms / total_denom);
        }
    }
}
