#include "engine/trainer.h"

#include "configuration/options.h"
#include "reporting/logger.h"
#include <c10/cuda/CUDACachingAllocator.h>

using std::get;
using std::tie;


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

        SPDLOG_INFO("################ Starting training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        for (int32_t device_idx = 0; device_idx < model_->device_models_.size(); device_idx ++) {
            threads.emplace_back(std::thread([this, &need_sync, &sync_generation, device_idx] {
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

                    if (has_relation) {
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
