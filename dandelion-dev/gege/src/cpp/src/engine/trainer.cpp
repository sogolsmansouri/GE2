#include "engine/trainer.h"

#include "configuration/options.h"
#include "reporting/logger.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <nvtx3/nvtx3.hpp>

using std::get;
using std::tie;


/*
    Struct SynchronousTrainer: single GPU traning
    Struct SynchronousMultiGPUTrainer: multi GPU traning


*/


/*
    @zizhong：
    Knowledge graph focus on link prediction task

*/
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


/*

*/
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
        /*
            @zizhong
            NVTX tags for profiling the training epoch.
        */


        //nvtx3::scoped_range epoch_range{epoch_range_name.c_str()};
        SPDLOG_INFO("################ Starting training epoch {} ################", epoch_number);

        /*
            @zizhong
            NVTX tags for more fine grained profiling.
            the following code consists of these traning pipeline profiling:
            1. get batch
            2. load parameters to GPU
            3. train on the batch
            4. update embeddings to dataloader
                4.1 update embeddings based on the value after training
                4.2 update embeddings based on the gradient after training
            5. clear batch and notify dataloader the batch is finished
            ===================
            Single GPU Training Pipeline:
            1. func train_batch_total: the total time for training on a batch, including data loading, training, and updating embeddings.
            2. func getBatch: the time for dataloader to prepare the batch,
            3. func prepare_gpu_parameters: the time for loading parameters to GPU, which includes two cases: 
                3.1 func batch_to_device: if embeddings are off device, move the whole batch to GPU;
                3.2 func load_gpu_parameters: if embeddings are on device, load parameters to GPU without moving the batch.
            4. func perform_map: the time for performing map on the dense graph, which
            4. func model_train_batch: the time for training on the batch.
            5. func update_embeddings_main: the time for updating embeddings to dataloader, which includes two cases:
                5.1 func embeddings_to_host: if embeddings are off device, move the updated embeddings back to host;
                5.2 func update_embeddings_true: if embeddings are on device, update the embeddings in dataloader with the updated embeddings from GPU.
            6.func update_embeddings_false: the time for updating embeddings in dataloader without GPU parameters, which is essentially the time for updating the non-embedding parameters in dataloader.
            7. func update_embeddings_grad: the time for updating embeddings based on the gradient,
                7.1 func embeddings_to_host_g: if gradients of embeddings are off device, move the gradients back to host;
                7.2 func update_embeddings_g_true: if gradients of embeddings are on device, update the gradients of embeddings in dataloader with the gradients from GPU.
                7.3 func update_embeddings_g_false: the time for updating gradients of embeddings in dataloader without GPU parameters, which is essentially the time for updating the gradients of non-embedding parameters in dataloader.
            8. fun batch_clear: the time for clearing the batch after training.
            9. func finished_batch: the time for notifying dataloader that the batch is finished.
        */
        // while (dataloader_->hasNextBatch()) {
        //     nvtx3::scoped_range batch_range{"train_total"};

        //     shared_ptr<Batch> batch;
        //     {
        //         nvtx3::scoped_range r{"getBatch"};
        //         batch = dataloader_->getBatch();
        //     }

        //     {
        //         nvtx3::scoped_range r{"prepare_gpu_parameters"};
        //         /*
        //             @zizhong: In our case, it does not go through path offDevice(CPU side Training)
        //         */
        //         if (dataloader_->graph_storage_->embeddingsOffDevice()) {
        //             nvtx3::scoped_range r2{"batch_to_device"};
        //             batch->to(model_->device_);
        //         } else {
        //             nvtx3::scoped_range r2{"load_gpu_parameters"};
        //             dataloader_->loadGPUParameters(batch);
        //         }
        //     }

        //     if (batch->node_embeddings_.defined()) {
        //         batch->node_embeddings_.requires_grad_();
        //     }

        //     {
        //         nvtx3::scoped_range r{"perform_map"};
        //         batch->dense_graph_.performMap();
        //     }
        //     /*
        //         @zizhong
        //         train_batch is the major bottleneck.
        //     */
        //     {
        //         nvtx3::scoped_range r{"model_train_batch"};
        //         model_->train_batch(batch);
        //     }

        //     if (batch->node_embeddings_.defined()) {
        //         nvtx3::scoped_range r{"update_embeddings_main"};
        //         if (dataloader_->graph_storage_->embeddingsOffDevice()) {
        //             nvtx3::scoped_range r2{"embeddings_to_host"};
        //             batch->embeddingsToHost();
        //         } else {
        //             nvtx3::scoped_range r2{"update_embeddings_true"};
        //             dataloader_->updateEmbeddings(batch, true);
        //         }
        //         {
        //             nvtx3::scoped_range r2{"update_embeddings_false"};
        //             dataloader_->updateEmbeddings(batch, false);
        //         }
        //     }

        //     if (batch->node_embeddings_g_.defined()) {
        //         nvtx3::scoped_range r{"update_embeddings_grad"};
        //         if (dataloader_->graph_storage_->embeddingsOffDeviceG()) {
        //             nvtx3::scoped_range r2{"embeddings_to_host_g"};
        //             batch->embeddingsToHostG();
        //         } else {
        //             nvtx3::scoped_range r2{"update_embeddings_g_true"};
        //             dataloader_->updateEmbeddingsG(batch, true);
        //         }
        //         {
        //             nvtx3::scoped_range r2{"update_embeddings_g_false"};
        //             dataloader_->updateEmbeddingsG(batch, false);
        //         }
        //     }

        //     {
        //         nvtx3::scoped_range r{"batch_clear"};
        //         batch->clear();
        //     }

        //     {
        //         nvtx3::scoped_range r{"finished_batch"};
        //         dataloader_->finishedBatch();
        //     }

        //     progress_reporter_->addResult(batch->batch_size_);
        // }

        /*
            No nvtx tag version
        */
        while (dataloader_->hasNextBatch()) {
            shared_ptr<Batch> batch;

            batch = dataloader_->getBatch();

            /*
                @zizhong: In our case, it does not go through path offDevice(CPU side Training)
            */
            if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                batch->to(model_->device_);
            } else {
                dataloader_->loadGPUParameters(batch);
            }

            if (batch->node_embeddings_.defined()) {
                batch->node_embeddings_.requires_grad_();
            }

            batch->dense_graph_.performMap();

            /*
                @zizhong
                train_batch is the major bottleneck.
            */
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

            dataloader_->finishedBatch();

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

/*
    Multi-GPU Trainer
    Each GPU assigns a CPU thread+each batch+local train+after each batch:all-reduce
*/

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
    /*
        @zizhong
        synchronization semaphores:
        1. need_sync: counts how many devices have finished the current batch and need synchronization.
        2. sync_finished: indicates whether the synchronization (all-reduce) has been completed
    */
    std::atomic<int64_t> need_sync = 0;
    std::atomic<bool> sync_finished = false;
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        timer.start();
        std::vector<std::thread> threads;
        /*
            @zizhong
            each GPU <-> 1 host thread
            each thread responsible for:
            1. get batch from dataloader
            2. load parameters to GPU
            3. train on the batch
            4. update embeddings to dataloader
            5. after each batch, check if synchronization is needed and perform all-reduce if
        */

        auto epoch_number = dataloader_->getEpochsProcessed() + 1;
        std::string epoch_range_name = "train_epoch_" + std::to_string(epoch_number);
        //nvtx3::scoped_range epoch_range{epoch_range_name.c_str()};
        SPDLOG_INFO("################ Starting training epoch {} ################", epoch_number);
        for (int32_t device_idx = 0; device_idx < model_->device_models_.size(); device_idx ++) {
            threads.emplace_back(std::thread([this, &need_sync, &sync_finished, device_idx] {
                while (dataloader_->hasNextBatch(device_idx)) {
                    // gets data and parameters for the next batch

                    shared_ptr<Batch> batch = dataloader_->getBatch(c10::nullopt, false, device_idx);
                    bool has_relation = (batch->edges_.size(1) == 3);
                    /*
                        @zizhong
                        ToDo:
                         1. check "loadGPUParameters" to see what parameters are loaded to GPU and how they are loaded.
                    */
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
                    /*
                        @zizhong
                        simple implementation:
                        1. for each gpu, host thread "need_sync++"
                        2. if "need_sync" == number of active devices(last GPU reach barrier), perform all-reduce and reset "need_sync"
                        ‼️ load imbalance
                        ToDo:
                        1. check all_reduce func: what to be synchronized and updated? relation or both relation and node embeddings? 
                    */
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
    }
}
