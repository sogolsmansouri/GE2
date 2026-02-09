#include "storage/checkpointer.h"

#include "configuration/util.h"
#include "reporting/logger.h"
#include "storage/io.h"
#include "storage/storage.h"

Checkpointer::Checkpointer(std::shared_ptr<Model> model, shared_ptr<GraphModelStorage> storage, std::shared_ptr<CheckpointConfig> config) {
    model_ = model;
    storage_ = storage;
    config_ = config;
}

void Checkpointer::create_checkpoint(string checkpoint_dir, CheckpointMeta checkpoint_meta, int epochs) {
    string tmp_checkpoint_dir = checkpoint_dir + "checkpoint_" + std::to_string(epochs) + "_tmp/";
    createDir(tmp_checkpoint_dir, false);

    std::string new_embeddings_file = tmp_checkpoint_dir + PathConstants::embeddings_file + PathConstants::file_ext;
    std::string new_embeddings_state_file = tmp_checkpoint_dir + PathConstants::embeddings_state_file + PathConstants::file_ext;

    std::string embeddings_file = checkpoint_dir + PathConstants::embeddings_file + PathConstants::file_ext;
    std::string embeddings_state_file = checkpoint_dir + PathConstants::embeddings_state_file + PathConstants::file_ext;

    if (fileExists(embeddings_file)) {
        copyFile(embeddings_file, new_embeddings_file);
        if (this->config_->save_state) copyFile(embeddings_state_file, new_embeddings_state_file);
    }

    this->save(tmp_checkpoint_dir, checkpoint_meta);

    string final_checkpoint_dir = checkpoint_dir + "checkpoint_" + std::to_string(epochs) + "/";
    renameFile(tmp_checkpoint_dir, final_checkpoint_dir);
}

void Checkpointer::save(string checkpoint_dir, CheckpointMeta checkpoint_meta) {
    if (checkpoint_meta.has_model) {
        if (storage_->storage_ptrs_.node_embeddings != nullptr) {
            storage_->storage_ptrs_.node_embeddings->write();
        }
        if (storage_->storage_ptrs_.node_embeddings_g != nullptr) {
            storage_->storage_ptrs_.node_embeddings_g->write();
        }
        model_->save(checkpoint_dir);
    }

    if (checkpoint_meta.has_state) {
        if (storage_->storage_ptrs_.node_optimizer_state != nullptr) {
            storage_->storage_ptrs_.node_optimizer_state->write();
        }
        if (storage_->storage_ptrs_.node_optimizer_state_g != nullptr) {
            storage_->storage_ptrs_.node_optimizer_state_g->write();
        }
    }

    saveMetadata(checkpoint_dir, checkpoint_meta);
}

std::tuple<std::shared_ptr<Model>, shared_ptr<GraphModelStorage>, CheckpointMeta> Checkpointer::load(string checkpoint_dir,
                                                                                                     std::shared_ptr<GegeConfig> gege_config, bool train) {
    CheckpointMeta checkpoint_meta = loadMetadata(checkpoint_dir);

    std::vector<torch::Device> devices = devices_from_config(gege_config->storage);
    std::shared_ptr<Model> model = initModelFromConfig(gege_config->model, devices, gege_config->storage->dataset->num_relations, train,
                                                       gege_config->training->negative_sampling_method, gege_config->training->negative_sampling_selected_ratio);
    model->load(checkpoint_dir, train);

    if (checkpoint_meta.link_prediction) {
        model->learning_task_ = LearningTask::LINK_PREDICTION;
    } else {
        model->learning_task_ = LearningTask::NODE_CLASSIFICATION;
    }

    shared_ptr<GraphModelStorage> storage = initializeStorage(model, gege_config->storage, false, train, gege_config->training->negative_sampling_method);

    return std::forward_as_tuple(model, storage, checkpoint_meta);
}

CheckpointMeta Checkpointer::loadMetadata(string directory) {
    CheckpointMeta ret_meta;

    std::ifstream input_file;
    string metadata_file = directory + PathConstants::checkpoint_metadata_file;

    input_file.open(metadata_file);
    if (!input_file.is_open()) {
        throw std::runtime_error("Checkpoint metadata file not found: " + metadata_file);
    }

    auto read_line_or_throw = [&](const char *field_name) -> string {
        string line;
        if (!std::getline(input_file, line)) {
            throw std::runtime_error("Checkpoint metadata is missing field: " + string(field_name) + " (" + metadata_file + ")");
        }
        return line;
    };

    auto parse_int_or_throw = [&](const string &line, const char *field_name) -> int {
        try {
            size_t pos = 0;
            int value = std::stoi(line, &pos);
            if (pos != line.size()) {
                throw std::invalid_argument("trailing characters");
            }
            return value;
        } catch (const std::exception &) {
            throw std::runtime_error("Invalid integer for metadata field '" + string(field_name) + "': '" + line + "' (" + metadata_file + ")");
        }
    };

    ret_meta.name = read_line_or_throw("name");

    string line = read_line_or_throw("num_epochs");
    ret_meta.num_epochs = parse_int_or_throw(line, "num_epochs");

    line = read_line_or_throw("checkpoint_id");
    ret_meta.checkpoint_id = parse_int_or_throw(line, "checkpoint_id");

    line = read_line_or_throw("link_prediction");
    std::istringstream(line) >> ret_meta.link_prediction;

    line = read_line_or_throw("has_state");
    std::istringstream(line) >> ret_meta.has_state;

    line = read_line_or_throw("has_encoded");
    std::istringstream(line) >> ret_meta.has_encoded;

    line = read_line_or_throw("has_model");
    std::istringstream(line) >> ret_meta.has_model;

    return ret_meta;
}

void Checkpointer::saveMetadata(string directory, CheckpointMeta checkpoint_meta) {
    std::ofstream output_file;
    output_file.open(directory + PathConstants::checkpoint_metadata_file);

    output_file << checkpoint_meta.name << "\n";
    output_file << checkpoint_meta.num_epochs << "\n";
    output_file << checkpoint_meta.checkpoint_id << "\n";
    output_file << checkpoint_meta.link_prediction << "\n";
    output_file << checkpoint_meta.has_state << "\n";
    output_file << checkpoint_meta.has_encoded << "\n";
    output_file << checkpoint_meta.has_model << "\n";
}
