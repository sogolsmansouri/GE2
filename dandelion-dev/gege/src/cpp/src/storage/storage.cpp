#include "storage/storage.h"

#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <cstdlib>
#include <atomic>

#include "common/util.h"
#include "configuration/constants.h"
#include "fused_sparse_update.h"
#include "reporting/logger.h"

using std::ios;
using std::ios_base;

namespace {
class CudaDeviceRestoreGuard {
   public:
    CudaDeviceRestoreGuard() : restore_(cudaGetDevice(&device_) == cudaSuccess) {}

    ~CudaDeviceRestoreGuard() {
        if (restore_) {
            cudaSetDevice(device_);
        }
    }

   private:
    int device_ = -1;
    bool restore_ = false;
};

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

bool should_log_fused_sparse_debug() {
    static std::atomic<int64_t> budget{0};
    if (fused_sparse_debug_level() <= 0) {
        return false;
    }
    // Limit volume in hot paths.
    return budget.fetch_add(1) < 64;
}

void log_fused_sparse_debug(const std::string &msg) {
    if (should_log_fused_sparse_debug()) {
        SPDLOG_INFO("FusedSparse debug: {}", msg);
    }
}
}  // namespace

void renameFile(string old_filename, string new_filename) {
    int result = rename(old_filename.c_str(), new_filename.c_str());
    if (result != 0) {
        SPDLOG_ERROR("Unable to rename {}\nError: {}", old_filename, errno);
        throw std::runtime_error("");
    }
}

void copyFile(string src_file, string dst_file) {
    std::ifstream src;
    std::ofstream dst;

    src.open(src_file, ios::in | ios::binary);
    dst.open(dst_file, ios::out | ios::binary);

    dst << src.rdbuf();

    src.close();
    dst.close();
}

bool fileExists(string filename) {
    if (FILE *file = fopen(filename.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

void createDir(string path, bool exist_ok) {
    if (mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
        if (errno == EEXIST) {
            if (exist_ok) {
                SPDLOG_DEBUG("{} directory already exists", path);
            } else {
                SPDLOG_ERROR("{} directory already exists", path);
                throw std::runtime_error("");
            }
        } else {
            SPDLOG_ERROR("Failed to create {}\nError: {}", path, errno);
            throw std::runtime_error("");
        }
    }
}

Storage::Storage() : device_(torch::kCPU) {}

PartitionBufferStorage::PartitionBufferStorage(string filename, int64_t dim0_size, int64_t dim1_size, shared_ptr<PartitionBufferOptions> options) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    options_ = options;
    dtype_ = options_->dtype;
    initialized_ = true;
    loaded_ = false;
    int64_t partition_size = ceil((double)dim0_size_ / options_->num_partitions);
    device_ = torch::kCPU;

    buffer_ = new PartitionBuffer(options_->buffer_capacity, options_->num_partitions, options_->fine_to_coarse_ratio, partition_size, dim1_size_, dim0_size_,
                                  dtype_, filename_, options_->prefetching);
}

PartitionBufferStorage::PartitionBufferStorage(string filename, torch::Tensor data, shared_ptr<PartitionBufferOptions> options) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = data.size(1);
    options_ = options;
    dtype_ = options_->dtype;
    append(data);
    initialized_ = true;
    loaded_ = false;
    int64_t partition_size = ceil((double)dim0_size_ / options_->num_partitions);
    device_ = torch::kCPU;

    buffer_ = new PartitionBuffer(options_->buffer_capacity, options_->num_partitions, options_->fine_to_coarse_ratio, partition_size, dim1_size_, dim0_size_,
                                  dtype_, filename_, options_->prefetching);
}

PartitionBufferStorage::PartitionBufferStorage(string filename, shared_ptr<PartitionBufferOptions> options) {
    filename_ = filename;
    dim0_size_ = 0;
    initialized_ = false;
    loaded_ = false;
    options_ = options;
    dtype_ = options_->dtype;
    int64_t partition_size = ceil((double)dim0_size_ / options_->num_partitions);
    device_ = torch::kCPU;

    buffer_ = new PartitionBuffer(options_->buffer_capacity, options_->num_partitions, options_->fine_to_coarse_ratio, partition_size, dim1_size_, dim0_size_,
                                  dtype_, filename_, options_->prefetching);
}

void PartitionBufferStorage::rangePut(int64_t offset, torch::Tensor values) {
    int fd = open(filename_.c_str(), O_RDWR | IO_FLAGS);
    if (fd == -1) {
        SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);
    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    if (pwrite_wrapper(fd, values.data_ptr(), values.size(0) * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }

    close(fd);
}

void PartitionBufferStorage::append(torch::Tensor values) {
    ios::openmode flags;

    if (dim0_size_ == 0) {
        flags = ios::trunc | ios::binary;
    } else {
        flags = ios::binary | ios_base::app;
    }

    dim0_size_ += values.size(0);
    dim1_size_ = values.size(1);
    dtype_ = values.scalar_type();

    std::ofstream outfile(filename_, flags);

    int dtype_size = get_dtype_size_wrapper(dtype_);

    outfile.write((char *)values.data_ptr(), values.size(0) * values.size(1) * dtype_size);

    outfile.close();
}

PartitionBufferStorage::~PartitionBufferStorage() { delete buffer_; }

void PartitionBufferStorage::load() {
    if (!loaded_ && initialized_) {
        buffer_->load();
        loaded_ = true;
    }
}

void PartitionBufferStorage::write() {
    if (loaded_) {
        buffer_->sync();
    }
}

void PartitionBufferStorage::unload(bool perform_write) {
    if (loaded_) {
        buffer_->unload(perform_write);
        loaded_ = false;
    }
}

torch::Tensor PartitionBufferStorage::indexRead(Indices indices) { return buffer_->indexRead(indices); }

void PartitionBufferStorage::indexAdd(Indices indices, torch::Tensor values) { return buffer_->indexAdd(indices, values); }

torch::Tensor PartitionBufferStorage::range(int64_t offset, int64_t n) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    throw std::runtime_error("");
}

void PartitionBufferStorage::indexPut(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    throw std::runtime_error("");
}

void PartitionBufferStorage::rangePut(int64_t offset, int64_t n, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    throw std::runtime_error("");
}

void PartitionBufferStorage::shuffle() {
    SPDLOG_ERROR("Shuffle not supported for PartitionBufferStorage");
    throw std::runtime_error("");
};

void PartitionBufferStorage::sort(bool src) {
    SPDLOG_ERROR("Sort not supported for PartitionBufferStorage");
    throw std::runtime_error("");
};

MemPartitionBufferStorage::MemPartitionBufferStorage(string filename, int64_t dim0_size, int64_t dim1_size, shared_ptr<PartitionBufferOptions> options, std::vector<torch::Device> devices) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    options_ = options;
    inter_gpu_swap_ = options_->inter_gpu_swap;
    dtype_ = options_->dtype;
    initialized_ = true;
    loaded_ = false;
    int64_t partition_size = ceil((double)dim0_size_ / options_->num_partitions);
    device_ = torch::kCUDA;
    devices_ = devices;
    swap_waiting_ = 0;
    swap_generation_ = 0;
    p2p_exec_ready_ = false;
    device_swap_rounds_.assign(devices_.size(), 0);
    coordinated_swap_round_ = 0;
    epoch_p2p_bytes_ = 0;
    epoch_p2p_partition_copies_ = 0;
    epoch_h2d_bytes_ = 0;
    epoch_d2h_bytes_ = 0;
    for (int i = 0; i < devices_.size(); i ++) {
        MemPartitionBuffer* buffer = new MemPartitionBuffer(options_->buffer_capacity, options_->num_partitions, options_->fine_to_coarse_ratio, partition_size, dim1_size_, dim0_size_,
                                  dtype_, filename_, options_->prefetching, devices_[i], devices_.size());
        buffers_.emplace_back(buffer);
    }
    enablePeerAccess_();
    initializeP2PExecution_();
}

void MemPartitionBufferStorage::enablePeerAccess_() {
    if (!inter_gpu_swap_ || devices_.size() <= 1) {
        return;
    }

    CudaDeviceRestoreGuard device_guard;

    for (int i = 0; i < devices_.size(); i++) {
        int src = devices_[i].index();
        cudaSetDevice(src);
        for (int j = 0; j < devices_.size(); j++) {
            if (i == j) {
                continue;
            }
            int dst = devices_[j].index();
            int can_access_peer = 0;
            cudaError_t can_access_err = cudaDeviceCanAccessPeer(&can_access_peer, src, dst);
            if (can_access_err != cudaSuccess) {
                SPDLOG_WARN("cudaDeviceCanAccessPeer failed for {} -> {}: {}", src, dst, cudaGetErrorString(can_access_err));
                continue;
            }
            if (!can_access_peer) {
                SPDLOG_WARN("P2P disabled for {} -> {} (no peer access)", src, dst);
                continue;
            }
            cudaError_t enable_err = cudaDeviceEnablePeerAccess(dst, 0);
            if (enable_err == cudaErrorPeerAccessAlreadyEnabled) {
                // Clear sticky CUDA error state when peer access was previously enabled.
                cudaGetLastError();
            } else if (enable_err != cudaSuccess) {
                SPDLOG_WARN("cudaDeviceEnablePeerAccess failed for {} -> {}: {}", src, dst, cudaGetErrorString(enable_err));
            }
        }
    }
    SPDLOG_INFO("MemPartitionBufferStorage: inter-GPU swap peer-access initialization complete");
}

void MemPartitionBufferStorage::initializeP2PExecution_() {
    p2p_exec_ready_ = false;
    p2p_streams_.clear();
    p2p_events_.clear();

    if (!inter_gpu_swap_ || devices_.size() <= 1) {
        return;
    }

    CudaDeviceRestoreGuard device_guard;
    p2p_streams_.resize(devices_.size(), nullptr);
    p2p_events_.resize(devices_.size(), nullptr);

    for (int i = 0; i < devices_.size(); i++) {
        cudaSetDevice(devices_[i].index());
        cudaError_t stream_err = cudaStreamCreateWithFlags(&p2p_streams_[i], cudaStreamNonBlocking);
        if (stream_err != cudaSuccess) {
            SPDLOG_WARN("Unable to create P2P stream for device {}: {}", devices_[i].index(), cudaGetErrorString(stream_err));
            destroyP2PExecution_();
            return;
        }

        cudaError_t event_err = cudaEventCreateWithFlags(&p2p_events_[i], cudaEventDisableTiming);
        if (event_err != cudaSuccess) {
            SPDLOG_WARN("Unable to create P2P event for device {}: {}", devices_[i].index(), cudaGetErrorString(event_err));
            destroyP2PExecution_();
            return;
        }
    }

    p2p_exec_ready_ = true;
}

void MemPartitionBufferStorage::destroyP2PExecution_() {
    CudaDeviceRestoreGuard device_guard;

    for (int i = 0; i < devices_.size(); i++) {
        cudaSetDevice(devices_[i].index());

        if (i < static_cast<int>(p2p_events_.size()) && p2p_events_[i] != nullptr) {
            cudaEventDestroy(p2p_events_[i]);
            p2p_events_[i] = nullptr;
        }
        if (i < static_cast<int>(p2p_streams_.size()) && p2p_streams_[i] != nullptr) {
            cudaStreamDestroy(p2p_streams_[i]);
            p2p_streams_[i] = nullptr;
        }
    }

    p2p_events_.clear();
    p2p_streams_.clear();
    p2p_exec_ready_ = false;
}

void MemPartitionBufferStorage::performNextSwapP2P_() {
    CudaDeviceRestoreGuard device_guard;

    int num_devices = static_cast<int>(devices_.size());
    std::vector<torch::Tensor> current_states(num_devices);
    std::vector<torch::Tensor> next_states(num_devices);
    std::vector<bool> has_swap(num_devices, false);

    int max_partition_id = -1;
    for (int device_idx = 0; device_idx < num_devices; device_idx++) {
        has_swap[device_idx] = buffers_[device_idx]->hasSwap();
        current_states[device_idx] = buffers_[device_idx]->getCurrentBufferState().to(torch::kCPU).to(torch::kInt64);
        next_states[device_idx] = has_swap[device_idx] ? buffers_[device_idx]->getNextBufferState().to(torch::kCPU).to(torch::kInt64) : current_states[device_idx];
        for (int i = 0; i < current_states[device_idx].size(0); i++) {
            max_partition_id = std::max(max_partition_id, current_states[device_idx][i].item<int>());
        }
    }

    if (max_partition_id < 0) {
        return;
    }

    std::vector<int> owner_gpu(max_partition_id + 1, -1);
    std::vector<int> owner_slot(max_partition_id + 1, -1);
    for (int device_idx = 0; device_idx < num_devices; device_idx++) {
        for (int slot = 0; slot < current_states[device_idx].size(0); slot++) {
            int part_id = current_states[device_idx][slot].item<int>();
            owner_gpu[part_id] = device_idx;
            owner_slot[part_id] = slot;
        }
    }

    std::vector<torch::Tensor> next_gpu_views(num_devices);
    for (int device_idx = 0; device_idx < num_devices; device_idx++) {
        next_gpu_views[device_idx] = torch::empty_like(buffers_[device_idx]->buffer_tensor_gpu_view_);
    }

    int transfer_ops = 0;
    uint64_t transfer_bytes = 0;
    for (int dst_idx = 0; dst_idx < num_devices; dst_idx++) {
        int dst_device = devices_[dst_idx].index();
        cudaSetDevice(dst_device);
        cudaStream_t dst_stream = 0;
        if (p2p_exec_ready_ && dst_idx < static_cast<int>(p2p_streams_.size()) && p2p_streams_[dst_idx] != nullptr) {
            dst_stream = p2p_streams_[dst_idx];
        }
        char *dst_base = static_cast<char *>(next_gpu_views[dst_idx].data_ptr());
        int64_t dst_slot_bytes = buffers_[dst_idx]->getSlotBytes();
        for (int dst_slot = 0; dst_slot < next_states[dst_idx].size(0); dst_slot++) {
            int part_id = next_states[dst_idx][dst_slot].item<int>();
            if (part_id >= owner_gpu.size() || owner_gpu[part_id] < 0) {
                SPDLOG_ERROR("P2P swap owner lookup failed for partition {}", part_id);
                throw std::runtime_error("");
            }

            int src_idx = owner_gpu[part_id];
            int src_slot = owner_slot[part_id];
            int src_device = devices_[src_idx].index();
            char *src_base = static_cast<char *>(buffers_[src_idx]->buffer_tensor_gpu_view_.data_ptr());
            int64_t src_slot_bytes = buffers_[src_idx]->getSlotBytes();
            int64_t copy_bytes = buffers_[src_idx]->getPartitionBytes(part_id);

            char *dst_ptr = dst_base + dst_slot * dst_slot_bytes;
            char *src_ptr = src_base + src_slot * src_slot_bytes;

            cudaError_t copy_err;
            if (src_idx == dst_idx) {
                copy_err = cudaMemcpyAsync(dst_ptr, src_ptr, copy_bytes, cudaMemcpyDeviceToDevice, dst_stream);
            } else {
                copy_err = cudaMemcpyPeerAsync(dst_ptr, dst_device, src_ptr, src_device, copy_bytes, dst_stream);
                transfer_ops++;
                transfer_bytes += static_cast<uint64_t>(copy_bytes);
            }
            if (copy_err != cudaSuccess) {
                SPDLOG_ERROR("P2P copy failed for p{} {}:{} -> {}:{} with error {}", part_id, src_idx, src_slot, dst_idx, dst_slot,
                             cudaGetErrorString(copy_err));
                throw std::runtime_error("");
            }
        }
    }

    for (int dst_idx = 0; dst_idx < num_devices; dst_idx++) {
        cudaSetDevice(devices_[dst_idx].index());
        cudaError_t sync_err = cudaSuccess;
        if (p2p_exec_ready_ &&
            dst_idx < static_cast<int>(p2p_streams_.size()) &&
            p2p_streams_[dst_idx] != nullptr &&
            dst_idx < static_cast<int>(p2p_events_.size()) &&
            p2p_events_[dst_idx] != nullptr) {
            cudaError_t record_err = cudaEventRecord(p2p_events_[dst_idx], p2p_streams_[dst_idx]);
            if (record_err != cudaSuccess) {
                SPDLOG_ERROR("P2P swap event record failed on device {}: {}", devices_[dst_idx].index(), cudaGetErrorString(record_err));
                throw std::runtime_error("");
            }
            sync_err = cudaEventSynchronize(p2p_events_[dst_idx]);
        } else {
            sync_err = cudaStreamSynchronize(0);
        }
        if (sync_err != cudaSuccess) {
            SPDLOG_ERROR("P2P swap stream/event sync failed on device {}: {}", devices_[dst_idx].index(), cudaGetErrorString(sync_err));
            throw std::runtime_error("");
        }
    }

    for (int device_idx = 0; device_idx < num_devices; device_idx++) {
        if (has_swap[device_idx]) {
            buffers_[device_idx]->buffer_tensor_gpu_view_ = next_gpu_views[device_idx];
            buffers_[device_idx]->finalizeExternalSwap(next_states[device_idx], true);
        }
    }
    uint64_t round_id = 0;
    {
        std::lock_guard<std::mutex> lock(comm_stats_mutex_);
        coordinated_swap_round_++;
        round_id = coordinated_swap_round_;
        epoch_p2p_bytes_ += transfer_bytes;
        epoch_p2p_partition_copies_ += static_cast<uint64_t>(transfer_ops);
    }
    SPDLOG_INFO("Swap comm round {} [P2P] storage={} inter_gpu_partition_copies={} inter_gpu_bytes={} ({:.2f} MiB)",
                round_id, filename_, transfer_ops, transfer_bytes, static_cast<double>(transfer_bytes) / (1024.0 * 1024.0));
    SPDLOG_INFO("P2P coordinated swap completed with {} inter-GPU partition copies", transfer_ops);
}

void MemPartitionBufferStorage::performNextSwap(int32_t device_idx) {
    if (!inter_gpu_swap_ || devices_.size() <= 1) {
        if (!buffers_[device_idx]->hasSwap()) {
            return;
        }
        uint64_t one_way_bytes = static_cast<uint64_t>(buffers_[device_idx]->getSlotBytes()) * static_cast<uint64_t>(options_->buffer_capacity);
        buffers_[device_idx]->performNextSwap();
        uint64_t round_id = 0;
        {
            std::lock_guard<std::mutex> lock(comm_stats_mutex_);
            if (device_idx >= static_cast<int32_t>(device_swap_rounds_.size())) {
                device_swap_rounds_.resize(device_idx + 1, 0);
            }
            device_swap_rounds_[device_idx]++;
            round_id = device_swap_rounds_[device_idx];
            epoch_h2d_bytes_ += one_way_bytes;
            epoch_d2h_bytes_ += one_way_bytes;
        }
        SPDLOG_INFO("Swap comm round {} [CPU<->GPU] storage={} device={} d2h_bytes={} ({:.2f} MiB) h2d_bytes={} ({:.2f} MiB) total_bytes={} ({:.2f} MiB)",
                    round_id, filename_, device_idx,
                    one_way_bytes, static_cast<double>(one_way_bytes) / (1024.0 * 1024.0),
                    one_way_bytes, static_cast<double>(one_way_bytes) / (1024.0 * 1024.0),
                    one_way_bytes * 2, static_cast<double>(one_way_bytes * 2) / (1024.0 * 1024.0));
        return;
    }

    std::unique_lock<std::mutex> lock(swap_mutex_);
    if (!buffers_[device_idx]->hasSwap()) {
        return;
    }

    int participants = 0;
    for (int i = 0; i < devices_.size(); i++) {
        if (buffers_[i]->hasSwap()) {
            participants++;
        }
    }
    if (participants <= 0) {
        return;
    }

    int generation = swap_generation_;
    swap_waiting_++;
    if (swap_waiting_ == participants) {
        performNextSwapP2P_();
        swap_waiting_ = 0;
        swap_generation_++;
        lock.unlock();
        swap_cv_.notify_all();
        return;
    }

    swap_cv_.wait(lock, [this, generation] { return swap_generation_ != generation; });
}

void MemPartitionBufferStorage::logEpochCommStatsAndReset(const std::string &label, int64_t epoch_id) {
    uint64_t p2p_bytes = 0;
    uint64_t p2p_copies = 0;
    uint64_t h2d_bytes = 0;
    uint64_t d2h_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(comm_stats_mutex_);
        p2p_bytes = epoch_p2p_bytes_;
        p2p_copies = epoch_p2p_partition_copies_;
        h2d_bytes = epoch_h2d_bytes_;
        d2h_bytes = epoch_d2h_bytes_;
        epoch_p2p_bytes_ = 0;
        epoch_p2p_partition_copies_ = 0;
        epoch_h2d_bytes_ = 0;
        epoch_d2h_bytes_ = 0;
    }
    uint64_t total_bytes = p2p_bytes + h2d_bytes + d2h_bytes;
    SPDLOG_INFO("Comm summary epoch {} [{}] storage={} p2p_bytes={} ({:.2f} GiB) p2p_partition_copies={} d2h_bytes={} ({:.2f} GiB) h2d_bytes={} ({:.2f} GiB) total_bytes={} ({:.2f} GiB)",
                epoch_id, label, filename_,
                p2p_bytes, static_cast<double>(p2p_bytes) / (1024.0 * 1024.0 * 1024.0),
                p2p_copies,
                d2h_bytes, static_cast<double>(d2h_bytes) / (1024.0 * 1024.0 * 1024.0),
                h2d_bytes, static_cast<double>(h2d_bytes) / (1024.0 * 1024.0 * 1024.0),
                total_bytes, static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0));
}


void MemPartitionBufferStorage::rangePut(int64_t offset, torch::Tensor values) {
    int fd = open(filename_.c_str(), O_RDWR | IO_FLAGS);
    if (fd == -1) {
        SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);
    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    if (pwrite_wrapper(fd, values.data_ptr(), values.size(0) * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }

    close(fd);
}

void MemPartitionBufferStorage::append(torch::Tensor values) {
    ios::openmode flags;

    if (dim0_size_ == 0) {
        flags = ios::trunc | ios::binary;
    } else {
        flags = ios::binary | ios_base::app;
    }

    dim0_size_ += values.size(0);
    dim1_size_ = values.size(1);
    dtype_ = values.scalar_type();

    std::ofstream outfile(filename_, flags);

    int dtype_size = get_dtype_size_wrapper(dtype_);

    outfile.write((char *)values.data_ptr(), values.size(0) * values.size(1) * dtype_size);

    outfile.close();
}

MemPartitionBufferStorage::~MemPartitionBufferStorage() { 
    destroyP2PExecution_();
    for(int i = 0; i < devices_.size(); i ++) {
        delete buffers_[i];
    }
}

void MemPartitionBufferStorage::load() {
    // SPDLOG_INFO("MemPartitionBufferStorage Loading {}", filename_);
    if (!loaded_ && !filename_.empty()) {
        fd_ = open((filename_).c_str(), O_RDWR);
        if (fd_ == -1) {
            SPDLOG_DEBUG("Unable to open {}\nError: {}", filename_, errno);
            return;
        }

        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        data_ = torch::empty({dim0_size_, dim1_size_}, dtype_);
        void* data_ptr_ = data_.data_ptr();
        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pread_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }
        loaded_ = true;
    }
    
    for (int i = 0; i < buffers_.size(); i ++)
        buffers_[i]->load(data_);
}

void MemPartitionBufferStorage::write() {
    if (loaded_ && !filename_.empty()) { 
        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        torch::Tensor data = data_;
        data = data_.to(torch::kCPU);


        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pwrite_wrapper(fd_, data.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }
    }
}

void MemPartitionBufferStorage::unload(bool perform_write) {
    if (loaded_) {
        for (int i = 0; i < buffers_.size(); i ++)
            buffers_[i]->unload(perform_write);

        if (perform_write) {
            write();
            close(fd_);
            data_ = torch::Tensor();
            loaded_ = false;
        }
    }
}

void MemPartitionBufferStorage::unload(bool perform_write, int32_t device_idx) {
    if (loaded_) {
        buffers_[device_idx]->unload(perform_write);
        if (perform_write) {
            write();
            close(fd_);
            data_ = torch::Tensor();
            loaded_ = false;
        }
    }
}

torch::Tensor MemPartitionBufferStorage::indexRead(Indices indices) { 
    if(device_ == torch::kCUDA) {
        torch::Tensor indices_on_device = indices.to(devices_[0], true);
        CudaDeviceRestoreGuard device_guard;
        cudaSetDevice(devices_[0].index());
        return buffers_[0]->indexRead(indices_on_device);
    } else { 
        if (indices.sizes().size() != 1) {
            // TODO: throw invalid input to func exception
            throw std::runtime_error("");
        }

        if (data_.defined()) {
            return data_.index_select(0, indices.to(torch::kCPU));
        } else {
            return torch::Tensor();
        }
    }
}

torch::Tensor MemPartitionBufferStorage::indexRead(Indices indices, int32_t device_idx) { 
    if(device_ == torch::kCUDA) {
        torch::Tensor indices_on_device = indices.to(devices_[device_idx], true);
        CudaDeviceRestoreGuard device_guard;
        cudaSetDevice(devices_[device_idx].index());
        return buffers_[device_idx]->indexRead(indices_on_device);
    } else { 
        if (indices.sizes().size() != 1) {
            // TODO: throw invalid input to func exception
            throw std::runtime_error("");
        }
        // std::cout << data_.device() << std::endl;
        if (data_.defined()) {
            return data_.index_select(0, indices.to(torch::kCPU));
        } else {
            return torch::Tensor();
        }
    }
}


void MemPartitionBufferStorage::indexAdd(Indices indices, torch::Tensor values) { 
    return buffers_[0]->indexAdd(indices, values); 
}

void MemPartitionBufferStorage::indexAdd(Indices indices, torch::Tensor values, int32_t device_idx) { 
    return buffers_[device_idx]->indexAdd(indices, values); 
}

bool MemPartitionBufferStorage::indexAddFused(Indices indices,
                                              torch::Tensor embedding_values,
                                              torch::Tensor state_values,
                                              const std::shared_ptr<MemPartitionBufferStorage> &state_storage,
                                              int32_t device_idx) {
#ifdef GEGE_CUDA
    static const bool dedup_before_scatter = []() {
        const char *env = std::getenv("GEGE_FUSED_SPARSE_DEDUP");
        if (env == nullptr) {
            return false;
        }
        return !(env[0] == '\0' || (env[0] == '0' && env[1] == '\0'));
    }();
    static const int64_t log_every = []() {
        const char *env = std::getenv("GEGE_FUSED_SPARSE_LOG_EVERY");
        if (env == nullptr || env[0] == '\0') {
            return static_cast<int64_t>(200);
        }
        int64_t v = std::atoll(env);
        if (v <= 0) {
            return static_cast<int64_t>(0);
        }
        return v;
    }();
    static std::atomic<int64_t> call_count{0};
    static std::atomic<int64_t> total_pre_nnz{0};
    static std::atomic<int64_t> total_post_nnz{0};

    if (!state_storage) {
        log_fused_sparse_debug("state_storage is null");
        return false;
    }
    if (device_ != torch::kCUDA || state_storage->device_ != torch::kCUDA) {
        log_fused_sparse_debug("storage device is not CUDA");
        return false;
    }
    if (device_idx < 0 || device_idx >= static_cast<int32_t>(buffers_.size()) ||
        device_idx >= static_cast<int32_t>(state_storage->buffers_.size())) {
        log_fused_sparse_debug("device_idx out of range for embedding/state buffers");
        return false;
    }

    auto embedding_table = buffers_[device_idx]->buffer_tensor_gpu_view_;
    auto state_table = state_storage->buffers_[device_idx]->buffer_tensor_gpu_view_;
    if (!embedding_table.defined() || !state_table.defined()) {
        log_fused_sparse_debug("embedding/state GPU table is undefined");
        return false;
    }
    if (!indices.defined() || !embedding_values.defined() || !state_values.defined()) {
        log_fused_sparse_debug("indices/embedding_values/state_values is undefined");
        return false;
    }
    if (!embedding_table.is_cuda() || !state_table.is_cuda() || !embedding_values.is_cuda() || !state_values.is_cuda()) {
        log_fused_sparse_debug("one of embedding/state table or update tensors is not CUDA");
        return false;
    }
    if (embedding_table.device() != state_table.device() || embedding_table.device() != embedding_values.device() ||
        embedding_table.device() != state_values.device()) {
        log_fused_sparse_debug("device mismatch across embedding/state table and update tensors");
        return false;
    }
    if (indices.dim() != 1 || embedding_table.dim() != 2 || state_table.dim() != 2 || embedding_values.dim() != 2 || state_values.dim() != 2) {
        log_fused_sparse_debug("unexpected tensor rank; expected indices=1D and others=2D");
        return false;
    }
    if (indices.scalar_type() != torch::kInt64) {
        log_fused_sparse_debug("indices dtype is not int64");
        return false;
    }
    if (embedding_table.scalar_type() != embedding_values.scalar_type() || embedding_table.scalar_type() != state_values.scalar_type() ||
        embedding_table.scalar_type() != state_table.scalar_type()) {
        log_fused_sparse_debug("dtype mismatch across embedding/state table and update tensors");
        return false;
    }
    if (embedding_table.scalar_type() != torch::kFloat32 && embedding_table.scalar_type() != torch::kFloat64) {
        log_fused_sparse_debug("dtype is neither float32 nor float64");
        return false;
    }
    int64_t nnz = indices.size(0);
    int64_t width = embedding_table.size(1);
    if (state_table.size(1) != width || embedding_values.size(0) != nnz || state_values.size(0) != nnz ||
        embedding_values.size(1) != width || state_values.size(1) != width) {
        log_fused_sparse_debug("shape mismatch between table width/nnz and update tensors");
        return false;
    }
    if (!embedding_table.is_contiguous() || !state_table.is_contiguous() || !embedding_values.is_contiguous() || !state_values.is_contiguous()) {
        log_fused_sparse_debug("at least one tensor is non-contiguous");
        return false;
    }

    int64_t pre_nnz = 0;
    int64_t post_nnz = 0;
    bool ok = gege_fused_sparse_add_cuda(embedding_table,
                                         state_table,
                                         indices,
                                         embedding_values,
                                         state_values,
                                         dedup_before_scatter,
                                         &pre_nnz,
                                         &post_nnz);
    if (!ok) {
        log_fused_sparse_debug("gege_fused_sparse_add_cuda returned false after pre-checks");
        return false;
    }

    int64_t curr_call = call_count.fetch_add(1) + 1;
    total_pre_nnz.fetch_add(pre_nnz);
    total_post_nnz.fetch_add(post_nnz);

    if (dedup_before_scatter && log_every > 0 && (curr_call % log_every == 0)) {
        int64_t agg_pre = total_pre_nnz.load();
        int64_t agg_post = total_post_nnz.load();
        double this_ratio = pre_nnz > 0 ? static_cast<double>(post_nnz) / static_cast<double>(pre_nnz) : 1.0;
        double agg_ratio = agg_pre > 0 ? static_cast<double>(agg_post) / static_cast<double>(agg_pre) : 1.0;
        SPDLOG_INFO(
            "FusedSparse dedup stats: call={} device={} step_pre_nnz={} step_post_nnz={} step_ratio={:.4f} agg_pre_nnz={} agg_post_nnz={} agg_ratio={:.4f}",
            curr_call,
            device_idx,
            pre_nnz,
            post_nnz,
            this_ratio,
            agg_pre,
            agg_post,
            agg_ratio);
    }

    return true;
#else
    (void)indices;
    (void)embedding_values;
    (void)state_values;
    (void)state_storage;
    (void)device_idx;
    return false;
#endif
}

void MemPartitionBufferStorage::rangePut(int64_t offset, int64_t n, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for MemPartitionBufferStorage");
    throw std::runtime_error("");
}

torch::Tensor MemPartitionBufferStorage::range(int64_t offset, int64_t n) {
    SPDLOG_ERROR("Unsupported operation for MemPartitionBufferStorage");
    throw std::runtime_error("");
}

void MemPartitionBufferStorage::indexPut(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for MemPartitionBufferStorage");
    throw std::runtime_error("");
}

void MemPartitionBufferStorage::shuffle() {
    SPDLOG_ERROR("Unsupported operation for MemPartitionBufferStorage");
    throw std::runtime_error("");
};

void MemPartitionBufferStorage::sort(bool src) {
    SPDLOG_ERROR("Sort not supported for MemPartitionBufferStorage");
    throw std::runtime_error("");
};

FlatFile::FlatFile(string filename, int64_t dim0_size, int64_t dim1_size, torch::Dtype dtype, bool alloc) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    dtype_ = dtype;
    initialized_ = true;
    loaded_ = false;
    device_ = torch::kCPU;

    if (alloc) {
        int64_t dtype_size = 0;

        if (dtype_ == torch::kFloat64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kFloat32) {
            dtype_size = 4;
        } else if (dtype_ == torch::kFloat16) {
            dtype_size = 2;
        } else if (dtype_ == torch::kInt64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kInt32) {
            dtype_size = 4;
        }

        std::ofstream ofs(filename_, std::ios::binary | std::ios::out);
        ofs.seekp(dim0_size_ * dim1_size_ * dtype_size - 1);
        ofs.write("", 1);
        ofs.close();
    }
}

FlatFile::FlatFile(string filename, torch::Tensor data) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = data.size(1);
    dtype_ = data.scalar_type();
    loaded_ = false;
    append(data);
    initialized_ = true;
    device_ = torch::kCPU;
}

FlatFile::FlatFile(string filename, torch::Dtype dtype) {
    filename_ = filename;
    dim0_size_ = 0;
    initialized_ = false;
    loaded_ = false;
    dtype_ = dtype;
    device_ = torch::kCPU;
}

void FlatFile::rangePut(int64_t offset, torch::Tensor values) {
    if (!values.defined() || (dim0_size_ != 0 && (values.size(0) + offset > dim0_size_ || values.size(1) != dim1_size_))) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);

    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    if (pwrite_wrapper(fd_, values.data_ptr(), values.size(0) * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }
}

void FlatFile::append(torch::Tensor values) {
    ios::openmode flags = dim0_size_ == 0 ? ios::trunc | ios::binary : ios::binary | ios_base::app;

    dim0_size_ += values.size(0);
    dim1_size_ = values.size(1);
    dtype_ = values.scalar_type();

    std::ofstream outfile(filename_, flags);

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);

    outfile.write((char *)values.data_ptr(), values.size(0) * values.size(1) * dtype_size);
    outfile.close();
}

void FlatFile::load() {
    if (!loaded_ && initialized_) {
        fd_ = open(filename_.c_str(), O_RDWR | IO_FLAGS);
        if (fd_ == -1) {
            SPDLOG_DEBUG("Unable to open {}\nError: {}", filename_, errno);
            return;
        }
        loaded_ = true;
    }
}

void FlatFile::write() { return; }

void FlatFile::unload(bool perform_write) {
    (void)perform_write;
    if (loaded_) {
        close(fd_);
        loaded_ = false;
    }
}

torch::Tensor FlatFile::indexRead(Indices indices) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, only sequential access is supported");
    throw std::runtime_error("");
}

void FlatFile::indexAdd(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, only sequential access is supported");
    throw std::runtime_error("");
}

void FlatFile::indexPut(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, only sequential access is supported");
    throw std::runtime_error("");
}

void FlatFile::move(string new_filename) {
    unload(false);

    renameFile(filename_, new_filename);

    load();
}

void FlatFile::copy(string new_filename, bool rename) {
    unload(false);

    copyFile(filename_, new_filename);

    if (rename) {
        filename_ = new_filename;
    }
    load();
}

torch::Tensor FlatFile::range(int64_t offset, int64_t n) {
    if (n + offset > dim0_size_) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }
    int dtype_size = get_dtype_size_wrapper(dtype_);

    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    torch::Tensor output_tensor = torch::empty({n, dim1_size_}, dtype_);
    if (pread_wrapper(fd_, output_tensor.data_ptr(), n * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }
    return output_tensor;
}

void FlatFile::rangePut(int64_t offset, int64_t n, torch::Tensor values) {
    int dtype_size = get_dtype_size_wrapper(dtype_);

    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    if (pwrite_wrapper(fd_, values.data_ptr(), n * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }
}

void FlatFile::shuffle() {
    bool loaded = loaded_;
    if (!loaded) {
        load();
    }
    if (edge_bucket_sizes_.empty()) {
        int64_t offset = 0;
        int64_t curr_size = 0;
        while (offset < dim0_size_) {
            if (dim0_size_ - offset < MAX_SHUFFLE_SIZE) {
                curr_size = dim0_size_ - offset;
            } else {
                curr_size = MAX_SHUFFLE_SIZE;
            }
            torch::Tensor chunk = range(offset, curr_size);
            auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
            chunk.copy_(chunk.index_select(0, torch::randperm(chunk.size(0), opts)));
            rangePut(offset, chunk);
            offset += curr_size;
        }
    } else {
        int64_t offset = 0;
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = range(offset, *itr);
            edge_bucket.copy_(edge_bucket.index_select(0, torch::randperm(edge_bucket.size(0), opts)));
            rangePut(offset, edge_bucket);
            offset += *itr;
        }
    }
    if (!loaded) {
        unload(true);
    }
}

void FlatFile::sort(bool src) {
    // function for sorting flat file storing edges
    int sort_dim = 0;
    if (!src) {
        sort_dim = -1;
    }

    bool loaded = loaded_;
    if (!loaded) {
        load();
    }
    if (edge_bucket_sizes_.empty()) {
        int64_t offset = 0;
        int64_t curr_size = 0;
        while (offset < dim0_size_) {
            if (dim0_size_ - offset < MAX_SORT_SIZE) {
                curr_size = dim0_size_ - offset;
            } else {
                curr_size = MAX_SORT_SIZE;
            }

            torch::Tensor chunk = range(offset, curr_size);
            // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
            chunk.copy_(chunk.index_select(0, torch::argsort(chunk.select(1, sort_dim))));
            rangePut(offset, chunk);
            offset += curr_size;
        }
    } else {
        int64_t offset = 0;
        // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = range(offset, *itr);
            edge_bucket.copy_(edge_bucket.index_select(0, torch::argsort(edge_bucket.select(1, sort_dim))));
            rangePut(offset, edge_bucket);
            offset += *itr;
        }
    }
    if (!loaded) {
        unload(true);
    }
}

void FlatFile::mem_load() {
    if (!loaded_) {
        fd_ = open((filename_).c_str(), O_RDWR);
        if (fd_ == -1) {
            SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }

        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        data_ = torch::empty({dim0_size_, dim1_size_}, dtype_);
        SPDLOG_DEBUG("Initialized memory edges");
        process_mem_usage();

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pread_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }

        SPDLOG_DEBUG("Read edges from disk");
        process_mem_usage();

        loaded_ = true;
    }
}

void FlatFile::mem_unload(bool write) {
    if (loaded_) {
        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (write) {
            if (pwrite_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
                SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
                throw std::runtime_error("");
            }
        }

        close(fd_);

        SPDLOG_DEBUG("Edges written");
        process_mem_usage();
        loaded_ = false;
        process_mem_usage();
        data_ = torch::Tensor();
        SPDLOG_DEBUG("Nulled tensor and pointer");
        process_mem_usage();
    }
}

InMemory::InMemory(string filename, int64_t dim0_size, int64_t dim1_size, torch::Dtype dtype, torch::Device device) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    dtype_ = dtype;
    initialized_ = true;
    loaded_ = false;
    device_ = device;
}

InMemory::InMemory(string filename, torch::Tensor data, torch::Device device) {
    filename_ = filename;
    dim0_size_ = data.size(0);
    dim1_size_ = data.size(1);
    dtype_ = data.scalar_type();
    device_ = device;
    loaded_ = false;

    torch::Tensor temp = data.to(torch::kCPU);

    std::ofstream outfile(filename_, ios::out | ios::binary);

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);

    outfile.write((char *)temp.data_ptr(), data.size(0) * data.size(1) * dtype_size);

    outfile.close();
}

InMemory::InMemory(string filename, torch::Dtype dtype) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = 0;
    initialized_ = false;
    dtype_ = dtype;
    device_ = torch::kCPU;
    loaded_ = false;
}

InMemory::InMemory(torch::Tensor data) {
    if (data.sizes().size() == 2) {
        dim0_size_ = data.size(0);
        dim1_size_ = data.size(1); 
    } else if (data.sizes().size() == 1) {
        dim0_size_ = data.size(0);
        dim1_size_ = 1;
    } else {
        throw GegeRuntimeException("Tensor must have 1 or two dimensions");
    }

    filename_ = "";
    data_ = data.reshape({dim0_size_, dim1_size_});

    initialized_ = true;
    dtype_ = data.scalar_type();
    device_ = data.device();
    loaded_ = true;
}

void InMemory::load() {
    if (!loaded_ && !filename_.empty()) {
        fd_ = open((filename_).c_str(), O_RDWR);
        if (fd_ == -1) {
            SPDLOG_DEBUG("Unable to open {}\nError: {}", filename_, errno);
            return;
        }
        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        data_ = torch::empty({dim0_size_, dim1_size_}, dtype_);

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pread_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }

        if (device_ == torch::kCUDA) {
            data_ = data_.to(device_);
        }

        loaded_ = true;
    }
}

void InMemory::write() {
    if (loaded_ && !filename_.empty()) {
        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        torch::Tensor data = data_;
        if (device_ == torch::kCUDA) {
            data = data_.to(torch::kCPU);
        }

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pwrite_wrapper(fd_, data.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }
    }
}

void InMemory::unload(bool perform_write) {
    if (loaded_ && !filename_.empty()) {
        if (perform_write) {
            write();
        }

        // close(fd_);
        // loaded_ = false;
        // data_ = torch::Tensor();
    }
}

torch::Tensor InMemory::indexRead(Indices indices) {
    if (indices.sizes().size() != 1) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }

    if (data_.defined()) {
        return data_.index_select(0, indices.to(device_));
    } else {
        return torch::Tensor();
    }
}

void InMemory::indexAdd(Indices indices, torch::Tensor values) {
    if (!values.defined() || indices.sizes().size() != 1 || indices.size(0) != values.size(0) || data_.size(1) != values.size(1)) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }
    if (values.device().is_cuda()) {
        data_.index_add_(0, indices, values);
    } else {
        // assumes this operation is only used on float valued data.
        auto data_accessor = data_.accessor<float, 2>();
        auto ids_accessor = indices.accessor<int64_t, 1>();
        auto values_accessor = values.accessor<float, 2>();

        int d = values.size(1);
        int64_t size = indices.size(0);
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++) {
            for (int j = 0; j < d; j++) {
                data_accessor[ids_accessor[i]][j] += values_accessor[i][j];
            }
        }
    }
}

void InMemory::indexPut(Indices indices, torch::Tensor values) {
    if (!values.defined() || indices.sizes().size() != 1 || indices.size(0) != values.size(0) || data_.size(1) != values.size(1)) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }
    if (values.device().is_cuda()) {
        data_[indices] = values;
    } else {
        // assumes this operation is only used on float valued data.
        auto data_accessor = data_.accessor<float, 2>();
        auto ids_accessor = indices.accessor<int64_t, 1>();
        auto values_accessor = values.accessor<float, 2>();

        int d = values.size(1);
        int64_t size = indices.size(0);
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++) {
            for (int j = 0; j < d; j++) {
                data_accessor[ids_accessor[i]][j] = values_accessor[i][j];
            }
        }
    }
}

torch::Tensor InMemory::range(int64_t offset, int64_t n) {
    if (n + offset > dim0_size_) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }
    return data_.narrow(0, offset, n);
}

void InMemory::rangePut(int64_t offset, int64_t n, torch::Tensor values) { data_.narrow(0, offset, n).copy_(values); }

void InMemory::shuffle() {
    bool loaded = loaded_;
    if (!loaded) {
        load();

        // may cause silent failures
        if (!loaded_) {
            return;
        }
    }

    // full shuffle
    if (edge_bucket_sizes_.empty()) {
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        data_.copy_(data_.index_select(0, torch::randperm(dim0_size_, opts)));
    }
    // shuffle within edge buckets
    else {
        int64_t start = 0;
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = data_.narrow(0, start, *itr);
            data_.narrow(0, start, *itr) = (edge_bucket.index_select(0, torch::randperm(edge_bucket.size(0), opts)));
            start += *itr;
        }
    }
    // if (!loaded) {
    //     unload(true);
    // }
}

// void InMemory::shuffle() {
//     auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
//     torch::Tenosr perm = torch::randperm(dim0_size_, opts);


// }

void InMemory::sort(bool src) {
    // function for sorting in memory edges
    int sort_dim = 0;
    if (!src) {
        sort_dim = -1;
    }

    bool loaded = loaded_;
    if (!loaded) {
        load();

        // may cause silent failures
        if (!loaded_) {
            return;
        }
    }

    // full sort
    if (edge_bucket_sizes_.empty()) {
        // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        data_.copy_(data_.index_select(0, torch::argsort(data_.select(1, sort_dim))));
    }
    // sort within edge buckets
    else {
        int64_t start = 0;
        // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = data_.narrow(0, start, *itr);
            data_.narrow(0, start, *itr) = (edge_bucket.index_select(0, torch::argsort(edge_bucket.select(1, sort_dim))));
            start += *itr;
        }
    }
    if (!loaded) {
        unload(true);
    }
}
