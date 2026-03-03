#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace runtime_profile {

struct RuntimeProfileSnapshot {
    int64_t load_gpu_params_ns = 0;
    int64_t load_gpu_params_calls = 0;
    int64_t update_embeddings_gpu_ns = 0;
    int64_t update_embeddings_gpu_calls = 0;
    int64_t update_embeddings_host_ns = 0;
    int64_t update_embeddings_host_calls = 0;
    int64_t storage_get_embeddings_ns = 0;
    int64_t storage_get_embeddings_calls = 0;
    int64_t storage_get_state_ns = 0;
    int64_t storage_get_state_calls = 0;
    int64_t buffer_index_read_ns = 0;
    int64_t buffer_index_read_calls = 0;
    int64_t buffer_index_add_ns = 0;
    int64_t buffer_index_add_calls = 0;
    int64_t batch_to_device_ns = 0;
    int64_t batch_to_device_calls = 0;
    int64_t batch_to_host_ns = 0;
    int64_t batch_to_host_calls = 0;

    bool empty() const {
        return load_gpu_params_calls == 0 &&
               update_embeddings_gpu_calls == 0 &&
               update_embeddings_host_calls == 0 &&
               storage_get_embeddings_calls == 0 &&
               storage_get_state_calls == 0 &&
               buffer_index_read_calls == 0 &&
               buffer_index_add_calls == 0 &&
               batch_to_device_calls == 0 &&
               batch_to_host_calls == 0;
    }
};

inline std::atomic<int64_t> load_gpu_params_ns{0};
inline std::atomic<int64_t> load_gpu_params_calls{0};

inline std::atomic<int64_t> update_embeddings_gpu_ns{0};
inline std::atomic<int64_t> update_embeddings_gpu_calls{0};

inline std::atomic<int64_t> update_embeddings_host_ns{0};
inline std::atomic<int64_t> update_embeddings_host_calls{0};

inline std::atomic<int64_t> storage_get_embeddings_ns{0};
inline std::atomic<int64_t> storage_get_embeddings_calls{0};

inline std::atomic<int64_t> storage_get_state_ns{0};
inline std::atomic<int64_t> storage_get_state_calls{0};

inline std::atomic<int64_t> buffer_index_read_ns{0};
inline std::atomic<int64_t> buffer_index_read_calls{0};

inline std::atomic<int64_t> buffer_index_add_ns{0};
inline std::atomic<int64_t> buffer_index_add_calls{0};

inline std::atomic<int64_t> batch_to_device_ns{0};
inline std::atomic<int64_t> batch_to_device_calls{0};

inline std::atomic<int64_t> batch_to_host_ns{0};
inline std::atomic<int64_t> batch_to_host_calls{0};

class ScopedTimer {
  public:
    ScopedTimer(std::atomic<int64_t> &total_ns, std::atomic<int64_t> &calls) : total_ns_(total_ns), calls_(calls), start_(std::chrono::steady_clock::now()) {
        calls_.fetch_add(1, std::memory_order_relaxed);
    }

    ~ScopedTimer() {
        auto end = std::chrono::steady_clock::now();
        int64_t delta_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
        total_ns_.fetch_add(delta_ns, std::memory_order_relaxed);
    }

  private:
    std::atomic<int64_t> &total_ns_;
    std::atomic<int64_t> &calls_;
    std::chrono::steady_clock::time_point start_;
};

inline RuntimeProfileSnapshot captureAndReset() {
    RuntimeProfileSnapshot snap;

    snap.load_gpu_params_ns = load_gpu_params_ns.exchange(0, std::memory_order_relaxed);
    snap.load_gpu_params_calls = load_gpu_params_calls.exchange(0, std::memory_order_relaxed);
    snap.update_embeddings_gpu_ns = update_embeddings_gpu_ns.exchange(0, std::memory_order_relaxed);
    snap.update_embeddings_gpu_calls = update_embeddings_gpu_calls.exchange(0, std::memory_order_relaxed);
    snap.update_embeddings_host_ns = update_embeddings_host_ns.exchange(0, std::memory_order_relaxed);
    snap.update_embeddings_host_calls = update_embeddings_host_calls.exchange(0, std::memory_order_relaxed);
    snap.storage_get_embeddings_ns = storage_get_embeddings_ns.exchange(0, std::memory_order_relaxed);
    snap.storage_get_embeddings_calls = storage_get_embeddings_calls.exchange(0, std::memory_order_relaxed);
    snap.storage_get_state_ns = storage_get_state_ns.exchange(0, std::memory_order_relaxed);
    snap.storage_get_state_calls = storage_get_state_calls.exchange(0, std::memory_order_relaxed);
    snap.buffer_index_read_ns = buffer_index_read_ns.exchange(0, std::memory_order_relaxed);
    snap.buffer_index_read_calls = buffer_index_read_calls.exchange(0, std::memory_order_relaxed);
    snap.buffer_index_add_ns = buffer_index_add_ns.exchange(0, std::memory_order_relaxed);
    snap.buffer_index_add_calls = buffer_index_add_calls.exchange(0, std::memory_order_relaxed);
    snap.batch_to_device_ns = batch_to_device_ns.exchange(0, std::memory_order_relaxed);
    snap.batch_to_device_calls = batch_to_device_calls.exchange(0, std::memory_order_relaxed);
    snap.batch_to_host_ns = batch_to_host_ns.exchange(0, std::memory_order_relaxed);
    snap.batch_to_host_calls = batch_to_host_calls.exchange(0, std::memory_order_relaxed);

    return snap;
}

} // namespace runtime_profile
