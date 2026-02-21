#include "common/util.h"

#include <fstream>
#include <unistd.h>

#include <iostream>

#include "reporting/logger.h"

void assert_no_nans(torch::Tensor values) {
    if (torch::isnan(values).any().item<bool>()) {
        throw GegeRuntimeException("Tensor contains Nans");
    }
}

void assert_no_neg(torch::Tensor values) {
    if ((values.le(-1)).any().item<bool>()) {
        throw GegeRuntimeException("Tensor contains negative values");
    }
}

void assert_in_range(torch::Tensor values, int64_t start, int64_t end) {
    if ((values.ge(start) & values.le(end)).any().item<bool>()) {
        throw GegeRuntimeException("Tensor contains is not in range: " + std::to_string(start) + "-" + std::to_string(end));
    }
}

void process_mem_usage() {
    double vm_usage = 0.0;
    double resident_set = 0.0;

    // the two fields we want
    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >>
            ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;  // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;

    SPDLOG_DEBUG("VM Usage: {}GB. RSS: {}GB", vm_usage / pow(2, 20), resident_set / pow(2, 20));
}

void *memset_wrapper(void *ptr, int value, int64_t num) {
    int64_t curr_bytes = 0;
    int64_t local_offset = 0;

    while (local_offset < num) {
        curr_bytes = num - local_offset;
        if (curr_bytes > 1e9) {
            curr_bytes = 1e9;
        }

        memset((char *)ptr + local_offset, value, curr_bytes);

        local_offset += curr_bytes;
    }

    return ptr;
}

void *memcpy_wrapper(void *dest, const void *src, int64_t count) {
    int64_t curr_bytes = 0;
    int64_t local_offset = 0;

    while (local_offset < count) {
        curr_bytes = count - local_offset;
        if (curr_bytes > 1e9) {
            curr_bytes = 1e9;
        }

        memcpy((char *)dest + local_offset, (char *)src + local_offset, curr_bytes);

        local_offset += curr_bytes;
    }

    return dest;
}

int64_t pread_wrapper(int fd, void *buf, int64_t count, int64_t offset) {
    int64_t curr_bytes = 0;
    int64_t local_offset = 0;

    while (local_offset < count) {
        curr_bytes = count - local_offset;
        if (curr_bytes > 1e9) {
            curr_bytes = 1e9;
        }

        if (pread(fd, (char *)buf + local_offset, curr_bytes, offset + local_offset) == -1) {
            return -1;
        }

        local_offset += curr_bytes;
    }

    return count;
}

int64_t pwrite_wrapper(int fd, const void *buf, int64_t count, int64_t offset) {
    int64_t curr_bytes = 0;
    int64_t local_offset = 0;

    while (local_offset < count) {
        curr_bytes = count - local_offset;
        if (curr_bytes > 1e9) {
            curr_bytes = 1e9;
        }

        if (pwrite(fd, (char *)buf + local_offset, curr_bytes, offset + local_offset) == -1) {
            return -1;
        }

        local_offset += curr_bytes;
    }

    return count;
}

int64_t get_dtype_size_wrapper(torch::Dtype dtype_) {
    if (dtype_ == torch::kFloat64) {
        return 8;
    }
    if (dtype_ == torch::kFloat32) {
        return 4;
    }
    if (dtype_ == torch::kFloat16) {
        return 2;
    }
    if (dtype_ == torch::kInt64) {
        return 8;
    }
    if (dtype_ == torch::kInt32) {
        return 4;
    }

    SPDLOG_ERROR("Unable to determine dtype_size_ for given dtype_ {}", dtype_);
    throw std::runtime_error("");
}

std::string get_directory(std::string filename) {
    assert(!filename.empty());

    string directory;
    const size_t last_slash_idx = filename.rfind('/');
    if (std::string::npos != last_slash_idx) {
        directory = filename.substr(0, last_slash_idx);
    }

    return directory;
}

namespace {
void validate_tensor_map_inputs(const std::vector<torch::Tensor> &unmapped_tensors) {
    for (const auto &tensor : unmapped_tensors) {
        if (tensor.sizes().size() > 1) {
            throw GegeRuntimeException("Input tensors must be 1D");
        }
    }
}

std::vector<torch::Tensor> split_mapped_tensor(const torch::Tensor &mapped_all_ids, const std::vector<torch::Tensor> &unmapped_tensors) {
    std::vector<torch::Tensor> mapped_tensors;
    int64_t offset = 0;
    for (const auto &tensor : unmapped_tensors) {
        int64_t size = tensor.size(0);
        mapped_tensors.emplace_back(mapped_all_ids.narrow(0, offset, size));
        offset += size;
    }
    return mapped_tensors;
}
}  // namespace

std::tuple<torch::Tensor, std::vector<torch::Tensor>> map_tensors(std::vector<torch::Tensor> unmapped_tensors) {
    validate_tensor_map_inputs(unmapped_tensors);

    torch::Tensor all_ids = torch::cat(unmapped_tensors);

    auto unique_tup = torch::_unique2(all_ids, true, true, false);

    torch::Tensor map = std::get<0>(unique_tup);
    torch::Tensor mapped_all_ids = std::get<1>(unique_tup);

    return std::forward_as_tuple(map, split_mapped_tensor(mapped_all_ids, unmapped_tensors));
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>> map_tensors_dense_range(std::vector<torch::Tensor> unmapped_tensors, int64_t id_space_size) {
    validate_tensor_map_inputs(unmapped_tensors);

    if (id_space_size <= 0) {
        throw GegeRuntimeException("map_tensors_dense_range id_space_size must be positive");
    }

    torch::Tensor all_ids = torch::cat(unmapped_tensors);
    if (all_ids.scalar_type() != torch::kInt64) {
        all_ids = all_ids.to(torch::kInt64);
    }

    if (all_ids.numel() == 0) {
        torch::Tensor empty_map = torch::empty({0}, all_ids.options().dtype(torch::kInt64));
        return std::forward_as_tuple(empty_map, split_mapped_tensor(empty_map, unmapped_tensors));
    }

    int64_t min_id = all_ids.min().item<int64_t>();
    int64_t max_id = all_ids.max().item<int64_t>();
    if (min_id < 0 || max_id >= id_space_size) {
        throw GegeRuntimeException("map_tensors_dense_range found IDs outside [0, id_space_size)");
    }

    auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(all_ids.device());
    torch::Tensor present = torch::zeros({id_space_size}, bool_opts);
    present.index_fill_(0, all_ids, true);

    torch::Tensor map = torch::nonzero(present).flatten(0, 1).to(torch::kInt64);

    auto idx_opts = torch::TensorOptions().dtype(torch::kInt64).device(all_ids.device());
    torch::Tensor inverse_lut = torch::full({id_space_size}, -1, idx_opts);
    inverse_lut.index_put_({map}, torch::arange(map.size(0), idx_opts));

    torch::Tensor mapped_all_ids = inverse_lut.index_select(0, all_ids);
    return std::forward_as_tuple(map, split_mapped_tensor(mapped_all_ids, unmapped_tensors));
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>> map_tensors_identity_range(std::vector<torch::Tensor> unmapped_tensors, int64_t id_space_size) {
    validate_tensor_map_inputs(unmapped_tensors);

    if (id_space_size <= 0) {
        throw GegeRuntimeException("map_tensors_identity_range id_space_size must be positive");
    }

    torch::Tensor all_ids = torch::cat(unmapped_tensors);
    if (all_ids.scalar_type() != torch::kInt64) {
        all_ids = all_ids.to(torch::kInt64);
    }

    if (all_ids.numel() > 0) {
        int64_t min_id = all_ids.min().item<int64_t>();
        int64_t max_id = all_ids.max().item<int64_t>();
        if (min_id < 0 || max_id >= id_space_size) {
            throw GegeRuntimeException("map_tensors_identity_range found IDs outside [0, id_space_size)");
        }
    }

    auto idx_opts = torch::TensorOptions().dtype(torch::kInt64).device(all_ids.device());
    torch::Tensor map = torch::arange(id_space_size, idx_opts);
    return std::forward_as_tuple(map, split_mapped_tensor(all_ids, unmapped_tensors));
}

// TODO this function uses a searchsorted to find the approriate value in the map tensor
// this can be made faster on the cpu by using an std::map to perform lookups
std::vector<torch::Tensor> apply_tensor_map(torch::Tensor map, std::vector<torch::Tensor> unmapped_tensors) {
    for (auto tensor : unmapped_tensors) {
        if (tensor.sizes().size() > 1) {
            throw GegeRuntimeException("Input tensors must be 1D");
        }
    }

    std::vector<torch::Tensor> mapped_tensors;

    for (auto tensor : unmapped_tensors) {
        mapped_tensors.emplace_back(torch::searchsorted(map, tensor));
    }

    return mapped_tensors;
}
