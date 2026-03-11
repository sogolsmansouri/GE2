#include "common/util.h"

#include <fstream>
#include <unistd.h>
#include <filesystem>

#include <iostream>

#include "reporting/logger.h"
#include <nvtx3/nvtx3.hpp>
#include <mutex>
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
/*
    KG related tensor mapping.
    2 major operations:
    1. tensor.unique()
    2. remapping

*/
// std::tuple<torch::Tensor, std::vector<torch::Tensor>> map_tensors(std::vector<torch::Tensor> unmapped_tensors) {
//     for (auto tensor : unmapped_tensors) {
//         if (tensor.sizes().size() > 1) {
//             throw GegeRuntimeException("Input tensors must be 1D");
//         }
//     }

//     torch::Tensor all_ids = torch::cat(unmapped_tensors);

//     auto unique_tup = torch::_unique2(all_ids, true, true, false);

//     torch::Tensor map = std::get<0>(unique_tup);
//     torch::Tensor mapped_all_ids = std::get<1>(unique_tup);

//     std::vector<torch::Tensor> mapped_tensors;

//     int64_t offset = 0;
//     int64_t size;
//     for (auto tensor : unmapped_tensors) {
//         size = tensor.size(0);
//         mapped_tensors.emplace_back(mapped_all_ids.narrow(0, offset, size));
//         offset += size;
//     }

//     return std::forward_as_tuple(map, mapped_tensors);
// }

/*
    NVTX tag version
*/

// std::tuple<torch::Tensor, std::vector<torch::Tensor>> map_tensors(std::vector<torch::Tensor> unmapped_tensors) {
//     {
//         nvtx3::scoped_range r{"map_tensors_validate_inputs"};
//         for (auto tensor : unmapped_tensors) {
//             if (tensor.sizes().size() > 1) {
//                 throw GegeRuntimeException("Input tensors must be 1D");
//             }
//         }
//     }

//     torch::Tensor all_ids;
//     {
//         nvtx3::scoped_range r{"map_tensors_cat"};
//         all_ids = torch::cat(unmapped_tensors);
//     }

//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> unique_tup;
//     {

//         /*
//             torch.unique is the major bottleneck in mapping.
//             avg_time >> medium_time
            
//         */
//         nvtx3::scoped_range r{"map_tensors_unique2"};
//         unique_tup = torch::_unique2(all_ids, true, true, false);
//     }

//     torch::Tensor map;
//     torch::Tensor mapped_all_ids;
//     {
//         nvtx3::scoped_range r{"map_tensors_unpack_unique"};
//         map = std::get<0>(unique_tup);
//         mapped_all_ids = std::get<1>(unique_tup);
//     }

//     std::vector<torch::Tensor> mapped_tensors;
//     {
//         nvtx3::scoped_range r{"map_tensors_split_inverse"};
//         int64_t offset = 0;
//         int64_t size;
//         for (auto tensor : unmapped_tensors) {
//             size = tensor.size(0);
//             mapped_tensors.emplace_back(mapped_all_ids.narrow(0, offset, size));
//             offset += size;
//         }
//     }

//     return std::forward_as_tuple(map, mapped_tensors);
// }

/*
    NVTX+_unique2 profiling version
    my_map_tensors add a new parameter: train_
*/
// std::tuple<torch::Tensor, std::vector<torch::Tensor>> my_map_tensors(std::vector<torch::Tensor> unmapped_tensors, bool train_) {
//     static int64_t debug_counter = 0;
//     static std::mutex log_mutex;

//     const char* mode_str = train_ ? "train" : "eval";

//     {
//         std::string tag = std::string("map_tensors_") + mode_str + "_validate_inputs";
//         nvtx3::scoped_range r{tag.c_str()};

//         for (auto tensor : unmapped_tensors) {
//             if (tensor.sizes().size() > 1) {
//                 throw GegeRuntimeException("Input tensors must be 1D");
//             }
//         }
//     }

//     torch::Tensor all_ids;
//     {
//         std::string tag = std::string("map_tensors_") + mode_str + "_cat";
//         nvtx3::scoped_range r{tag.c_str()};
//         all_ids = torch::cat(unmapped_tensors);
//     }

//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> unique_tup;
//     {
//         std::string tag = std::string("map_tensors_") + mode_str + "_unique2";
//         nvtx3::scoped_range r{tag.c_str()};
//         unique_tup = torch::_unique2(all_ids, true, true, false);//ascending order+return inverse = descending order
//     }

//     torch::Tensor map;
//     torch::Tensor mapped_all_ids;
//     {
//         std::string tag = std::string("map_tensors_") + mode_str + "_unpack_unique";
//         nvtx3::scoped_range r{tag.c_str()};
//         map = std::get<0>(unique_tup);
//         mapped_all_ids = std::get<1>(unique_tup);
//     }

//     {
//         std::string tag = std::string("map_tensors_") + mode_str + "_stats";
//         nvtx3::scoped_range r{tag.c_str()};
//         std::lock_guard<std::mutex> guard(log_mutex);

//         int64_t num_all_ids = all_ids.numel();
//         int64_t num_unique_ids = map.numel();

//         double unique_ratio = (num_all_ids > 0)
//             ? static_cast<double>(num_unique_ids) / static_cast<double>(num_all_ids)
//             : 0.0;
//         double dup_ratio = 1.0 - unique_ratio;

//         int64_t edge_src_num = unmapped_tensors.size() > 0 ? unmapped_tensors[0].numel() : 0;
//         int64_t edge_dst_num = unmapped_tensors.size() > 1 ? unmapped_tensors[1].numel() : 0;
//         int64_t src_neg_num  = unmapped_tensors.size() > 2 ? unmapped_tensors[2].numel() : 0;
//         int64_t dst_neg_num  = unmapped_tensors.size() > 3 ? unmapped_tensors[3].numel() : 0;

//         std::ofstream ofs("profiles/getBatch_profiling/single_GPU/unique_rate.txt", std::ios::app);

//         ofs << "[map_tensors][" << debug_counter << "] "
//             << "mode=" << mode_str
//             << ", all_ids=" << num_all_ids
//             << ", unique_ids=" << num_unique_ids
//             << ", unique_ratio=" << unique_ratio
//             << ", dup_ratio=" << dup_ratio
//             << ", edge_src=" << edge_src_num
//             << ", edge_dst=" << edge_dst_num
//             << ", src_neg=" << src_neg_num
//             << ", dst_neg=" << dst_neg_num
//             << "\n";

//         debug_counter++;
//     }

//     std::vector<torch::Tensor> mapped_tensors;
//     {
//         std::string tag = std::string("map_tensors_") + mode_str + "_split_inverse";
//         nvtx3::scoped_range r{tag.c_str()};

//         int64_t offset = 0;
//         int64_t size;
//         for (auto tensor : unmapped_tensors) {
//             size = tensor.size(0);
//             mapped_tensors.emplace_back(mapped_all_ids.narrow(0, offset, size));
//             offset += size;
//         }
//     }

//     return std::forward_as_tuple(map, mapped_tensors);
// }

/*

    Get data slice
*/

std::tuple<torch::Tensor, std::vector<torch::Tensor>> my_map_tensors(std::vector<torch::Tensor> unmapped_tensors, bool train_) {
    static int64_t debug_counter = 0;
    static std::mutex log_mutex;

    const char* mode_str = train_ ? "train" : "eval";

    {
        std::string tag = std::string("map_tensors_") + mode_str + "_validate_inputs";
        nvtx3::scoped_range r{tag.c_str()};

        for (auto tensor : unmapped_tensors) {
            if (tensor.sizes().size() > 1) {
                throw GegeRuntimeException("Input tensors must be 1D");
            }
        }
    }

    torch::Tensor all_ids;
    {
        std::string tag = std::string("map_tensors_") + mode_str + "_cat";
        nvtx3::scoped_range r{tag.c_str()};
        all_ids = torch::cat(unmapped_tensors);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> unique_tup;
    torch::Tensor map;
    torch::Tensor mapped_all_ids;

    int64_t current_debug_counter = -1;
    if (train_) {
        std::lock_guard<std::mutex> guard(log_mutex);
        current_debug_counter = debug_counter;
    }

    if (train_) {
        std::string slice_dir = "unique_kernel_data_slice/" + std::to_string(current_debug_counter);
        std::filesystem::create_directories(slice_dir);

        std::ofstream ofs(slice_dir + "/input.txt", std::ios::out);
        ofs << all_ids;
    }

    {
        std::string tag = std::string("map_tensors_") + mode_str + "_unique2";
        nvtx3::scoped_range r{tag.c_str()};
        unique_tup = torch::_unique2(all_ids, true, true, false);
    }

    {
        std::string tag = std::string("map_tensors_") + mode_str + "_unpack_unique";
        nvtx3::scoped_range r{tag.c_str()};
        map = std::get<0>(unique_tup);
        mapped_all_ids = std::get<1>(unique_tup);
    }

    if (train_) {
        std::string slice_dir = "unique_kernel_data_slice/" + std::to_string(current_debug_counter);
        std::filesystem::create_directories(slice_dir);

        {
            std::ofstream ofs(slice_dir + "/output_1.txt", std::ios::out);
            ofs << map;
        }
        {
            std::ofstream ofs(slice_dir + "/output_2.txt", std::ios::out);
            ofs << mapped_all_ids;
        }
    }

    if (train_) {
        std::string tag = std::string("map_tensors_") + mode_str + "_stats";
        nvtx3::scoped_range r{tag.c_str()};
        std::lock_guard<std::mutex> guard(log_mutex);

        int64_t num_all_ids = all_ids.numel();
        int64_t num_unique_ids = map.numel();

        double unique_ratio = (num_all_ids > 0)
            ? static_cast<double>(num_unique_ids) / static_cast<double>(num_all_ids)
            : 0.0;
        double dup_ratio = 1.0 - unique_ratio;

        int64_t edge_src_num = unmapped_tensors.size() > 0 ? unmapped_tensors[0].numel() : 0;
        int64_t edge_dst_num = unmapped_tensors.size() > 1 ? unmapped_tensors[1].numel() : 0;
        int64_t src_neg_num  = unmapped_tensors.size() > 2 ? unmapped_tensors[2].numel() : 0;
        int64_t dst_neg_num  = unmapped_tensors.size() > 3 ? unmapped_tensors[3].numel() : 0;

        std::ofstream ofs("profiles/getBatch_profiling/single_GPU/unique_rate.txt", std::ios::app);
        ofs << "[map_tensors][" << debug_counter << "] "
            << "mode=" << mode_str
            << ", all_ids=" << num_all_ids
            << ", unique_ids=" << num_unique_ids
            << ", unique_ratio=" << unique_ratio
            << ", dup_ratio=" << dup_ratio
            << ", edge_src=" << edge_src_num
            << ", edge_dst=" << edge_dst_num
            << ", src_neg=" << src_neg_num
            << ", dst_neg=" << dst_neg_num
            << "\n";

        debug_counter++;
    }

    std::vector<torch::Tensor> mapped_tensors;
    {
        std::string tag = std::string("map_tensors_") + mode_str + "_split_inverse";
        nvtx3::scoped_range r{tag.c_str()};

        int64_t offset = 0;
        int64_t size;
        for (auto tensor : unmapped_tensors) {
            size = tensor.size(0);
            mapped_tensors.emplace_back(mapped_all_ids.narrow(0, offset, size));
            offset += size;
        }
    }

    return std::forward_as_tuple(map, mapped_tensors);
}

// TODO this function uses a searchsorted to find the approriate value in the map tensor
// this can be made faster on the cpu by using an std::map to perform lookups

/*
    @zizhong
    This apply_tensor_map is used in gnn.
*/
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
