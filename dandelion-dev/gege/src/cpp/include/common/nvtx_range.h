#pragma once

#if __has_include(<nvtx3/nvtx3.hpp>)
#include <nvtx3/nvtx3.hpp>
#else
namespace nvtx3 {
class scoped_range {
  public:
    explicit scoped_range(const char *) {}
};
} // namespace nvtx3
#endif
