#include <vector>
#include <cstdint>

namespace device_tensor_utils {

    // compute total number of elements
    inline size_t numel(const std::vector<int64_t>& dims) {
        size_t n = 1;
        for (auto d : dims) n *= static_cast<size_t>(d);
        return n;
    }

    // turn a flat index into a multi-dim index (row-major)
    inline std::vector<int64_t> unravel_index(
        size_t linear,
        const std::vector<int64_t>& dims
    ) {
        size_t rank = dims.size();
        std::vector<int64_t> idx(rank);
        for (int64_t i = rank - 1; i >= 0; --i) {
            idx[i] = static_cast<int64_t>(linear % dims[i]);
            linear /= dims[i];
        }
        return idx;
    }

}  // namespace device_tensor_utils
