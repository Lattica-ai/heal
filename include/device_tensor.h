#pragma once

namespace lattica_hw_api {

/**
 * @brief Abstract device-side tensor for hardware-accelerated memory.
 */
template <typename T>
class DeviceTensor {
public:
    void print() const;
    void print_metadata() const;
};

}