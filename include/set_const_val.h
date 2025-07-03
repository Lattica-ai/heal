#ifndef SET_CONST_VAL_H
#define SET_CONST_VAL_H

namespace lattica_hw_api {

/**
 * @brief Sets all elements of the tensor to a constant value.
 *
 * This function assigns the given constant value `val` to every element
 * in the device tensor `a`.
 *
 * @tparam T  Data type of the tensor elements.
 * @param a   Shared pointer to the device tensor to modify.
 * @param val The constant value to set for all elements.
 */
template <typename T>
std::shared_ptr<DeviceTensor<T>> set_const_val(
    const std::shared_ptr<DeviceTensor<T>>& a,
    T val
);

}; // namespace lattica_hw_api

#endif // SET_CONST_VAL_H