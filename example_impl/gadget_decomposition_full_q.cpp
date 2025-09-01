#include "gadget_decomposition_full_q.h"
#include "device_tensor_ex.h"
#include "../include/num.h"
#include <vector>

namespace lattica_hw_api {

// From type T to Num
template <typename T>
Num to_num(T value) {
    return Num(static_cast<int64_t>(value));
}

// Modular inverse using extended Euclidean algorithm
Num modular_inverse_num(const Num& a, const Num& m) {
    if (m == Num(1)) {
        return Num(0);
    }
    
    Num x0(0), x1(1);
    Num a_copy = a;
    Num m_copy = m;
    
    while (a_copy > Num(1)) {
        Num q = a_copy / m_copy;
        Num temp = m_copy;
        
        m_copy = a_copy % m_copy;
        a_copy = temp;
        
        temp = x0;
        x0 = x1 - q * x0;
        x1 = temp;
    }
    
    // Make x1 positive
    if (x1 < Num(0)) {
        x1 = x1 + m;
    }
    
    return x1;
}

// From RNS representation to big int using Chinese Remainder Theorem
template <typename T>
Num rns_to_integer(const std::vector<T>& residues, const std::vector<T>& moduli) {

    Num q(1);
    for (size_t i = 0; i < moduli.size(); i++) {
        q = q * to_num(moduli[i]);
    }
    
    Num result(0);
    for (size_t i = 0; i < residues.size(); i++) {
        Num M_i = q / to_num(moduli[i]);
        Num y_i = modular_inverse_num(M_i % to_num(moduli[i]), to_num(moduli[i]));
        
        result = result + to_num(residues[i]) * M_i * y_i;
    }
    
    return result % q;
}

// Gadget decomposition relative to full q
template <typename T, typename U>
void apply_g_decomp_relative_to_full_q(
    const std::shared_ptr<DeviceTensor<T>>& a,
    const std::shared_ptr<DeviceTensor<T>>& q_list,
    int g_exp,
    int g_base_bits,
    std::shared_ptr<DeviceTensor<U>>& out) {

    int reps_l = a->dims[0];
    int reps_r = a->dims[2]; 
    int q_list_len = q_list->dims[0];

    T g_base = static_cast<T>(1) << g_base_bits;

    for (int l = 0; l < reps_l; l++) {
        for (int r = 0; r < reps_r; r++) {

            // Extract RNS residues
            std::vector<T> residues(q_list_len);
            std::vector<T> moduli(q_list_len);
            for (int j = 0; j < q_list_len; j++) {
                residues[j] = a->at({l, j, r});
                moduli[j] = q_list->at({j});
            }

            // Convert RNS to big int using CRT
            Num x = rns_to_integer(residues, moduli);

            // Gadget decomposition
            for (int i = 0; i < g_exp; i++) {

                Num digit = x % to_num(g_base);
                x = x / to_num(g_base);

                int result;
                digit.can_convert_to_int(&result);
                out->at({l, i, r}) = static_cast<T>(result);

                // If x becomes 0, remaining digits are 0
                if (x == Num(0)) {
                    for (int k = i + 1; k < g_exp; k++) {
                        out->at({l, k, r}) = 0;
                    }
                    break;
                }
            }
        }
    }
}


#define INSTANTIATE_APPLY_G_DECOMP_RELATIVE_TO_FULL_Q(T1, T2) \
    template void apply_g_decomp_relative_to_full_q<T1, T2>( \
        const std::shared_ptr<DeviceTensor<T1>>& a, \
        const std::shared_ptr<DeviceTensor<T1>>& q_list, \
        int g_exp, \
        int g_base_bits, \
        std::shared_ptr<DeviceTensor<T2>>& out \
    );

INSTANTIATE_APPLY_G_DECOMP_RELATIVE_TO_FULL_Q(int32_t, int8_t)
INSTANTIATE_APPLY_G_DECOMP_RELATIVE_TO_FULL_Q(int32_t, int32_t)
INSTANTIATE_APPLY_G_DECOMP_RELATIVE_TO_FULL_Q(int64_t, int8_t)
INSTANTIATE_APPLY_G_DECOMP_RELATIVE_TO_FULL_Q(int64_t, int64_t)

} // namespace lattica_hw_api