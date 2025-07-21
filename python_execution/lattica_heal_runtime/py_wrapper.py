from abc import ABC

import torch

# Add the build directory to sys.path
import sys
import os
build_dir = os.path.join(os.path.dirname(__file__), "../../build/example_impl")
sys.path.insert(0, build_dir)

import lattica_hw as lhw
from lattica_hw import DeviceTensor8, DeviceTensor32, DeviceTensor64

_host_to_device = {
    torch.int8: lhw.host_to_device_8,
    torch.int32: lhw.host_to_device_32,
    torch.int64: lhw.host_to_device_64,
}

_device_to_host = {
    DeviceTensor8: lhw.device_to_host_8,
    DeviceTensor32: lhw.device_to_host_32,
    DeviceTensor64: lhw.device_to_host_64,
}

_zeros = {
    torch.int32: lhw.zeros_32,
    torch.int64: lhw.zeros_64,
}

_empty = {
    torch.int8: lhw.empty_8,
    torch.int32: lhw.empty_32,
    torch.int64: lhw.empty_64,
}

_expand_impls = {
    DeviceTensor8: lhw.expand_8,
    DeviceTensor32: lhw.expand_32,
    DeviceTensor64: lhw.expand_64,
}

_squeeze_impls = {
    DeviceTensor32: lhw.squeeze_32,
    DeviceTensor64: lhw.squeeze_64,
}

_unsqueeze_impls = {
    DeviceTensor32: lhw.unsqueeze_32,
    DeviceTensor64: lhw.unsqueeze_64,
}

_get_slice_impls = {
    DeviceTensor8: lhw.get_slice_8,
    DeviceTensor32: lhw.get_slice_32,
    DeviceTensor64: lhw.get_slice_64,
}

_set_const_val_impls = {
    DeviceTensor32: lhw.set_const_val_32,
    DeviceTensor64: lhw.set_const_val_64,
}

_flatten_impls = {
    DeviceTensor32: lhw.flatten_32,
    DeviceTensor64: lhw.flatten_64,
}

_contiguous_impls = {
    DeviceTensor8: lhw.contiguous_8,
    DeviceTensor32: lhw.contiguous_32,
    DeviceTensor64: lhw.contiguous_64,
}

_modmul = {
    'ttt': {
        DeviceTensor32: lhw.modmul_ttt_32,
        DeviceTensor64: lhw.modmul_ttt_64,
    },
    'tcc': {
        DeviceTensor32: lhw.modmul_tcc_32,
        DeviceTensor64: lhw.modmul_tcc_64,
    },
    'tct': {
        DeviceTensor32: lhw.modmul_tct_32,
        DeviceTensor64: lhw.modmul_tct_64,
    },
    'ttc': {
        DeviceTensor32: lhw.modmul_ttc_32,
        DeviceTensor64: lhw.modmul_ttc_64,
    }
}

_modsum = {
    'ttt': {
        DeviceTensor32: lhw.modsum_ttt_32,
        DeviceTensor64: lhw.modsum_ttt_64,
    },
    'tcc': {
        DeviceTensor32: lhw.modsum_tcc_32,
        DeviceTensor64: lhw.modsum_tcc_64,
    },
    'tct': {
        DeviceTensor32: lhw.modsum_tct_32,
        DeviceTensor64: lhw.modsum_tct_64,
    },
    'ttc': {
        DeviceTensor32: lhw.modsum_ttc_32,
        DeviceTensor64: lhw.modsum_ttc_64,
    }
}

_mod = {
    'tt': {
        DeviceTensor32: lhw.mod_tt_32,
        DeviceTensor64: lhw.mod_tt_64,
    },
    'tc': {
        DeviceTensor32: lhw.mod_tc_32,
        DeviceTensor64: lhw.mod_tc_64,
    },
    'ct': {
        DeviceTensor32: lhw.mod_ct_32,
        DeviceTensor64: lhw.mod_ct_64,
    },
}

_modneg = {
    'tt': {
        DeviceTensor32: lhw.modneg_tt_32,
        DeviceTensor64: lhw.modneg_tt_64,
    },
    'tc': {
        DeviceTensor32: lhw.modneg_tc_32,
        DeviceTensor64: lhw.modneg_tc_64,
    }
}

_modmul_axis_sum = {
    DeviceTensor32: lhw.modmul_axis_sum_32,
    DeviceTensor64: lhw.modmul_axis_sum_64,
}

_axis_modsum = {
    DeviceTensor32: lhw.axis_modsum_32,
    DeviceTensor64: lhw.axis_modsum_64,
}

_apply_g_decomp = {
    (DeviceTensor32, DeviceTensor8): lhw.apply_g_decomp_32_8,
    (DeviceTensor64, DeviceTensor8): lhw.apply_g_decomp_64_8,
    (DeviceTensor32, DeviceTensor32): lhw.apply_g_decomp_32_32,
    (DeviceTensor64, DeviceTensor64): lhw.apply_g_decomp_64_64,
}

_ntt = {
    (DeviceTensor8, DeviceTensor32): lhw.ntt_8_32,
    (DeviceTensor8, DeviceTensor64): lhw.ntt_8_64,
    (DeviceTensor32, DeviceTensor32): lhw.ntt_32_32,
    (DeviceTensor64, DeviceTensor64): lhw.ntt_64_64,
}

_intt = {
    DeviceTensor32: lhw.intt_32,
    DeviceTensor64: lhw.intt_64,
}

_pad_single_axis_impls = {
    DeviceTensor32: lhw.pad_single_axis_32,
    DeviceTensor64: lhw.pad_single_axis_64,
}

_take_along_axis = {
    DeviceTensor32: lhw.take_along_axis_32,
    DeviceTensor64: lhw.take_along_axis_64,
}

_moveaxis = {
    DeviceTensor8: lhw.moveaxis_8,
    DeviceTensor32: lhw.moveaxis_32,
    DeviceTensor64: lhw.moveaxis_64,
}

_reshape = {
    DeviceTensor8: lhw.reshape_8,
    DeviceTensor32: lhw.reshape_32,
    DeviceTensor64: lhw.reshape_64,
}

def _dispatch(key, *args, impls):
    try:
        return impls[key](*args)
    except KeyError:
        raise TypeError(f"Unsupported tensor type: {key}")



class PythonToCppDispatcher(ABC):

    def host_to_device(self, tensor, dtype):
        return _dispatch(dtype, torch.tensor(tensor, dtype=dtype), impls=_host_to_device)

    def device_to_host(self, a):
        return _dispatch(type(a), a, impls=_device_to_host)

    def zeros(self, shape, dtype):
        return _dispatch(dtype, shape, impls=_zeros)

    def empty(self, shape, dtype):
        return _dispatch(dtype, shape, impls=_empty)

    def modmul_axis_sum(self, a, b, p, log2q_list, mu_list, perm, out, apply_perm, axis):
        _dispatch(type(a), a, b, p, perm, log2q_list, mu_list, axis, apply_perm, out, impls=_modmul_axis_sum)
        return out

    def axis_modsum(self, a, axis, q_list, out):
        _dispatch(type(a), a, q_list, out, axis, impls=_axis_modsum)
        return out

    def apply_g_decomp(self, a, g_exp, g_base_bits, out):
        _dispatch((type(a), type(out)), a, out, g_exp, g_base_bits, impls=_apply_g_decomp)
        return out

    def reshape(self, device_tensor, new_shape):
        return _dispatch(type(device_tensor), device_tensor, new_shape, impls=_reshape)

    def _modmul_ttt(self, a, b, p, out):
        _dispatch(type(a), a, b, p, out, impls=_modmul['ttt'])
        return out

    def _modmul_tcc(self, a, b, p, out):
        _dispatch(type(a), a, b, p, out, impls=_modmul['tcc'])
        return out

    def _modmul_tct(self, a, b, p, out):
        _dispatch(type(a), a, b, p, out, impls=_modmul['tct'])
        return out

    def _modmul_ttc(self, a, b, p, out):
        _dispatch(type(a), a, b, p, out, impls=_modmul['ttc'])
        return out

    def _modsum_ttt(self, a, b, p, out):
        _dispatch(type(a), a, b, p, out, impls=_modsum['ttt'])
        return out

    def _modsum_tcc(self, a, b, p, out):
        _dispatch(type(a), a, b, p, out, impls=_modsum['tcc'])
        return out

    def _modsum_tct(self, a, b, p, out):
        _dispatch(type(a), a, b, p, out, impls=_modsum['tct'])
        return out

    def _modsum_ttc(self, a, b, p, out):
        _dispatch(type(a), a, b, p, out, impls=_modsum['ttc'])
        return out

    def _mod_tt(self, a, b, out):
        _dispatch(type(a), a, b, out, impls=_mod['tt'])
        return out

    def _mod_tc(self, a, b, out):
        _dispatch(type(a), a, b, out, impls=_mod['tc'])
        return out

    def _mod_ct(self, a, b, out):
        _dispatch(type(a), a, b, out, impls=_mod['ct'])
        return out

    def _modneg_tt(self, a, p, out):
        _dispatch(type(a), a, p, out, impls=_modneg['tt'])
        return out

    def _modneg_tc(self, a, p, out):
        _dispatch(type(a), a, p, out, impls=_modneg['tc'])
        return out

    def expand(self, a, axis, repeat):
        return _dispatch(type(a), a, axis, repeat, impls=_expand_impls)

    def squeeze(self, a, axis):
        return _dispatch(type(a), a, axis, impls=_squeeze_impls)

    def unsqueeze(self, a, axis):
        return _dispatch(type(a), a, axis, impls=_unsqueeze_impls)

    def moveaxis(self, a, axis_src, axis_dst):
        return _dispatch(type(a), a, axis_src, axis_dst, impls=_moveaxis)

    def get_slice(self, a, sliceList):
        return _dispatch(type(a), a, sliceList, impls=_get_slice_impls)

    def set_const_val(self, a, value):
        return _dispatch(type(a), a, value, impls=_set_const_val_impls)

    def flatten(self, a, start_dim, end_dim):
        return _dispatch(type(a), a, start_dim, end_dim, impls=_flatten_impls)

    def contiguous(self, a):
        return _dispatch(type(a), a, impls=_contiguous_impls)

    def pad_single_axis(self, a, pad, axis, out):
        _dispatch(type(a), a, pad, axis, out, impls=_pad_single_axis_impls)
        return out

    def take_along_axis(self, a, b, axis, out):
        _dispatch(type(a), a, b, axis, out, impls=_take_along_axis)
        return out

    def ntt(self, a, axis, perm, perm_pairs, q_list, log2q, mu_list, psi_arr, out, tile, skip_perm):
        if tile:
            if axis == -1:
                a = self.expand(a, -2, 2)
            else:
                a = self.expand(a, -1, 2)
        _dispatch((type(a), type(out)), a, q_list, perm, psi_arr, log2q, mu_list, axis, skip_perm, out, impls=_ntt)
        return out

    def intt(self, a, perm, perm_pairs, q_list, log2q, mu_list, psi_arr, n_inv_list, out, tile):
        if tile:
            a = self.expand(a, -1, 2)
        _dispatch(type(a), a, q_list, perm, psi_arr, n_inv_list, log2q, mu_list, out, impls=_intt)
        return out
