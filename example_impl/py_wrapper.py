from abc import ABC

import torch

# Add the build directory to sys.path
import sys
import os
build_dir = os.path.join(os.path.dirname(__file__), "../build/example_impl")
sys.path.insert(0, build_dir)

import lattica_hw as lhw
from lattica_hw import DeviceTensor32, DeviceTensor64, DeviceTensorfloat64

_host_to_device = {
    torch.int32: lhw.host_to_device_32,
    torch.int64: lhw.host_to_device_64,
    torch.float64: lhw.host_to_device_float64
}

_device_to_host = {
    DeviceTensor32: lhw.device_to_host_32,
    DeviceTensor64: lhw.device_to_host_64,
    DeviceTensorfloat64: lhw.device_to_host_float64
}

_allocate = {
    torch.int32: lhw.allocate_on_hardware_32,
    torch.int64: lhw.allocate_on_hardware_64,
}

_expand_impls = {
    DeviceTensor32: lhw.expand_32,
    DeviceTensor64: lhw.expand_64,
    DeviceTensorfloat64: lhw.expand_float64
}

_squeeze_impls = {
    DeviceTensor32: lhw.squeeze_32,
    DeviceTensor64: lhw.squeeze_64,
    DeviceTensorfloat64: lhw.squeeze_float64
}

_unsqueeze_impls = {
    DeviceTensor32: lhw.unsqueeze_32,
    DeviceTensor64: lhw.unsqueeze_64,
    DeviceTensorfloat64: lhw.unsqueeze_float64
}

_contiguous_impls = {
    DeviceTensor32: lhw.make_contiguous_32,
    DeviceTensor64: lhw.make_contiguous_64,
    DeviceTensorfloat64: lhw.make_contiguous_float64
}

# modmul / modsum
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

_axis_modsum = {
    DeviceTensor32: lhw.axis_modsum_32,
    DeviceTensor64: lhw.axis_modsum_64,
}

_ntt = {
    DeviceTensor32: lhw.ntt_32,
    DeviceTensor64: lhw.ntt_64,
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

    def empty(self, shape, dtype):
        return _dispatch(dtype, shape, impls=_allocate)

    def axis_modsum(self, a, axis, q_list, out):
        _dispatch(type(a), a, q_list, out, axis, impls=_axis_modsum)
        return out

    def reshape(self, device_tensor, new_shape):
        device_tensor.reshape(new_shape)
        return device_tensor

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

    def expand(self, a, repeat, axis):
        return _dispatch(type(a), a, axis, repeat, impls=_expand_impls)

    def squeeze(self, a, axis):
        return _dispatch(type(a), a, axis, impls=_squeeze_impls)

    def unsqueeze(self, a, axis):
        return _dispatch(type(a), a, axis, impls=_unsqueeze_impls)

    def contiguous(self, a):
        return _dispatch(type(a), a, impls=_contiguous_impls)

    def ntt(self, a, perm, perm_pairs, q_list, psi_arr, out, tile, skip_perm):
        if skip_perm:
            raise NotImplementedError(f"skip_perm is not supported. {skip_perm=}")
        if tile:
            a = self.expand(a, 2, -1)
        _dispatch(type(a), a, q_list, perm, psi_arr, out, impls=_ntt)
        return out