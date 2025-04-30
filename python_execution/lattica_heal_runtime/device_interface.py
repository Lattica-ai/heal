from abc import ABC

class DeviceDispatcher(ABC):

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher

        # ================== Memory Allocation Ops ==========

    def empty(self, *args, **kwargs):
        return self.dispatcher.empty(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        return self.dispatcher.zeros(*args, **kwargs)

        # ================== I/O Ops ========================

    def device_to_host(self, *args, orig_python_path=None, **kwargs):
        return self.dispatcher.device_to_host(*args, **kwargs)

    def host_to_device(self, *args, orig_python_path=None, **kwargs):
        return self.dispatcher.host_to_device(*args, **kwargs)

        # ================== Mod ops ==================

    def _modmul_ttt(self, *args, **kwargs):
        return self.dispatcher._modmul_ttt(*args, **kwargs)
    def _modmul_ttc(self, *args, **kwargs):
        return self.dispatcher._modmul_ttc(*args, **kwargs)
    def _modmul_tct(self, *args, **kwargs):
        return self.dispatcher._modmul_tct(*args, **kwargs)
    def _modmul_tcc(self, *args, **kwargs):
        return self.dispatcher._modmul_tcc(*args, **kwargs)

    def _modsum_ttt(self, *args, **kwargs):
        return self.dispatcher._modsum_ttt(*args, **kwargs)
    def _modsum_ttc(self, *args, **kwargs):
        return self.dispatcher._modsum_ttc(*args, **kwargs)
    def _modsum_tct(self, *args, **kwargs):
        return self.dispatcher._modsum_tct(*args, **kwargs)
    def _modsum_tcc(self, *args, **kwargs):
        return self.dispatcher._modsum_tcc(*args, **kwargs)

    def modneg_tt(self, *args, **kwargs):
        return self.dispatcher._modneg_tt(*args, **kwargs)

    def modneg_tc(self, *args, **kwargs):
        return self.dispatcher._modneg_tc(*args, **kwargs)

    def axis_modsum(self, *args, **kwargs):
        return self.dispatcher.axis_modsum(*args, **kwargs)

    def ntt(self, *args, **kwargs):
        return self.dispatcher.ntt(*args, **kwargs)

    def intt(self, *args, **kwargs):
        return self.dispatcher.intt(*args, **kwargs)

        # ================== Other Compute Ops ==================

    def modmul_axis_sum(self, *args, **kwargs):
        return self.dispatcher.modmul_axis_sum(*args, **kwargs)

    def take_along_axis(self, *args, **kwargs):
        return self.dispatcher.take_along_axis(*args, **kwargs)

    def apply_g_decomp(self, *args, **kwargs):
        return self.dispatcher.apply_g_decomp(*args, **kwargs)

    def abs(self, *args, **kwargs):
        return self.dispatcher.abs(*args, **kwargs)

    def set_const_val(self, *args, **kwargs):
        return self.dispatcher.set_const_val(*args, **kwargs)

        # ================== Real Reshape Ops ================

    def moveaxis(self, *args, **kwargs):
        return self.dispatcher.moveaxis(*args, **kwargs)

    def pad_single_axis(self, *args, **kwargs):
        return self.dispatcher.pad_single_axis(*args, **kwargs)

    def expand(self, *args, **kwargs):
        return self.dispatcher.expand(*args, **kwargs)

    def contiguous(self, *args, **kwargs):
        return self.dispatcher.contiguous(*args, **kwargs)

        # ================== Virtual Reshape Ops ============

    def get_slice(self, *args, **kwargs):
        return self.dispatcher.get_slice(*args, **kwargs)

    def new_reference(self, *args, **kwargs):
        return self.dispatcher.new_reference(*args, **kwargs)

    def flatten(self, a, *args, **kwargs):
        return self.dispatcher.flatten(a, *args, **kwargs)

    def unsqueeze(self, *args, **kwargs):
        return self.dispatcher.unsqueeze(*args, **kwargs)

    def squeeze(self, *args, **kwargs):
        return self.dispatcher.squeeze(*args, **kwargs)

    def reshape(self, a, *args, **kwargs):
        return self.dispatcher.reshape(a, *args, **kwargs)
