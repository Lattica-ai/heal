import time

import torch

from lattica_heal_runtime.datatypes import (
    HostTensor, DeviceTensorPointer,
    DeviceOp, ExecutionTranscriptOpType, DeviceOpArgType
)

def _host_to_device(device_t_eng, memory_refs, op: DeviceOp):
    host_tensor = op.args[0].value.tensor
    device_tensor = device_t_eng.host_to_device(host_tensor, op.out.value.dtype)
    memory_refs[op.out.value.inf_name] = device_tensor

def _device_to_host(device_t_eng, memory_refs, op: DeviceOp, verify) -> HostTensor:
    device_tensor = memory_refs[op.args[0].value.inf_name]
    host_tensor = device_t_eng.device_to_host(device_tensor)
    expected_host_tensor = op.out.value.tensor
    if verify:
        assert (host_tensor == expected_host_tensor).all()
    return host_tensor

def _get_args_value(memory_refs, arg):
    match arg.arg_type:
        case DeviceOpArgType.DEVICE_TENSOR:
            return memory_refs[arg.value.inf_name]
        case DeviceOpArgType.HOST_TENSOR:
            raise ValueError("HostTensor not supported")
        case DeviceOpArgType.SHAPE:
            return tuple([_get_args_value(memory_refs, a) for a in arg.value])
        case DeviceOpArgType.INT:
            return arg.value
        case DeviceOpArgType.NONE:
            return None
        case DeviceOpArgType.TENSOR_TYPE:
            return arg.value
        case DeviceOpArgType.SLICE:
            return slice(arg.value.start, arg.value.stop, arg.value.step)
        case DeviceOpArgType.ELLIPSIS:
            return Ellipsis
    raise ValueError(f"Unknown arg type: {arg.arg_type}")

def _run_device_instruction(device_t_eng, memory_refs, op: DeviceOp, verify):
    if op.name == 'device_to_host':
        _device_to_host(device_t_eng, memory_refs, op, verify)
        return
    if op.name == 'host_to_device':
        _host_to_device(device_t_eng, memory_refs, op)
        return
    op_args = [_get_args_value(memory_refs, arg) for arg in op.args]
    if op.name == 'modmul':
        print(f"Running device instruction: {op.name}")
        print(f"op_args: {[(o.data_ptr(), o.shape) if isinstance(o, torch.Tensor) else o for o in op_args]}")
    op_t_eng_fun = getattr(device_t_eng, op.name)
    out = op_t_eng_fun(*op_args)
    memory_refs[op.out.value.inf_name] = out

def _run_op(device_t_eng, memory_refs, op, verify):
    match op[0]:
        case ExecutionTranscriptOpType.SEGMENT_START:
            print(f"Starting segment: {op[1]}")
            return
        case ExecutionTranscriptOpType.SEGMENT_END:
            print(f"End of segment")
            return
        case ExecutionTranscriptOpType.DEVICE_OP:
            # print(f"Running device instruction: {op[1].name}")
            _run_device_instruction(device_t_eng, memory_refs, op[1], verify)
            return
        case ExecutionTranscriptOpType.FREE_DEVICE_TENSOR:
            # print(f"Freeing tensor: {op[1].tensor_name}")
            del memory_refs[op[1].tensor_name]
            return
        case None:
            return
        case _:
            raise ValueError(f"Unknown op type: {op[0]}")

def run_transcript(device_t_eng, transcript, verify=False):
    print("\n\n######### Running transcript... #########")
    memory_refs: dict[str, DeviceTensorPointer] = {}

    # Synchronize and start timer
    torch.cuda.synchronize()
    start = time.time()

    for i, op in enumerate(transcript):
        # print(f"Running operation {i}")
        _run_op(device_t_eng, memory_refs, op, verify)

    # Synchronize and end timer
    torch.cuda.synchronize()
    end = time.time()

    print(f"Elapsed time: {end - start:.6f} seconds")
    if verify:
        print("######### Verification successful #########\n\n")