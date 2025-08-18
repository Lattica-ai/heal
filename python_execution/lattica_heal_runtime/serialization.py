import json
import io
import base64
import numpy as np
import torch

from lattica_heal_runtime.datatypes import (
    DeviceTensorPointer, HostTensor,
    DeviceOpArg, DeviceOp, FreeDeviceTensor,
    ExecutionTranscript, ExecutionTranscriptOpType, DeviceOpArgType
)


def decode_tensor(obj):
    buffer = io.BytesIO(base64.b64decode(obj["data"]))
    if obj["type"] == "torch":
        return torch.load(buffer, weights_only=True)
    elif obj["type"] == "numpy":
        return np.load(buffer)
    else:
        raise ValueError(f"Unknown tensor type tag: {obj['type']}")


# JSON Decoder for custom types
def heal_json_hook(dct):
    if "__type__" not in dct:
        return dct

    t = dct["__type__"]

    if t == "DeviceOpArgType":
        return DeviceOpArgType[dct["value"]]
    if t == "ExecutionTranscriptOpType":
        if dct["value"] in ExecutionTranscriptOpType.__members__:
            return ExecutionTranscriptOpType[dct["value"]]

    if t == "DeviceTensorPointer":
        obj = DeviceTensorPointer.__new__(DeviceTensorPointer)
        obj.dtype = getattr(torch, dct["dtype"].split(".")[-1])
        obj.inf_name = dct["inf_name"]
        return obj
    if t == "HostTensor":
        obj = HostTensor.__new__(HostTensor)
        obj.tensor = decode_tensor(dct["tensor_base64"])
        return obj
    if t == "DeviceOpArg":
        arg = DeviceOpArg.__new__(DeviceOpArg)
        arg.arg_type = dct["arg_type"]
        arg.value = dct["value"]
        return arg
    if t == "DeviceOp":
        op = DeviceOp.__new__(DeviceOp)
        op.name = dct["name"]
        op.args = dct["args"]
        op.out = dct["out"]
        return op
    if t == "FreeDeviceTensor":
        return FreeDeviceTensor(dct["tensor_name"])
    if t == "slice":
        return slice(dct["start"], dct["stop"], dct["step"])
    if t == "ellipsis":
        return Ellipsis
    if t == "ExecutionTranscript":
        et = ExecutionTranscript()
        et.transcript = [tuple(item) for item in dct["transcript"]]
        return et
    if t == "dtype":
        return getattr(torch, dct["value"])
    if t == "complex":
        return complex(dct["real"], dct["imag"])


def load_transcript(filename):
    print(f'Loading transcript from {filename}')
    with open(filename, "r") as f:
        res = json.load(f, object_hook=heal_json_hook)
    print(f'Transcript loaded {res=}')
    return res
