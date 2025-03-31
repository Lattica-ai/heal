import json

import numpy as np
import torch

from datatypes import (
    DeviceTensorPointer, HostTensor,
    DeviceOpArg, DeviceOp, FreeDeviceTensor,
    ExecutionTranscript, ExecutionTranscriptOpType, DeviceOpArgType
)

import io
import base64

def encode_tensor(tensor):
    buffer = io.BytesIO()
    if isinstance(tensor, torch.Tensor):
        torch.save(tensor, buffer)
        type_tag = "torch"
    elif isinstance(tensor, np.ndarray):
        np.save(buffer, tensor)
        type_tag = "numpy"
    else:
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")
    return {
        "type": type_tag,
        "data": base64.b64encode(buffer.getvalue()).decode("utf-8")
    }

def decode_tensor(obj):
    buffer = io.BytesIO(base64.b64decode(obj["data"]))
    if obj["type"] == "torch":
        return torch.load(buffer, weights_only=True)
    elif obj["type"] == "numpy":
        return np.load(buffer)
    else:
        raise ValueError(f"Unknown tensor type tag: {obj['type']}")

# JSON Encoder for custom types
class HEALJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, DeviceOpArgType):
            return {"__type__": "DeviceOpArgType", "value": obj.name}
        elif isinstance(obj, ExecutionTranscriptOpType):
            return {"__type__": "ExecutionTranscriptOpType", "value": obj.name}

        elif isinstance(obj, DeviceTensorPointer):
            return {
                "__type__": "DeviceTensorPointer",
                "dtype": str(obj.dtype),
                "inf_name": obj.inf_name,
            }
        elif isinstance(obj, HostTensor):
            return {
                "__type__": "HostTensor",
                "tensor_base64": encode_tensor(obj.tensor),
                # "orig_python_path": obj.orig_python_path,
            }
        elif isinstance(obj, DeviceOpArg):
            return {
                "__type__": "DeviceOpArg",
                "arg_type": obj.arg_type,
                "value": obj.value,
            }
        elif isinstance(obj, DeviceOp):
            return {
                "__type__": "DeviceOp",
                "name": obj.name,
                "args": obj.args,
                "out": obj.out,
            }
        elif isinstance(obj, FreeDeviceTensor):
            return {"__type__": "FreeDeviceTensor", "tensor_name": obj.tensor_name}
        elif isinstance(obj, ExecutionTranscript):
            return {"__type__": "ExecutionTranscript", "transcript": obj.transcript}
        elif isinstance(obj, slice):
            return {"__type__": "slice", "start": obj.start, "stop": obj.stop, "step": obj.step}
        elif obj is Ellipsis:
            return {"__type__": "ellipsis"}
        if isinstance(obj, torch.dtype):
            return {"__type__": "dtype", "value": str(obj).split(".")[-1]}
        elif isinstance(obj, (complex, np.complexfloating)):
            return {
                "__type__": "complex",
                "real": obj.real,
                "imag": obj.imag
            }

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


def save_transcript_to_json(transcript, filename):
    with open(filename, "w") as f:
        json.dump(transcript, f, cls=HEALJSONEncoder, indent=2)


def load_transcript_from_json(filename):
    print(f'Loading transcript from {filename}')
    with open(filename, "r") as f:
        res = json.load(f, object_hook=heal_json_hook)
    print(f'Transcript loaded {res=}')
    return res
