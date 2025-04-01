from enum import Enum
from typing import Any

class DeviceTensorPointer:
    dtype: Any = None
    inf_name: str = None

class HostTensor:
    tensor = None

class DeviceOpArgType(Enum):
    HOST_TENSOR = "HostTensor"
    DEVICE_TENSOR = "DeviceTensor"
    SHAPE = "Shape"
    INT = "int"
    NONE = "None"
    TENSOR_TYPE = "tensor_type"
    SLICE = "slice"
    ELLIPSIS = "Ellipsis"

class DeviceOpArg:
    arg_type: DeviceOpArgType
    value: Any

class DeviceOp:
    name: str
    args: list[DeviceOpArg]
    out: DeviceOpArg

class FreeDeviceTensor:
    def __init__(self, tensor_name: str):
        self.tensor_name = tensor_name

class ExecutionTranscriptOpType(Enum):
    SEGMENT_START = "SegmentStart"
    SEGMENT_END = "SegmentEnd"
    DEVICE_OP = "DeviceOp"
    FREE_DEVICE_TENSOR = "FreeDeviceTensor"


class ExecutionTranscript:
    transcript: list[tuple[ExecutionTranscriptOpType, Any]]