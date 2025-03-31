from datatypes import DeviceOpArgType, ExecutionTranscriptOpType


def _print_arg(arg, indent):
    arg_type = arg.arg_type
    arg = arg.value
    prop_indent = indent + "\t"
    match arg_type:
        case DeviceOpArgType.DEVICE_TENSOR:
            return (f"DeviceTensor: "
                    f"\n{prop_indent}inf_name[{arg.inf_name}]"
                    f"\n{prop_indent}dtype[{arg.dtype}]")
        case DeviceOpArgType.HOST_TENSOR:
            return (f"HostTensor: "
                    f"\n{prop_indent}shape{list(arg.tensor.shape)}"
                    f"\n{prop_indent}dtype[{arg.tensor.dtype}] ")
        case DeviceOpArgType.SHAPE:
            return 'Shape: {' + f"{', '.join([_print_arg(a, "") for a in arg])}" + '}'
        case DeviceOpArgType.INT:
            return str(arg)
        case DeviceOpArgType.NONE:
            return 'None'
        case DeviceOpArgType.TENSOR_TYPE:
            return str(arg)
        case DeviceOpArgType.SLICE:
            return f"slice({arg.start}:{arg.stop}:{arg.step})"
        case DeviceOpArgType.ELLIPSIS:
            return '...'
    raise ValueError(f"Unknown arg type: {arg}")


def _print_op(op, indent):
    arg_indent = indent + "\t"
    args_str = f'\n{arg_indent}'.join([f'({i}) {_print_arg(arg, arg_indent)}' for i, arg in enumerate(op.args)])
    return (
            f"instruction name: {op.name}\n"
            f"{indent}args:\n{arg_indent}{args_str}\n"
            f"{indent}out:\n{arg_indent}{_print_arg(op.out, arg_indent)}")


def print_transcript(transcript, indent="\t"):
        print(f"=====================================================================")
        print(f"{indent}Number of operations: {len(transcript.transcript)}")
        print(f"=====================================================================")
        op_indent = indent + "\t"
        for i, op in enumerate(transcript.transcript):
            match op[0]:
                case ExecutionTranscriptOpType.DEVICE_OP:
                    print(f'{i=} {_print_op(op[1], op_indent)}')
                case ExecutionTranscriptOpType.FREE_DEVICE_TENSOR:
                    print(f"{i=} Free Device Tensor: inf_name[{op[1].tensor_name}]")
                case ExecutionTranscriptOpType.SEGMENT_START:
                    print(f"{i=} Segment Start: {op[1]}")
                case ExecutionTranscriptOpType.SEGMENT_END:
                    print(f"{i=} Segment End")
            print(f"=====================================================================")
        print(f"Number of operations: {len(transcript.transcript)}")
