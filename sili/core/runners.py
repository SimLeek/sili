import kp

from typing import List, Callable

import numpy as np

from sili.core.module import Module


def get_forward_sequence(
        modules: List[Module]
):
    # we're assuming all modules are on the same device
    sequence = modules[0].gpu.manager.sequence()

    # Record the operations into the sequences in the most unoptimized way possible
    for module in modules:
        sequence.record(kp.OpTensorSyncDevice([*module.forward_input_buffers]))
        [sequence.record(fo) for fo in module.forward_ops()]
        sequence.record(kp.OpTensorSyncLocal([*module.forward_output_buffers]))

    return sequence

def get_fast_forward_sequence(
        modules: List[Module]
):
    # we're assuming all modules are on the same device
    initialization_sequence = modules[0].gpu.manager.sequence()

    # Record the operations into the sequences in the most unoptimized way possible
    for module in modules:
        initialization_sequence.record(kp.OpTensorSyncDevice([*module.forward_input_buffers]))
        [initialization_sequence.record(fo) for fo in module.forward_ops()]
        initialization_sequence.record(kp.OpTensorSyncLocal([*module.forward_output_buffers]))
    initialization_sequence.eval()

    sequence = modules[0].gpu.manager.sequence()

    # Record the operations into the sequences in the most unoptimized way possible
    sequence.record(kp.OpTensorSyncDevice([*modules[0].forward_input_buffers]))
    for module in modules:
        [sequence.record(fo) for fo in module.forward_ops()]
    sequence.record(kp.OpTensorSyncLocal([*modules[-1].forward_output_buffers]))

    return sequence

def get_full_sequence(
        modules: List[Module],
        input_buffers,
        output_buffers
):
    # still assuming all modules are on the same device
    initialization_sequence = modules[0].gpu.manager.sequence()
    for m in modules:
        if m.has_forward and m.forward_input_buffers:
            initialization_sequence.record(kp.OpTensorSyncDevice([*m.forward_input_buffers]))
    for m in reversed(modules):
        if m.has_backward and m.backward_input_buffers:
            initialization_sequence.record(kp.OpTensorSyncDevice([*m.backward_input_buffers]))
    for m in modules:
        if m.has_optim and m.optim_input_buffers:
            initialization_sequence.record(kp.OpTensorSyncDevice([*m.optim_input_buffers]))
    initialization_sequence.eval()

    full_sequence = modules[0].gpu.manager.sequence()
    full_sequence.record(kp.OpTensorSyncDevice([*input_buffers]))
    for m in modules:
        for f in m.forward_ops():
            full_sequence.record(f)
    for m in reversed(modules):
        for b in m.backward_ops():
            full_sequence.record(b)
    for m in modules:
        for opt in m.optim_ops():
            full_sequence.record(opt)
    full_sequence.record(kp.OpTensorSyncLocal([*output_buffers]))

    return full_sequence
