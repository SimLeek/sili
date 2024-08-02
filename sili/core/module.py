import abc
from typing import List

from sili.core.buffers import Buffer
from sili.core.devices.gpu import GPUManager


class Module(object):
    __slots__ = (
        'gpu',
        'optim_setup_buffers', 'forward_setup_buffers', 'backward_setup_buffers',
        'optim_input_buffers', 'forward_input_buffers', 'backward_input_buffers',
        'optim_output_buffers', 'forward_output_buffers', 'backward_output_buffers',
        'has_forward', 'has_backward', 'has_optim'
    )

    gpu: GPUManager
    optim_setup_buffers: List[Buffer]
    forward_setup_buffers: List[Buffer]
    backward_setup_buffers: List[Buffer]
    optim_input_buffers: List[Buffer]
    forward_input_buffers: List[Buffer]
    backward_input_buffers: List[Buffer]
    optim_output_buffers: List[Buffer]
    forward_output_buffers: List[Buffer]
    backward_output_buffers: List[Buffer]
    has_forward: bool
    has_backward: bool
    has_optim: bool
    def __init__(self):
        self.setup_check()

    def setup_check(self):
        assert hasattr(self, 'gpu') and isinstance(self.gpu, GPUManager)
        assert any([
            hasattr(self, 'has_forward') and self.has_forward,
            hasattr(self, 'has_backward') and self.has_backward,
            hasattr(self, 'has_optim') and self.has_optim
        ]), "Module must do _something_."
        assert any([
            hasattr(self, 'optim_input_buffers') and len(self.optim_input_buffers) != 0,
            hasattr(self, 'forward_input_buffers') and len(self.forward_input_buffers) != 0,
            hasattr(self, 'backward_input_buffers') and len(self.backward_input_buffers) != 0
        ]), "Module must take input."
        assert any([
            hasattr(self, 'optim_output_buffers') and len(self.optim_output_buffers) != 0,
            hasattr(self, 'forward_output_buffers') and len(self.forward_output_buffers) != 0,
            hasattr(self, 'backward_output_buffers') and len(self.backward_output_buffers) != 0
        ]), "Module must give output."

    @abc.abstractmethod
    def forward_ops(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def backward_ops(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def optim_ops(self):
        raise NotImplementedError()

def reverse_module(cls):
    class ReversedClass(cls):
        def forward_ops(self):
            return super().backward_ops()

        def backward_ops(self):
            return super().forward_ops()
    return ReversedClass