import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class MoGLayer(nn.Module):

    def __init__(self,
                 input_shape,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                ):
        
        super(MoGLayer, self).__init__()

        # if 'input_shape' not in kwargs and 'input_dim' in kwargs:
        #     kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.input_shape = input_shape
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        input_dim = self.input_shape
        self.std = nn.Parameter(torch.FloatTensor(input_dim).uniform_(self.kernel_initializer[0], self.kernel_initializer[1]),requires_grad=True).to(self.device)

        self.mean = nn.Parameter(torch.FloatTensor(input_dim).uniform_(self.bias_initializer[0], self.bias_initializer[1]),requires_grad=True).to(self.device)

        self.built = False
        self.device = "cpu"
        # self.kernel_regularizer = kernel_regularizer

    def build(self):
        assert self.input_shape >= 2
        input_dim = self.input_shape

        self.std = nn.Parameter(torch.FloatTensor(input_dim).uniform_(self.kernel_initializer[0], self.kernel_initializer[1]),requires_grad=True).to(self.device)

        self.mean = nn.Parameter(torch.FloatTensor(input_dim).uniform_(self.bias_initializer[0], self.bias_initializer[1]),requires_grad=True).to(self.device)

        self.built = True

    def forward(self, inputs):
        if not self.built: self.build()
        output = inputs * self.std
        output = output + self.mean
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = input_shape[-1]
        return tuple(output_shape)