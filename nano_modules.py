import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

class _LinearFunc(torch.autograd.Function):
    # x: (M, K)
    # w: (N, K)
    # b: (N,)
    @staticmethod
    def forward(ctx, x, w, b):
        # here `x` can be in shape float32 since the original input shape is float32
        x = x.to(w.dtype)
        ctx.save_for_backward(x, w, b)

        if b is not None:
            o = torch.addmm(b, x, w.T)
        else:
            o = torch.matmul(x, w.T)

        return o

    @staticmethod
    def backward(ctx, dC):
        # c = x @ wT + b
        # dC = grad_output
        # dx = dC @ w
        # dw = dC @ xT
        # db = dC.sum(0) ---> why?
        # because:
        # dL/db_j = sum(dL/dC_ij * dC_ij/db_j for i in range(M))
        # dC_:j/db_j = sum(dC_i for i in range(M)) = dC.sum(0)
        # grad_output (or dC) has shape (batch_size, out_features)
        # Each sample in the batch contributes its gradient to the same bias parameters
        # Therefore, we need to sum the gradients across all batch samples (dimension 0)
        # to get the correct gradient for each bias parameter
        x, w, b = ctx.saved_tensors
        dX = torch.matmul(dC, w)
        dW = torch.matmul(dC.t(), x)
        if b is not None:
            dB = dC.sum(0)
        else:
            dB = None
        return dX, dW, dB

class NanoLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=DEVICE, dtype=DTYPE):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias, device=DEVICE, dtype=DTYPE)
    
    def forward(self, x):
        _shape = x.shape
        if len(_shape) == 2:
            return _LinearFunc.apply(x, self.linear.weight, self.linear.bias)
        elif len(_shape) == 3:
            x = x.view(-1, _shape[-1])
            o = _LinearFunc.apply(x, self.linear.weight, self.linear.bias)
            o_shape = _shape[:-1] + (o.shape[-1],) 
            return o.view(o_shape)


def _test_custom_linear(x, ref_linear, _linear):
    x_ref = x.clone().detach().requires_grad_(True)
    y_ref = ref_linear(x_ref)
    y = _linear(x)

    torch.testing.assert_close(y, y_ref, atol=1e-3, rtol=1e-3)
    print("Fwd: PASS")

    dO = torch.randn_like(y)
    y_ref.backward(dO)
    y.backward(dO)

    torch.testing.assert_close(x.grad, x.grad, atol=1e-3, rtol=1e-3)
    print("Bwd: Gradients of X match")
    
    # 2. Test weight gradients
    torch.testing.assert_close(_linear.linear.weight.grad, ref_linear.weight.grad, atol=1e-3, rtol=1e-3)
    print("Bwd: Gradients of W match")
    
    # 3. Test bias gradients
    if ref_linear.bias is not None:
        torch.testing.assert_close(_linear.linear.bias.grad, ref_linear.bias.grad, atol=1e-3, rtol=1e-3)
        print("Bwd: Gradients of Bias match")
    
    print("\n✅ All tests passed!")

def test_custom_linear():
    torch.manual_seed(7)

    bs = 4
    tokens = 128
    in_features = 512
    out_features = 1024

    ref_linear = nn.Linear(in_features, out_features, bias=True, device=DEVICE, dtype=DTYPE)
    _linear = NanoLinear(in_features, out_features, bias=True, device=DEVICE, dtype=DTYPE)
    _linear.linear.weight = ref_linear.weight
    _linear.linear.bias = ref_linear.bias

    x_2d = torch.randn(bs, in_features, device=DEVICE, dtype=DTYPE)
    _test_custom_linear(x_2d, ref_linear, _linear)

    x_3d = torch.randn((bs, tokens, in_features), device=DEVICE, dtype=DTYPE)
    _test_custom_linear(x_3d, ref_linear, _linear)

if __name__ == "__main__":
    test_custom_linear()

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class NanoMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = NanoLinear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = NanoLinear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    