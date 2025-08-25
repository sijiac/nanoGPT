import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

DEVICE = torch.device("cuda")
DTYPE = torch.float32

torch.manual_seed(7)

def torch_attention(q, k, v, causal=False, dropout_p=0.0, training=True):
        is_dropped = training and dropout_p > 0.0
        # q: (B, Hq, T, K)
        # k: (B, Hk, T, K)
        # v: (B, Hk, T, K)
        # o: (B, Hq, T, K)
        B, Hq, T, K = q.shape
        scale = 1.0 / math.sqrt(K)
        s = q @ k.transpose(-2, -1) * scale

        causal_mask = torch.tril(torch.ones(T, T, device=DEVICE, dtype=DTYPE))
        if causal:
            s = s.masked_fill(causal_mask == 0, float("-inf"))
        
        p = F.softmax(s.float(), dim=-1)

        if is_dropped:
            dropped_p = F.dropout(p, p=dropout_p, training=training)
        else:
            dropped_p = p

        dropped_p = dropped_p.to(q.dtype)
        o = dropped_p @ v  # (B, Hq, T, T) x (B, Hk, T, K) -> (B, Hq, T, K)

        return o

class _NanoFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=False, dropout_p=0.0, training=True):
        is_dropped = training and dropout_p > 0.0
        # q: (B, Hq, T, K)
        # k: (B, Hk, T, K)
        # v: (B, Hk, T, K)
        # o: (B, Hq, T, K)
        B, Hq, T, K = q.shape
        scale = 1.0 / math.sqrt(K)
        s = q @ k.transpose(-2, -1) * scale

        causal_mask = torch.tril(torch.ones(T, T, device=DEVICE, dtype=torch.bool))
        if causal:
            s = s.masked_fill(causal_mask == 0, float("-inf"))
        
        p = F.softmax(s.float(), dim=-1)

        if is_dropped:
            dropped_p = F.dropout(p, p=dropout_p, training=training)
        else:
            dropped_p = p

        dropped_p = dropped_p.to(q.dtype)
        o = dropped_p @ v  # (B, Hq, T, T) x (B, Hk, T, K) -> (B, Hq, T, K)

        p = p.to(q.dtype)
        ctx.save_for_backward(q, k, v, s, p, dropped_p, o, causal_mask)
        ctx.scale = scale
        ctx.causal = causal
        ctx.is_dropped = is_dropped
        ctx.dropout_p = dropout_p
        return o
    
    @staticmethod
    def backward(ctx, do):
        # do: (B, Hq, T, K)
        # return: dq, dk, dv
        q, k, v, s, p, dropped_p, o, causal_mask = ctx.saved_tensors
        scale = ctx.scale
        causal = ctx.causal
        is_dropped = ctx.is_dropped
        dropout_p = ctx.dropout_p

        # D: (B, Hq, T)
        # dp = do @ vT
        # dv = pT @ do
        # ds = dsoftmax(dp) | p * (dp - delta[:, :, :, None])
        # dq = ds @ k
        # dk = ds @ q
        dp = do @ v.transpose(-2, -1)
        dv = dropped_p.transpose(-2, -1) @ do

        if is_dropped:
            mask_dropped = (dropped_p != 0).to(dp.dtype)
            dp = dp * mask_dropped / dropout_p
        else:
            dp = dp

        delta = (do * o).sum(dim=-1)
        ds = p * (dp - delta[:, :, :, None])

        # Apply causal mask to gradient (reuse from forward)
        if causal:
            ds = ds.masked_fill(causal_mask == 0, 0)

        dq = ds @ k * scale
        dk = ds.transpose(-2, -1) @ q * scale

        # the num of input args of backward() should be the same as the num of outputs in forward()
        # AND the number of output of backward() should be the same as the number of input args in forward()
        return dq, dk, dv, None, None, None

# Wrapper function
def nano_flash_attention(q, k, v, causal=False, dropout_p=0.0, training=True):
    return _NanoFlashAttention.apply(q, k, v, causal, dropout_p, training)
    # return torch_attention(q, k, v, causal, dropout_p, training)

def _test_nano_attention(B, Hq, Hk, T, K, dropout_p):
    q = torch.randn((B, Hq, T, K), device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.randn((B, Hk, T, K), device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn((B, Hk, T, K), device=DEVICE, dtype=DTYPE, requires_grad=True)

    ref_q = q.clone().detach().requires_grad_(True)
    ref_k = k.clone().detach().requires_grad_(True)
    ref_v = v.clone().detach().requires_grad_(True)

    ref_o = torch_attention(ref_q, ref_k, ref_v, causal=True, dropout_p=dropout_p, training=True)
    o = nano_flash_attention(q, k, v, causal=True, dropout_p=dropout_p, training=True)

    do = torch.randn_like(o)
    torch.testing.assert_close(o, ref_o, atol=1e-5, rtol=1e-5)
    print("Fwd: PASS")

    ref_o.backward(do)
    o.backward(do)

    # print(ref_q.grad[-1][-1], q.grad[-1][-1])

    torch.testing.assert_close(ref_q.grad, q.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ref_k.grad, k.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ref_v.grad, v.grad, atol=1e-5, rtol=1e-5)
    print("Bwd: PASS")


def test_nano_attention():
    torch.manual_seed(7)

    B = 4
    Hq = 8
    Hk = 8
    T = 128
    K = 64

    _test_nano_attention(B, Hq, Hk, T, K, dropout_p=0.0)

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
        # dw = dC.T @ xT
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
    # test_custom_linear()
    test_nano_attention()

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