import torch

import numpy as np
import time

def q(x, m):
    idx0, idx1, idx2, idx3 = torch.where(m)
    y1 = x[idx0, idx1, idx2, idx3]
    y2 = x[idx0, idx2, idx3, idx1]
    y3 = x[idx0, idx2, idx1, idx3]
    return y1, y2, y3

@torch.jit.script
def qjit(x, m):
    idx0, idx1, idx2, idx3 = torch.where(m)
    y1 = x[idx0, idx1, idx2, idx3]
    y2 = x[idx0, idx2, idx3, idx1]
    y3 = x[idx0, idx2, idx1, idx3]
    return y1, y2, y3

def test1(dtype=torch.float32):
    ns = []
    ts = []
    n = 1_000
    d = 33
    m1 = torch.randint(0, 2, (n, d, d, d), dtype=torch.int32, requires_grad=False, device='cuda')
    m2 = torch.randint(0, 2, (n, d, d, d), dtype=torch.int32, requires_grad=False, device='cuda')
    x = torch.rand((n,d,d,d), dtype=torch.float32, device='cuda', requires_grad=False)
    torch.cuda.synchronize()
    start_epoch = time.time()
    qjit(x, m1)
    torch.cuda.synchronize()
    end_epoch = time.time()
    elapsed = end_epoch - start_epoch
    print('Elapsed', elapsed*1E3, 'ms for filling factor m = ', torch.sum(m1)/m1.numel())
    for i in range(30):
        qjit(x, m1)
    print('Finished warmup')

    for i in range(10):
        torch.cuda.synchronize()
        start_epoch = time.time()
        qjit(x, m1)
        torch.cuda.synchronize()
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        print('Elapsed', elapsed*1E3, 'ms for filling factor m = ', torch.sum(m1)/m1.numel())

    print(qjit.graph_for(x, m1))

    for i in range(10):
        torch.cuda.synchronize()
        start_epoch = time.time()
        qjit(x, m2)
        torch.cuda.synchronize()
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        print('Elapsed', elapsed*1E3, 'ms for filling factor m = ', torch.sum(m2)/m2.numel())

    print('python version')

    for i in range(20):
        torch.cuda.synchronize()
        start_epoch = time.time()
        q(x, m2)
        torch.cuda.synchronize()
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        print('Elapsed', elapsed*1E3, 'ms for filling factor m = ', torch.sum(m2)/m2.numel())
test1()
