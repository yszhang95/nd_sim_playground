import torch

import numpy as np
import time

def q(x):
    for i in range(80):
        x = torch.erf(x)
    return x

@torch.jit.script
def qjit(x):
    for i in range(80):
        x = torch.erf(x)
    return x

def test1(dtype=torch.float32):
    ns = []
    ts = []
    for i in range(20):
        if i<10:
            n = np.random.randint(10_000_000, 50_000_000)
        if i>10:
            n = np.random.randint(50_000_000, 60_000_000)
        x = 3*torch.rand(n, dtype=torch.float32, device='cuda', requires_grad=False)
        x = x.to(dtype)
        torch.cuda.synchronize()
        start_epoch = time.time()
        qjit(x)
        torch.cuda.synchronize()
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        ns.append(n)
        ts.append(elapsed)

    for i in range(20):
        n = ns[i]
        elapsed = ts[i]
        print('Elapsed', elapsed*1E3, 'ms for n =',n)
    for i in range(20):
        n = ns[i]
        elapsed = ts[i]
        print('Average is', elapsed*1E9/n, 'ns for n =',n)

test1()
test1()
test1(dtype=torch.int32)
