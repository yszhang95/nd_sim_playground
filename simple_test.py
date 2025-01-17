import torch

import torch.utils.benchmark as benchmark

def q(x):
    for i in range(20):
        x = torch.erf(x)
    return x

@torch.jit.script
def qjit(x):
    for i in range(20):
        x = torch.erf(x)
    return x

x = torch.rand(50_000_000, device='cuda', requires_grad=False)

qjit(x)
qjit(x)
qjit(x)
t1 = benchmark.Timer(stmt='q(x)', setup='from __main__ import q', globals={'x' : x})
t2 = benchmark.Timer(stmt='qjit(x)', setup='from __main__ import qjit', globals={'x' : x})

print(t1.blocked_autorange())
print(t2.blocked_autorange())

torch.cuda.memory._record_memory_history()
qjit(x)
torch.cuda.memory._dump_snapshot('simple_test.pickle')
