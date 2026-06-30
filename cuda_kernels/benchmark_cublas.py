import torch
import time

device = torch.device('cuda')

# Same matrix dims as the CUDA kernel
M, K, N = 1000, 256, 256
A = torch.randn(M, K, device=device)
B = torch.randn(K, N, device=device)

# Warmup
for _ in range(10):
    C = torch.mm(A, B)
torch.cuda.synchronize()

# Benchmark
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(100):
    C = torch.mm(A, B)
end.record()
torch.cuda.synchronize()

ms = start.elapsed_time(end) / 100
gflops = 2 * M * K * N / (ms * 1e6)
print(f"cuBLAS (torch.mm): {ms:.3f} ms avg  ({gflops:.2f} GFLOP/s)")
