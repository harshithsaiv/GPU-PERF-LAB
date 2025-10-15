import os, time, torch
torch.manual_seed(0)
device = "cuda"

M = int(os.getenv("M", 2048))
N = int(os.getenv("N", 2048))
K = int(os.getenv("K", 2048))
dtype = torch.float16

A = torch.randn(M, K, device=device, dtype=dtype)
B = torch.randn(K, N, device=device, dtype=dtype)

# warmup
for _ in range(5):
    C = A @ B
torch.cuda.synchronize()

# timed runs
it = int(os.getenv("IT", 30))
t0 = time.time()
for _ in range(it):
    C = A @ B
torch.cuda.synchronize()
t1 = time.time()

elapsed = (t1 - t0)/it
flops = 2.0 * M * N * K
tflops = (flops / elapsed) / 1e12
print(f"M={M} N={N} K={K} dtype={dtype}  avg_ms={elapsed*1e3:.3f}  TFLOPS={tflops:.2f}")
