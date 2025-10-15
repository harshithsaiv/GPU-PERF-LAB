import torch,time
A= torch.randn(1024,1024,device="cuda")
B= torch.randn(1024,1024,device="cuda")
torch.cuda.synchronize()
start = time.time()
for _ in range(10):
    C = A @ B
torch.cuda.synchronize()
print("Time:",time.time() -start)

