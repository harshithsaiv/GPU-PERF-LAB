# GPU Perf Notes — Matmul (L4, CUDA 12.4)

## Experiment Meta
- Date: $(date +%Y-%m-%d)
- GPU: NVIDIA L4 (24GB), Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
- CUDA: $(nvcc --version | head -n1)
- PyTorch: $(python -c "import torch;print(torch.__version__)")
- Commit: $(git rev-parse --short HEAD)
- Problem: MxK @ KxN (dtype: fp16/int8), batch: -

## Build/Launch Config
- BlockDim: 
- GridDim: 
- Shared Mem: 
- Num Warps / SM: 
- Tensor Cores: on/off
- Libraries: CUTLASS / custom CUDA / Triton

## Metrics (Nsight Compute)
- Achieved FLOPs (TFLOPS): 
- Tensor Core Utilization (%): 
- DRAM Throughput (GB/s): 
- L2 Hit Rate (%): 
- SM Occupancy (%): 
- Warp Stall Reasons (top 3): 
- Kernel Duration (ms): 

## Baseline vs Optimized
| Variant         | TFLOPS | DRAM GB/s | Occupancy | Time (ms) |
|-----------------|--------|-----------|-----------|-----------|
| PyTorch (torch.mm) |        |           |           |           |
| Custom CUDA v1  |        |           |           |           |
| Custom CUDA v2  |        |           |           |           |

## Bottlenecks Observed
- 

## Actions / Tweaks Tried
- [ ] Tile sizes
- [ ] CTA scheduling
- [ ] Shared-mem swizzle
- [ ] Vectorized LD/ST (ldmatrix)
- [ ] Double buffering / cp.async
- [ ] MMA shapes (m16n8k16, etc.)

## Next Experiments
- 

## Artifacts
- ncu report: ./matmul_nsight.ncu-rep
- csv: ./matmul_nsight.csv
- script: ./matmul_baseline.py
