# Kaggle CPU Offload Pipeline

DM100 3D offloads heavy CPU multi-case validation suites and GNC parameter sweeps
to free Kaggle CPU kernels (4 vCPU, 30 GB RAM, up to 12 hours per kernel, zero GPU quota).
Running 4-5 sharded kernels in parallel provides 16-20 vCPU throughput with zero local
laptop heat or CPU load.

## One-Time Setup
1. Verify your Kaggle account phone number so kernel `enable_internet: true` is available.
2. Download `kaggle.json` from Kaggle Account Settings -> API -> Create New Token,
   and place it at `%USERPROFILE%\.kaggle\kaggle.json` (never commit it to git).
3. Configure `kernel-metadata.json` from `kernel-metadata.example.json` with your Kaggle username.

## Sharded Execution Contract
- Validation scripts accept `--shard <index>/<total> --resume <jsonl> --workers 4`.
- Outputs are written to `/kaggle/working/` and pulled back via `kaggle kernels output`.
