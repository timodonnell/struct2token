# Training Log

## Run: first-run

- **Date**: 2026-02-23
- **WandB**: https://wandb.ai/timodonnell/struct2token/runs/czwlf7ie
- **GPU**: A100 80GB (56GB used), 100% utilization
- **Config**: `configs/default.yaml` (defaults)
- **Model**: 79,407,111 parameters
- **Dataset**: 1,135,668 chains (1.02M protein, 62k RNA, 69k small molecule)
- **Precision**: float32
- **Batch size**: 2
- **Optimizer**: AdamW, lr=3e-4 with 1000-step warmup + cosine decay

### Training progress

| Step | flow_loss | size_loss | lr | grad_norm |
|------|-----------|-----------|-----|-----------|
| 100  | 1.7768    | 9.4807    | 2.97e-05 | 3.97  |
| 200  | 1.7782    | 9.0627    | 5.97e-05 | 22.10 |
| 300  | 0.7579    | 9.0691    | 8.97e-05 | 14.97 |
| 400  | 0.6774    | 8.2093    | 1.20e-04 | 13.11 |
| 500  | 0.6759    | 9.1405    | 1.50e-04 | 33.74 |
| 600  | 0.7714    | 9.0514    | 1.80e-04 | 21.89 |
| 700  | 1.5874    | 9.0182    | 2.10e-04 | 9.56  |
| 800  | 1.2347    | 9.2133    | 2.40e-04 | 15.91 |
| 900  | 0.8812    | 9.0617    | 2.70e-04 | 7.28  |
| 1000 | 0.5780    | 8.0517    | 3.00e-04 | 12.51 |
| 1100 | 0.6246    | 8.6306    | 3.00e-04 | 2.85  |
| 1200 | 0.8548    | 8.0822    | 3.00e-04 | 5.49  |

### Notes

- Flow loss dropping from ~1.8 to ~0.6 in first 1200 steps. Warmup ends at step 1000.
- Size loss (atom count prediction CE) starting around 9.0, slowly decreasing.
- Grad norms spiky (2-34), gradient clipping at 1.0 keeping things stable.
- ~8s/step with num_workers=4 and on-the-fly mmCIF parsing. Cache warms up over time.
- SDPA fallback attention (no Flash Attention installed). Install with `uv sync --extra flash` for faster training.
