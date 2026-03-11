# MLX_LM example

## Install
```pip install mlx-lm```
## Speculative decoding

### Model
- DEFAULT_DRAFT_MODEL = "mlx-community/Qwen3-1.7B-4bit"
- DEFAULT_TARGET_MODEL = "mlx-community/Qwen3-8B-4bit"

### Run
```python
python speculative_decode.py --prompt "Introduce Singapore" --num-draft-tokens 4 --max-tokens 1024 --temp 0 --top-p 1
```

### Results
```
=== Comparison Summary ===
baseline_wall_time:  5.73 sec
spec_wall_time:      5.35 sec
wall_time_speedup:   1.07x
baseline_gen_tps:    92.65
spec_gen_tps:        98.33
generation_tps_gain: 1.06x
spec_acceptance:     40.82%
```
