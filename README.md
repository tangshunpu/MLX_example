# MLX_LM example

## Install
```pip install mlx-lm```
## Speculative decoding

### Model
- DEFAULT_DRAFT_MODEL = "mlx-community/Qwen3-0.6B-4bit"
- DEFAULT_TARGET_MODEL = "mlx-community/Qwen3-8B-4bit"

### Run
```python
python speculative_decode.py --prompt "Introduce Apple MLX framework" --num-draft-tokens 4 --max-tokens 2048 --temp 0.1 --top-p 0.9
```

### Results
```
accepted_from_draft: 769/1263
acceptance_rate:     60.89%
prompt_tps:          195.40
generation_tps:      67.85
peak_memory_gb:      5.44
wall_time_sec:       18.79
```
