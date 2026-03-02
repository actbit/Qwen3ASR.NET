// Disabled for broader GPU support (SM75, SM60)
// Original file uses bfloat16 WMMA which requires SM80+
// Qwen3-ASR doesn't use MOE, so this is safe to disable
