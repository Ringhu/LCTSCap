"""LCTSCap model components.

This package contains all neural network modules for the LCTSCap
(Large-scale Channel-wise Time-Series Captioning) framework:

- local_encoder: Channel-wise local patch embedding and Transformer encoder
- channel_fusion: Attention-based channel fusion
- planner: Hierarchical Planner with segment pooling
- event_head: Event proposal head for event detection
- aligner: CLIP-style retrieval aligner
- text_encoder: Sentence-transformer wrapper
- decoder: Transformer caption decoder
- llm_bridge: Perceiver resampler and LLM bridge (Phase 4)
- full_model: Combined model with ablation flags
- losses: All loss functions
"""
