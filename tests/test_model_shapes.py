"""Tests for model component shapes and ablation modes."""
import torch


def test_local_encoder_output_shape():
    from lctscap.models.local_encoder import LocalEncoder
    enc = LocalEncoder(d_model=64, patch_size=25, num_layers=1, num_heads=4, max_patches=20)
    x = torch.randn(6, 1, 500)
    out = enc(x)
    assert out.shape == (6, 64)


def test_local_encoder_different_lengths():
    from lctscap.models.local_encoder import LocalEncoder
    enc = LocalEncoder(d_model=128, patch_size=50, num_layers=1, num_heads=4, max_patches=10)
    x = torch.randn(4, 1, 500)
    out = enc(x)
    assert out.shape == (4, 128)


def test_channel_fusion_output_shape():
    from lctscap.models.channel_fusion import ChannelFusion
    fusion = ChannelFusion(d_model=64, num_channels=3)
    x = torch.randn(8, 3, 64)
    out = fusion(x)
    assert out.shape == (8, 64)


def test_channel_fusion_single_channel():
    from lctscap.models.channel_fusion import ChannelFusion
    fusion = ChannelFusion(d_model=64, num_channels=1)
    x = torch.randn(4, 1, 64)
    out = fusion(x)
    assert out.shape == (4, 64)


def test_planner_shapes():
    from lctscap.models.planner import HierarchicalPlanner
    planner = HierarchicalPlanner(d_model=64, num_layers=1, num_heads=4, segment_size=4)
    x = torch.randn(2, 16, 64)
    h_token, h_seg = planner(x)
    assert h_token.shape == (2, 16, 64)
    assert h_seg.shape == (2, 4, 64)  # 16/4 = 4 segments


def test_event_head_shapes():
    from lctscap.models.event_head import EventProposalHead
    head = EventProposalHead(d_model=64, n_event_types=10, max_events=8)
    x = torch.randn(2, 16, 64)
    out = head(x)
    assert "event_type_logits" in out
    assert out["event_type_logits"].shape == (2, 16, 10)
    assert "span_logits" in out
    assert out["span_logits"].shape == (2, 16, 2)


def test_decoder_shapes():
    from lctscap.models.decoder import CaptionDecoder
    dec = CaptionDecoder(d_model=64, vocab_size=100, num_layers=1, num_heads=4, max_seq_len=32)
    encoder_out = torch.randn(2, 4, 64)  # 4 segments
    tgt_ids = torch.randint(0, 100, (2, 10))
    tgt_mask = torch.tensor([[1] * 10, [1] * 8 + [0] * 2])
    logits = dec(encoder_out, tgt_ids, target_mask=tgt_mask)
    assert logits.shape == (2, 10, 100)


def test_full_model_with_hierarchy():
    from lctscap.models.full_model import LCTSCapModel, ModelConfig
    cfg = ModelConfig(
        d_model=64, num_channels=3, patch_size=25,
        local_encoder_layers=1, local_encoder_heads=2,
        planner_layers=1, planner_heads=2, segment_size=4,
        decoder_layers=1, decoder_heads=2,
        vocab_size=100, max_seq_len=16,
        n_event_types=5, max_events=4, d_align=32,
        no_align=True,
    )
    model = LCTSCapModel(cfg)
    x = torch.randn(2, 8, 3, 500)
    target_ids = torch.randint(0, 100, (2, 10))
    with torch.no_grad():
        out = model(x, target_ids=target_ids)
    assert out["H_token"].shape == (2, 8, 64)
    assert out["H_seg"].shape == (2, 2, 64)
    assert out["caption_logits"].shape[0] == 2
    assert out["event_proposals"] is not None


def test_full_model_no_hierarchy():
    from lctscap.models.full_model import LCTSCapModel, ModelConfig
    cfg = ModelConfig(
        d_model=64, num_channels=3, patch_size=25,
        local_encoder_layers=1, local_encoder_heads=2,
        planner_layers=1, planner_heads=2, segment_size=4,
        decoder_layers=1, decoder_heads=2,
        vocab_size=100, max_seq_len=16,
        n_event_types=5, max_events=4, d_align=32,
        no_hierarchy=True, no_align=True,
    )
    model = LCTSCapModel(cfg)
    x = torch.randn(2, 8, 3, 500)
    target_ids = torch.randint(0, 100, (2, 10))
    with torch.no_grad():
        out = model(x, target_ids=target_ids)
    assert out["H_seg"] is None
    assert out["caption_logits"] is not None


def test_full_model_no_event():
    from lctscap.models.full_model import LCTSCapModel, ModelConfig
    cfg = ModelConfig(
        d_model=64, num_channels=3, patch_size=25,
        local_encoder_layers=1, local_encoder_heads=2,
        planner_layers=1, planner_heads=2, segment_size=4,
        decoder_layers=1, decoder_heads=2,
        vocab_size=100, max_seq_len=16,
        n_event_types=5, max_events=4, d_align=32,
        no_event=True, no_align=True,
    )
    model = LCTSCapModel(cfg)
    x = torch.randn(2, 8, 3, 500)
    with torch.no_grad():
        out = model.encode(x)
    assert out[2] is None  # event_proposals should be None


def test_full_model_encode_decode_shapes():
    from lctscap.models.full_model import LCTSCapModel, ModelConfig
    cfg = ModelConfig(
        d_model=64, num_channels=6, patch_size=25,
        local_encoder_layers=1, local_encoder_heads=2,
        planner_layers=1, planner_heads=2, segment_size=4,
        decoder_layers=1, decoder_heads=2,
        vocab_size=50, max_seq_len=32,
        n_event_types=12, max_events=8, d_align=32,
        no_align=True,
    )
    model = LCTSCapModel(cfg)
    B, T, C, L = 1, 16, 6, 500
    x = torch.randn(B, T, C, L)
    target = torch.randint(0, 50, (B, 8))
    with torch.no_grad():
        out = model(x, target_ids=target)
    assert out["H_token"].shape == (B, T, 64)
    assert out["H_seg"].shape == (B, 4, 64)
    assert out["caption_logits"].shape == (B, 8, 50)
