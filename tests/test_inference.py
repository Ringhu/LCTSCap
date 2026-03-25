import torch

from lctscap.inference import (
    build_prediction_records,
    decode_sequences,
    encode_aux_timeseries,
    event_proposals_to_records,
    generate_from_prompt,
    normalize_prediction_text,
    resize_ts_windows,
    verbalize_event_evidence_text,
)


class DummyTokenizer:
    def batch_decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return ["  walked   to  the kitchen  ", "\n sat down \n"]


def test_normalize_prediction_text_collapses_whitespace():
    assert normalize_prediction_text("  one \n two\t three  ") == "one two three"


def test_decode_sequences_uses_tokenizer_batch_decode():
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    decoded = decode_sequences(DummyTokenizer(), token_ids)
    assert decoded == ["walked to the kitchen", "sat down"]


def test_build_prediction_records_preserves_metadata():
    metadata = [
        {"sample_id": "a", "dataset": "capture24", "participant_id": "p1", "split": "val", "context_len": 128},
        {"sample_id": "b", "dataset": "capture24", "participant_id": "p2", "split": "val", "context_len": 256},
    ]
    predictions = ["pred a", "pred b"]
    records = build_prediction_records(metadata, predictions)
    assert records == [
        {
            "sample_id": "a",
            "dataset": "capture24",
            "participant_id": "p1",
            "split": "val",
            "context_len": 128,
            "prediction": "pred a",
        },
        {
            "sample_id": "b",
            "dataset": "capture24",
            "participant_id": "p2",
            "split": "val",
            "context_len": 256,
            "prediction": "pred b",
        },
    ]


class FakeDecoder:
    def __init__(self):
        self.pos_embed = torch.zeros(1, 8, 4)
        self.cross_attn_proj = lambda x: x
        self.token_embed = lambda ids: torch.zeros(ids.size(0), ids.size(1), 4)
        self.decoder_layers = lambda tgt, memory, tgt_mask: torch.zeros_like(tgt)
        self.output_head = lambda last: torch.tensor([[0.0, 10.0, -1.0]]).repeat(last.size(0), 1)
        self._make_causal_mask = lambda seq_len, device: torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)


def test_generate_from_prompt_repeats_single_prompt_for_batch():
    decoder = FakeDecoder()
    encoder_output = torch.zeros(2, 3, 4)
    prompt_ids = torch.tensor([[5, 6]])
    generated = generate_from_prompt(
        decoder,
        encoder_output,
        prompt_ids,
        max_len=4,
        temperature=0.0,
        eos_token_id=1,
    )
    assert generated.shape == (2, 3)
    assert torch.equal(generated[:, :2], torch.tensor([[5, 6], [5, 6]]))


def test_resize_ts_windows_changes_signal_length():
    ts_windows = torch.arange(2 * 3 * 10, dtype=torch.float32).reshape(2, 3, 10)
    resized = resize_ts_windows(ts_windows, 20)
    assert resized.shape == (2, 3, 20)


class FakePlanner:
    segment_size = 32

    def __call__(self, fused):
        return fused, fused[:, :0, :]


class FakeAligner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ts_proj = torch.nn.Identity()

    def encode_ts(self, H_seg):
        return torch.nn.functional.normalize(H_seg.mean(dim=1), dim=-1)


class FakeAuxModel:
    def __init__(self):
        self.local_encoder = lambda x: torch.ones(x.size(0), 4)
        self.channel_fusion = lambda x: x.mean(dim=1)
        self.planner = FakePlanner()
        self.aligner = FakeAligner()


def test_encode_aux_timeseries_falls_back_to_token_mean_for_short_context():
    model = FakeAuxModel()
    ts_batch = torch.zeros(2, 3, 1, 500)
    z_ts = encode_aux_timeseries(model, ts_batch)
    assert z_ts.shape == (2, 4)
    assert torch.allclose(z_ts.norm(dim=-1), torch.ones(2))


def test_event_proposals_to_records_maps_ids_to_activity_names():
    proposals = [[(3, 10, 20), (1, 20, 25)]]
    idx_to_activity = {1: "sitting", 3: "walking"}
    records = event_proposals_to_records(proposals, idx_to_activity)
    assert records == [
        [
            {"activity": "walking", "start_token": 10, "end_token": 20},
            {"activity": "sitting", "start_token": 20, "end_token": 25},
        ]
    ]


class TinyTokenizer:
    bos_token_id = 101
    eos_token_id = 102
    pad_token_id = 0

    def __call__(self, texts, padding=True, truncation=True, max_length=256, return_tensors="pt"):
        mapping = {
            "a": [11, 12],
            "b": [21],
        }
        encoded = [mapping[text] for text in texts]
        max_len = max(len(seq) for seq in encoded)
        input_ids = []
        attention_mask = []
        for seq in encoded:
            pad_len = max_len - len(seq)
            input_ids.append(seq + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(seq) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def test_collator_builds_decoder_inputs_with_bos_eos_and_ignored_padding():
    from lctscap.data.collator import LCTSCapCollator

    collator = LCTSCapCollator(tokenizer=TinyTokenizer(), convert_events_to_per_token=False)
    batch = [
        {
            "ts_input": torch.zeros(2, 3, 4),
            "caption_short": "a",
            "caption_long": "a",
            "segment_summaries": [],
            "evidence_bullets": [],
            "metadata": {"sample_id": "x"},
            "events": [],
        },
        {
            "ts_input": torch.zeros(2, 3, 4),
            "caption_short": "b",
            "caption_long": "b",
            "segment_summaries": [],
            "evidence_bullets": [],
            "metadata": {"sample_id": "y"},
            "events": [],
        },
    ]

    out = collator(batch)
    assert torch.equal(
        out["decoder_input_ids"],
        torch.tensor([[101, 11, 12], [101, 21, 0]], dtype=torch.long),
    )
    assert torch.equal(
        out["decoder_attention_mask"],
        torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
    )
    assert torch.equal(
        out["target_ids"],
        torch.tensor([[11, 12, 102], [21, 102, -100]], dtype=torch.long),
    )


def test_verbalize_event_evidence_text_outputs_parseable_spans():
    text = verbalize_event_evidence_text([
        {"activity": "walking", "start_token": 0, "end_token": 10},
        {"activity": "sitting", "start_token": 10, "end_token": 20},
    ])
    assert text == "Evidence: walking windows 0 to 10; sitting windows 10 to 20."
