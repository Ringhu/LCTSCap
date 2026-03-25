import json

from lctscap.data.auxiliary_alignment import (
    load_ucr_tsv,
    make_aux_caption,
    save_aux_records,
    selected_ucr_datasets,
)


def test_selected_ucr_datasets_curated_shortlist():
    assert selected_ucr_datasets() == [
        "Chinatown",
        "ECG200",
        "EOGHorizontalSignal",
        "PLAID",
        "PowerCons",
        "SemgHandMovementCh2",
    ]


def test_load_ucr_tsv_parses_labels_and_nan(tmp_path):
    path = tmp_path / "toy.tsv"
    path.write_text("1\t0.1\tNaN\t0.3\n2\t1.0\t2.0\t3.0\n")
    rows = load_ucr_tsv(str(path))
    assert rows[0][0] == "1"
    assert rows[0][1].shape == (1, 3)
    assert rows[1][1][0, 2] == 3.0


def test_make_aux_caption_uses_dataset_template():
    text = make_aux_caption("PowerCons", "2")
    assert "cold-season household power consumption" in text


def test_make_aux_caption_supports_sleep_edf():
    text = make_aux_caption("SleepEDF", "REM")
    assert "REM sleep" in text


def test_save_aux_records_writes_manifest_and_tensors(tmp_path):
    manifest = save_aux_records(
        dataset_name="Chinatown",
        split_name="train",
        records=[("1", __import__("numpy").zeros((1, 4), dtype="float32"))],
        output_root=str(tmp_path),
    )
    assert len(manifest) == 1
    tensor_path = tmp_path / "Chinatown" / "tensors" / "train" / "Chinatown_train_000000.pt"
    manifest_path = tmp_path / "Chinatown" / "train_manifest.jsonl"
    assert tensor_path.exists()
    assert manifest_path.exists()
    line = json.loads(manifest_path.read_text().strip())
    assert line["caption_text"] == "This pedestrian-count time series corresponds to weekend pedestrian traffic."
