import pandas as pd

from arccnet.models.fulldisk.yolo import dataset_config as cfg
from arccnet.models.fulldisk.yolo import make_yolo_dataset as mk


def test_encode_labels_filters_none(monkeypatch):
    monkeypatch.setattr(
        cfg,
        "LABEL_MAPPING",
        {
            "IA": "None",
            "Alpha": "Alpha",
            "Beta": "Beta",
        },
    )

    df = pd.DataFrame({"magnetic_class": ["IA", "Alpha", "Beta", "Alpha"]})
    encoded, label_to_index = mk._encode_labels(df)

    assert encoded["grouped_label"].tolist() == ["Alpha", "Beta", "Alpha"]
    assert label_to_index == {"Alpha": 0, "Beta": 1}
    assert encoded["encoded_label"].tolist() == [0, 1, 0]


def test_validate_bbox_checks_bounds_and_dimensions():
    img_sizes = {"MDI": 1024}
    valid = pd.Series(
        {
            "bottom_left_cutout": (10, 20),
            "top_right_cutout": (110, 120),
            "instrument": "MDI",
        }
    )
    invalid = pd.Series(
        {
            "bottom_left_cutout": (50, 20),
            "top_right_cutout": (40, 120),  # negative width
            "instrument": "MDI",
        }
    )

    assert mk._validate_bbox(valid, img_sizes) is True
    assert mk._validate_bbox(invalid, img_sizes) is False


def test_split_temporal_falls_back_when_gap_unavailable(monkeypatch):
    monkeypatch.setattr(cfg, "TRAIN_SPLIT_RATIO", 0.6)
    monkeypatch.setattr(cfg, "TEMPORAL_GAP_DAYS", 1000)

    df_yolo = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]),
            "path_mag": [f"img_{idx}" for idx in range(5)],
            "yolo_label": [""] * 5,
        }
    )

    train_df, val_df, gap = mk._split_temporal(df_yolo)

    assert len(train_df) == 3
    assert len(val_df) == 2
    assert gap == 1
