from __future__ import annotations

from synthdet.evaluation.analysis import _match_image


def test_error_matching_is_class_aware_and_deterministic() -> None:
    image = {
        "image_reference": "protected/example.jpg",
        "image_content_sha256": "a" * 64,
        "ground_truth": [
            {
                "class_id": 0,
                "class_name": "fish",
                "bbox_xywh_pixels": [10.0, 10.0, 20.0, 20.0],
                "area_pixels": 400.0,
            },
            {
                "class_id": 1,
                "class_name": "jellyfish",
                "bbox_xywh_pixels": [50.0, 50.0, 20.0, 20.0],
                "area_pixels": 400.0,
            },
        ],
        "predictions": [
            {
                "class_id": 0,
                "class_name": "fish",
                "bbox": [10.0, 10.0, 20.0, 20.0],
                "score": 0.9,
            },
            {
                "class_id": 2,
                "class_name": "penguin",
                "bbox": [50.0, 50.0, 20.0, 20.0],
                "score": 0.8,
            },
        ],
    }
    events = _match_image("synthetic_only", image)
    assert [event["error_type"] for event in events] == [
        "true_positive",
        "class_confusion",
        "false_negative",
    ]
