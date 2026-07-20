# Final Limitations

- The protected test contains 68 images from a limited aquarium source. It supports a controlled
  project comparison, not a population-wide deployment claim.
- Penguin appears in only four test images (20 objects), so its class estimate has high sampling
  uncertainty.
- Strong class imbalance, repeated scenes, reflections, occlusion, artificial lighting, and unknown
  per-image venue/date provenance constrain interpretation.
- The synthetic set is copy-paste augmentation dominated by real-train canvas pixels. It does not
  represent a fully rendered synthetic domain, and GrabCut boundaries or lighting mismatch can be
  visible.
- Mixed-regime allocation is controlled at image level; multi-label images prevent exact object- and
  class-frequency equality.
- Only YOLO11n, one seed, one training schedule, one synthetic generator, and one dataset were tested.
- No confidence interval or repeated-seed variance estimate is available; small differences should
  not be overinterpreted.
- The real-percentage curve is non-monotonic. Results are associational under this frozen design and
  do not prove a causal dose-response relation or universal best ratio.
- Size AP is a custom class-aware descriptive calculation under frozen pixel thresholds, not COCO's
  area-range implementation.
- CPU latency was measured inside the evaluation pass and excludes model loading. It is not a GPU or
  production capacity benchmark.
- Interactive inference thresholds are user controls for external demonstration images only. They
  do not modify the sealed scientific campaign.
- The service is a local educational deployment and has no authentication or multi-tenant isolation;
  it should not be exposed directly to the public internet.
