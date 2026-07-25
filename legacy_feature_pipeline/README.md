# Legacy 2048-D Feature Pipeline

This directory preserves the previous official-distilled-image -> 2048-D feature
-> linear Stage B/C implementation for historical reproducibility only.

It is not imported by the current algorithmic main pipeline. The current method
uses distilled NCHW images, full ConvNet local/KD models, full-state candidate
aggregation, and the same chosen ConvNet in Stage C.

Legacy outputs must retain their historical names. New direct-image outputs use
the `crossStageConvNet_` prefix so the two experiment families cannot be mixed.
