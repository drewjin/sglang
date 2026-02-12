import dataclasses


@dataclasses.dataclass
class MultiBlockThresholds:
    decoding_threshold: float
    add_block_threshold: float
    semi_complete_threshold: float
