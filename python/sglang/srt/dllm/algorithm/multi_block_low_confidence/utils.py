import dataclasses


@dataclasses.dataclass
class DecodingThresholds:
    add_block_threshold: float  # whether add a new block
    semi_complete_threshold: float  # whether unleash the decoding of the next block
    decoding_threshold: float  # whether the decoding is complete
