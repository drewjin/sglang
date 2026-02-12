from typing import List, Tuple, Union

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.algorithm.multi_block.thresholds import MultiBlockThresholds
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class MultiBlockLowConfidence(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.thresholds = MultiBlockThresholds(
            **config.algorithm_config.get(
                "multi_block_thresholds",
                {
                    "decoding_threshold": 0.95,
                    "add_block_threshold": 0.50,
                    "semi_complete_threshold": 0.95,
                },
            ),
        )

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        pass
