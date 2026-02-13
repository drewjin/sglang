from __future__ import annotations

import enum
from typing import TYPE_CHECKING, List, Optional

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.util.block import DllmBlock, DllmBlockBuffer, DllmBlockStatus

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


class ReqMultiBlockDllmMixin:
    def init_multi_block_dllm(self: Req):
        self.dllm_blocks: List[DllmBlock] = []
        self.dllm_buffer: DllmBlockBuffer = None

    def _extend_block(self: Req, extend_size: int):
        self.dllm_ids += [self.dllm_config.mask_token_id] * extend_size

    def _determine_multi_block_dllm_phase(self: Req):
        pass

    def _init_fill_ids_for_multi_block_dllm(self: Req):
        if not self.dllm_ids:
            self.dllm_ids = self.origin_input_ids
            self._init_blocks()
        else:
            self._next_diffusion_step()

        self.fill_ids = self.dllm_ids

    def _init_blocks(self: Req):
        self.dllm_ids = self.origin_input_ids
        prefix_length = len(self.dllm_ids)
        padding_length = prefix_length % self.dllm_config.block_size
        self._extend_block(padding_length)

        # Initialize prefix dLLM blocks
        num_prefix_blocks = len(self.dllm_ids) // self.dllm_config.block_size
        for block_id in range(num_prefix_blocks):
            block = DllmBlock(
                req=self,
                mask_token_id=self.dllm_config.mask_token_id,
                block_id=block_id,
                block_size=self.dllm_config.block_size,
                block_status=(
                    DllmBlockStatus.TO_CACHE
                    if block_id != num_prefix_blocks - 1
                    else DllmBlockStatus.ACTIVE
                ),
                prev_block=None if block_id == 0 else self.dllm_blocks[block_id - 1],
            )
            self.dllm_blocks.append(block)

        # Initialize buffer blocks
        remain_buffer_size = (
            self.buffer_size - 1 if padding_length > 0 else self.buffer_size
        )
        for block_id in range(
            num_prefix_blocks, num_prefix_blocks + remain_buffer_size
        ):
            block = DllmBlock(
                req=self,
                mask_token_id=self.dllm_config.mask_token_id,
                block_id=block_id,
                block_size=self.dllm_config.block_size,
                block_status=DllmBlockStatus.DUMMY,
                prev_block=self.dllm_blocks[-1],
            )
            self.dllm_blocks.append(block)

        self.dllm_block_buffer = DllmBlockBuffer(
            req=self,
            buffer_size=self.dllm_config.buffer_size,
            dllm_blocks=self.dllm_blocks[-self.dllm_config.buffer_size :],
        )

    def _next_diffusion_step(self: Req):
        if (
            self.dllm_block_buffer.should_add_block()
            and not self.dllm_block_buffer.is_overflow()
        ) or (
            not self.block_buffer.active_blocks
            and self.block_buffer.blocks[0].is_dummy
            and self.block_buffer.blocks[0].prev_block.is_in_cache
        ):
            self.dllm_block_buffer.activate_cursor_block()


class ReqDllmMixin(ReqMultiBlockDllmMixin):
    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
        self.dllm_phase: Optional[DllmReqPhase] = None
        self.dllm_ids = []
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config

        if self.dllm_config is not None:
            if len(self.origin_input_ids) < self.dllm_config.block_size:
                self.dllm_phase = DllmReqPhase.INCOMING_DECODE
            else:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL

        if self.is_multi_block_dllm():
            self.init_multi_block_dllm()

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def is_multi_block_dllm(self: Req) -> bool:
        if self.dllm_config is None:
            raise ValueError("dllm_config is not set")
        return self.dllm_config.algorithm == "LowConfidenceMultiBlock"

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase in [
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.INCOMING_PREFILL,
        ]

    def determine_dllm_phase(self: Req):
        if self.is_multi_block_dllm():
            self._determine_multi_block_dllm_phase()
            return

        prefix_length = len(self.prefix_indices)
        min_required_length = prefix_length + self.dllm_config.block_size

        if len(self.fill_ids) < min_required_length:
            # still incoming stage
            return

        input_block = self.fill_ids[prefix_length:min_required_length]
        is_prefill_phase = self.dllm_config.mask_token_id not in input_block

        if is_prefill_phase:
            self.dllm_phase = DllmReqPhase.STAGING_PREFILL
        else:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _init_fill_ids_for_dllm(self: Req):
        if self.is_multi_block_dllm():
            self._init_fill_ids_for_multi_block_dllm()
            return

        if not self.dllm_ids:
            self.dllm_ids = self.origin_input_ids
            self._extend_block(self.dllm_config.block_size)
        else:
            self.dllm_block_offset += self.dllm_config.block_size
            self._extend_block(self.dllm_config.block_size)

        self.fill_ids = self.dllm_ids
