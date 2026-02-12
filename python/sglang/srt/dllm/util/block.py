import enum
import weakref
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmBlockStatus(str, enum.Enum):
    ACTIVE = "active"
    TO_CACHE = "to_cache"
    IN_CACHE = "in_cache"
    DUMMY = "dummy"


class DllmBlock:
    def __init__(
        self,
        req: "Req",
        mask_token_id: int,
        block_id: int,
        block_size: int,
        block_status: DllmBlockStatus,
        add_block_threshold: float,
        decode_threshold: float,
        semi_complete_threshold: float,
        prev_block: Optional["DllmBlock"] = None,
    ):
        self.req = weakref.ref(req)
        self.dllm_block_buffer: Optional["DllmBlockBuffer"] = None

        self.mask_token_id = mask_token_id
        self.block_id = block_id
        self.block_size = block_size
        self.block_status = block_status
        self.prev_block = prev_block

        self.start = block_id * block_size
        self.end = (block_id + 1) * block_size

        self.add_block_threshold = add_block_threshold
        self.semi_complete_threshold = semi_complete_threshold
        self.decode_threshold = decode_threshold

    @property
    def is_active(self) -> bool:
        return self.block_status == DllmBlockStatus.ACTIVE

    @property
    def is_to_cache(self) -> bool:
        return self.block_status == DllmBlockStatus.TO_CACHE

    @property
    def is_in_cache(self) -> bool:
        return self.block_status == DllmBlockStatus.IN_CACHE

    @property
    def is_dummy(self) -> bool:
        return self.block_status == DllmBlockStatus.DUMMY

    @property
    def token_ids(self) -> List[int]:
        return self.dllm_ids[self.start : self.end]

    @property
    def decode_progress(self) -> float:
        return sum(self.token_ids == self.mask_token_id) / self.block_size

    def should_add_block(self) -> bool:
        return self.decode_progress >= self.add_block_threshold

    def is_semi_complete(self) -> bool:
        return self.decode_progress >= self.semi_complete_threshold


class DllmBlockBuffer:
    def __init__(
        self,
        req: "Req",
        buffer_size: int,
        dllm_blocks: List[DllmBlock],
    ):
        self.req = weakref.ref(req)
        self.buffer_size = buffer_size
        self.dllm_blocks = dllm_blocks

        for block in dllm_blocks:
            block.dllm_block_buffer = weakref.ref(self)

    def activate_cursor_block(self):
        # activate the block at the cursor slot index, which is the first dummy block
        self.cursor_block.block_status = DllmBlockStatus.ACTIVE

    def push_back(self, block: DllmBlock):
        self.dllm_blocks[-1] = block

    def pop_front(self):
        for i in range(0, self.buffer_size - 1):
            self.dllm_blocks[i] = self.dllm_blocks[i + 1]

    def should_add_block(self) -> bool:
        # return True if the cursor block is active and reached add_block_threshold
        return self.cursor_block.is_active and self.cursor_block.should_add_block()

    def is_overflow(self) -> bool:
        # return True if the number of active blocks is greater than the buffer size
        return len(self.active_blocks) > self.buffer_size

    @property
    def cursor_block(self) -> DllmBlock:
        # return the block at the cursor slot index, which is the first dummy block
        return self.dllm_blocks[self.cursor_slot_idx]

    @property
    def cursor_slot_idx(self) -> int:
        # return the index of the first dummy block
        return len(self.valid_blocks)

    @property
    def valid_blocks(self) -> List[DllmBlock]:
        # return all blocks that are NOT DUMMY
        return [block for block in self.dllm_blocks if not block.is_dummy]

    @property
    def dummy_blocks(self) -> List[DllmBlock]:
        # return all blocks that are DUMMY
        return [block for block in self.dllm_blocks if block.is_dummy]

    @property
    def active_blocks(self) -> List[DllmBlock]:
        # return all blocks that are ACTIVE
        return [block for block in self.dllm_blocks if block.is_active]
