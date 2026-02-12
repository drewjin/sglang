import enum
import dataclasses


class DllmBlockStatus(str, enum.Enum):
    ACTIVE = "active"
    TO_CACHE = "to_cache"
    IN_CACHE = "in_cache"
    DUMMY = "dummy"
    

@dataclasses.dataclass
class DllmBlock:
    block_id: int
    


@dataclasses.dataclass
class DllmBuffer:
    pass