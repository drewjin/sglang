from typing import Any

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs


class DllmConfig:
    def __init__(
        self,
        algorithm: str,
        algorithm_config: dict[str, Any],
        buffer_size: int,
        block_size: int,
        mask_token_id: int,
        max_running_requests: int,
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.buffer_size = buffer_size
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.max_running_requests = max_running_requests

    @staticmethod
    def from_server_args(
        server_args: ServerArgs,
    ):
        if server_args.dllm_algorithm is None:
            return None

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
            model_revision=server_args.revision,
        )

        if model_config.hf_config.architectures[0] == "LLaDA2MoeModelLM":
            buffer_size = 1
            block_size = 32
            mask_token_id = 156895
        else:
            raise RuntimeError(
                f"Unknown diffusion LLM: {model_config.hf_config.architectures[0]}"
            )

        max_running_requests = (
            1
            if server_args.max_running_requests is None
            else server_args.max_running_requests
        )

        algorithm_config = {}
        if server_args.dllm_algorithm_config is not None:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Please install PyYAML to use YAML config files. "
                    "`pip install pyyaml`"
                )
            with open(server_args.dllm_algorithm_config, "r") as f:
                algorithm_config = yaml.safe_load(f)

            # Parse common algorithm configurations
            block_size = algorithm_config.get("block_size", block_size)
            buffer_size = algorithm_config.get("buffer_size", buffer_size)

        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            buffer_size=buffer_size,
            block_size=block_size,
            mask_token_id=mask_token_id,
            max_running_requests=max_running_requests,
        )
