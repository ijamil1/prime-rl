import copy
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_tensor

from prime_rl.configs.trainer import GradientDiagnosticConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.data import TensorMicroBatch
from prime_rl.trainer.world import World
from prime_rl.utils.logger import get_logger


def _step_dir(root: Path, step: int) -> Path:
    return root / f"step_{step:06d}"


def _hash_update_value(hasher: Any, value: Any) -> None:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        hasher.update(b"tensor")
        hasher.update(str(tensor.dtype).encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        hasher.update(buffer.getvalue())
    elif isinstance(value, dict):
        hasher.update(b"dict")
        for key in sorted(value):
            hasher.update(str(key).encode())
            _hash_update_value(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        hasher.update(b"list")
        hasher.update(str(len(value)).encode())
        for item in value:
            _hash_update_value(hasher, item)
    elif value is None:
        hasher.update(b"none")
    else:
        hasher.update(repr(value).encode())


def _hash_value(value: Any) -> str:
    hasher = hashlib.sha256()
    _hash_update_value(hasher, value)
    return hasher.hexdigest()


def _to_cpu_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_value(item) for item in value)
    return value


def _materialize_tensor(tensor: Tensor | DTensor) -> Tensor:
    if isinstance(tensor, DTensor):
        return tensor.full_tensor().detach().cpu()
    return tensor.detach().cpu()


def _zeros_like_full_param(param: nn.Parameter) -> Tensor:
    return torch.zeros_like(_materialize_tensor(param), device="cpu")


def _load_torch_file(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _maybe_get_git_ref() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _make_dummy_tensor_micro_batch(source: TensorMicroBatch) -> TensorMicroBatch:
    dummy = copy.deepcopy(source)
    dummy["advantages"] = torch.zeros_like(source["advantages"])
    dummy["loss_mask"] = torch.zeros_like(source["loss_mask"], dtype=torch.bool)
    return dummy


def _pad_tensor_sequence(tensor: Tensor, padding_size: int, value: int | float | bool = 0) -> Tensor:
    padding_shape = (tensor.shape[0], padding_size, *tensor.shape[2:])
    padding = torch.full(padding_shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=1)


def _pad_replay_micro_batch_for_cp(source: TensorMicroBatch, cp_world_size: int) -> TensorMicroBatch:
    seq_len = source["input_ids"].shape[1]
    padding_size = -seq_len % cp_world_size
    if padding_size == 0 or cp_world_size <= 1:
        return source

    padded = copy.deepcopy(source)
    padded["input_ids"] = _pad_tensor_sequence(source["input_ids"], padding_size, value=1)
    padded["advantages"] = _pad_tensor_sequence(source["advantages"], padding_size, value=0.0)
    padded["loss_mask"] = _pad_tensor_sequence(source["loss_mask"], padding_size, value=False)
    position_padding = torch.arange(
        padding_size, dtype=source["position_ids"].dtype, device=source["position_ids"].device
    ).unsqueeze(0)
    padded["position_ids"] = torch.cat([source["position_ids"], position_padding], dim=1)
    padded["inference_logprobs"] = _pad_tensor_sequence(source["inference_logprobs"], padding_size, value=0.0)
    padded["temperatures"] = _pad_tensor_sequence(source["temperatures"], padding_size, value=1.0)
    if source["teacher_logprobs"] is not None:
        padded["teacher_logprobs"] = _pad_tensor_sequence(source["teacher_logprobs"], padding_size, value=0.0)
    padded["lora_num_tokens"] = source["lora_num_tokens"].clone()
    padded["lora_num_tokens"][-1] += padding_size
    if source["mm_token_type_ids"] is not None:
        padded["mm_token_type_ids"] = _pad_tensor_sequence(source["mm_token_type_ids"], padding_size, value=0)
    return padded


def _pad_replay_micro_batches_for_cp(
    micro_batches: list[TensorMicroBatch], cp_world_size: int
) -> list[TensorMicroBatch]:
    if cp_world_size <= 1:
        return micro_batches

    padded_micro_batches = [_pad_replay_micro_batch_for_cp(micro_batch, cp_world_size) for micro_batch in micro_batches]
    if all(padded is original for padded, original in zip(padded_micro_batches, micro_batches, strict=True)):
        return micro_batches
    return padded_micro_batches


def _pad_replay_micro_batches_for_distribution(
    micro_batches: list[TensorMicroBatch], dp_world_size: int
) -> list[TensorMicroBatch]:
    num_padding = -len(micro_batches) % dp_world_size
    if num_padding == 0 or not micro_batches:
        return micro_batches

    padded_micro_batches = list(micro_batches)
    dummy = _make_dummy_tensor_micro_batch(micro_batches[0])
    padded_micro_batches.extend([dummy] * num_padding)
    return padded_micro_batches


class GradientReplayDataLoader:
    def __init__(
        self,
        config: GradientDiagnosticConfig,
        dp_world_size: int,
        cp_world_size: int,
        world: World,
        start_step: int,
    ):
        if config.source_dir is None:
            raise ValueError("gradient diagnostic replay mode requires source_dir")
        if config.partition != "round_robin":
            raise ValueError(f"Unsupported gradient diagnostic replay partition: {config.partition}")
        self.config = config
        self.dp_world_size = dp_world_size
        self.cp_world_size = cp_world_size
        self.world = world
        self.non_dp_world_size = world.world_size // dp_world_size
        self.dp_rank = world.rank // self.non_dp_world_size
        self.non_dp_rank = world.rank % self.non_dp_world_size
        self.current_step = start_step
        self.last_batch_hash: str | None = None
        self.last_local_batch_hash: str | None = None
        self.last_execution_batch_hash: str | None = None
        self.last_execution_local_batch_hash: str | None = None
        self.last_replay_padding_micro_batches = 0
        self.last_replay_padding_tokens = 0

    def wait_for_batch(self) -> None:
        path = _step_dir(self.config.source_dir, self.current_step) / "micro_batches.pt"
        if not path.exists():
            raise FileNotFoundError(f"Recorded gradient diagnostic microbatch artifact not found: {path}")

    def get_batch(self) -> list[TensorMicroBatch]:
        micro_batches = load_micro_batches(self.config.source_dir, self.current_step)
        original_count = len(micro_batches)
        original_local_micro_batches = micro_batches[self.dp_rank :: self.dp_world_size]
        self.last_batch_hash = _hash_value(micro_batches)
        self.last_local_batch_hash = _hash_value(original_local_micro_batches)

        cp_padded_micro_batches = _pad_replay_micro_batches_for_cp(micro_batches, self.cp_world_size)
        self.last_replay_padding_tokens = sum(
            padded["input_ids"].shape[1] - original["input_ids"].shape[1]
            for padded, original in zip(cp_padded_micro_batches, micro_batches, strict=True)
        )
        execution_micro_batches = _pad_replay_micro_batches_for_distribution(
            cp_padded_micro_batches, self.dp_world_size
        )
        self.last_replay_padding_micro_batches = len(execution_micro_batches) - original_count
        self.last_execution_batch_hash = _hash_value(execution_micro_batches)
        local_micro_batches = execution_micro_batches[self.dp_rank :: self.dp_world_size]
        self.last_execution_local_batch_hash = _hash_value(local_micro_batches)
        if not local_micro_batches:
            raise ValueError(
                f"No replay microbatches assigned to DP rank {self.dp_rank}; "
                f"recorded={len(micro_batches)}, dp_world_size={self.dp_world_size}"
            )
        if self.last_replay_padding_micro_batches:
            get_logger().debug(
                "Padded gradient replay microbatches for DP distribution: "
                f"step={self.current_step}, recorded={original_count}, "
                f"padding={self.last_replay_padding_micro_batches}, "
                f"padded={len(execution_micro_batches)}, dp_world_size={self.dp_world_size}, dp_rank={self.dp_rank}"
            )
        if self.last_replay_padding_tokens:
            get_logger().debug(
                "Padded gradient replay microbatch tokens for CP: "
                f"step={self.current_step}, padding_tokens={self.last_replay_padding_tokens}, "
                f"cp_world_size={self.cp_world_size}, dp_rank={self.dp_rank}"
            )
        get_logger().debug(
            "Assigned gradient replay microbatches: "
            f"step={self.current_step}, rank={self.world.rank}, dp_rank={self.dp_rank}, "
            f"non_dp_rank={self.non_dp_rank}, world_size={self.world.world_size}, "
            f"dp_world_size={self.dp_world_size}, non_dp_world_size={self.non_dp_world_size}, "
            f"cp_world_size={self.cp_world_size}, recorded={original_count}, "
            f"padded={len(execution_micro_batches)}, padding={self.last_replay_padding_micro_batches}, "
            f"padding_tokens={self.last_replay_padding_tokens}, "
            f"original_local={len(original_local_micro_batches)}, execution_local={len(local_micro_batches)}, "
            f"local_batch_hash={self.last_local_batch_hash}, "
            f"execution_local_batch_hash={self.last_execution_local_batch_hash}"
        )
        self.current_step += 1
        return local_micro_batches


def save_micro_batches(root: Path, step: int, micro_batches: list[TensorMicroBatch]) -> str:
    cpu_micro_batches = _to_cpu_value(micro_batches)
    path = _step_dir(root, step)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_micro_batches, path / "micro_batches.pt")
    return _hash_value(cpu_micro_batches)


def load_micro_batches(root: Path, step: int) -> list[TensorMicroBatch]:
    return _load_torch_file(_step_dir(root, step) / "micro_batches.pt")


def collect_trainable_state(model: nn.Module) -> dict[str, Tensor]:
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = _materialize_tensor(param)
    return state


def load_trainable_state(model: nn.Module, root: Path, step: int) -> str:
    state = _load_torch_file(_step_dir(root, step) / "trainable_state.pt")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in state:
                raise KeyError(f"Recorded trainable state is missing parameter {name}")
            value = state[name].to(dtype=param.dtype)
            if isinstance(param, DTensor):
                distributed = distribute_tensor(
                    value,
                    device_mesh=param.device_mesh,
                    placements=param.placements,
                )
                param.copy_(distributed)
            else:
                param.copy_(value.to(device=param.device))
    return _hash_value(state)


def collect_gradients(model: nn.Module) -> dict[str, Tensor]:
    gradients = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            gradients[name] = _zeros_like_full_param(param)
        else:
            gradients[name] = _materialize_tensor(param.grad)
    return gradients


def save_trainable_state(model: nn.Module, root: Path, step: int, world: World) -> str | None:
    state = collect_trainable_state(model)
    state_hash = _hash_value(state)
    if world.is_master:
        path = _step_dir(root, step)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(state, path / "trainable_state.pt")
    return state_hash if world.is_master else None


def _parameter_metadata(gradients: dict[str, Tensor]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(gradient.shape),
            "dtype": str(gradient.dtype),
            "numel": gradient.numel(),
        }
        for name, gradient in gradients.items()
    ]


def _loss_scale_metadata(local_loss_scale: int, dp_mesh: DeviceMesh) -> dict[str, Any]:
    local = torch.tensor([local_loss_scale], dtype=torch.float64, device="cuda")
    dp_world_size = dp_mesh.size()
    gathered = [torch.zeros_like(local) for _ in range(dp_world_size)]
    dist.all_gather(gathered, local, group=dp_mesh.get_group())
    scales = torch.stack(gathered).squeeze().detach().cpu()
    return {
        "local_loss_scale": local_loss_scale,
        "dp_loss_scales": scales.tolist() if scales.ndim > 0 else [scales.item()],
        "dp_loss_scale_mean": scales.mean().item(),
        "dp_loss_scale_std": scales.std().item() if scales.numel() > 1 else 0.0,
        "dp_loss_scale_min": scales.min().item(),
        "dp_loss_scale_max": scales.max().item(),
    }


class GradientDiagnosticSession:
    def __init__(
        self,
        config: GradientDiagnosticConfig,
        model: nn.Module,
        parallel_dims: ParallelDims,
        world: World,
    ):
        self.config = config
        self.model = model
        self.parallel_dims = parallel_dims
        self.world = world
        self.batch_hash: str | None = None
        self.local_batch_hash: str | None = None
        self.execution_batch_hash: str | None = None
        self.execution_local_batch_hash: str | None = None
        self.replay_padding_micro_batches = 0
        self.replay_padding_tokens = 0
        self.pre_step_state_hash: str | None = None
        self.loss_metadata: dict[str, Any] = {}
        self.git_ref = _maybe_get_git_ref()

        if self.enabled and self.config.artifact_dir is None:
            raise ValueError("gradient diagnostic mode requires artifact_dir")
        if self.replay_enabled and self.config.source_dir is None:
            raise ValueError("gradient diagnostic replay mode requires source_dir")

    @property
    def enabled(self) -> bool:
        return self.config.mode != "off"

    @property
    def record_enabled(self) -> bool:
        return self.config.mode == "record"

    @property
    def replay_enabled(self) -> bool:
        return self.config.mode == "replay"

    @property
    def skip_optimizer_step(self) -> bool:
        return self.replay_enabled and self.config.skip_optimizer_step

    @property
    def artifact_dir(self) -> Path:
        if self.config.artifact_dir is None:
            raise ValueError("gradient diagnostic artifact_dir is not configured")
        return self.config.artifact_dir

    def build_replay_dataloader(self, start_step: int) -> GradientReplayDataLoader:
        return GradientReplayDataLoader(
            self.config, self.parallel_dims.get_mesh("dp").size(), self.parallel_dims.cp, self.world, start_step
        )

    def after_batch_loaded(self, step: int, micro_batches: list[TensorMicroBatch], dataloader: Any) -> None:
        if not self.enabled:
            return
        if self.record_enabled and self.world.is_master:
            self.batch_hash = save_micro_batches(self.artifact_dir, step, micro_batches)
            self.local_batch_hash = self.batch_hash
        elif self.replay_enabled:
            self.batch_hash = getattr(dataloader, "last_batch_hash", None)
            self.local_batch_hash = getattr(dataloader, "last_local_batch_hash", None)
            self.execution_batch_hash = getattr(dataloader, "last_execution_batch_hash", None)
            self.execution_local_batch_hash = getattr(dataloader, "last_execution_local_batch_hash", None)
            self.replay_padding_micro_batches = getattr(dataloader, "last_replay_padding_micro_batches", 0)
            self.replay_padding_tokens = getattr(dataloader, "last_replay_padding_tokens", 0)

    def align_pre_step_state(self, step: int) -> None:
        if not self.enabled:
            return
        if self.replay_enabled and self.config.load_pre_step_trainable_state:
            self.pre_step_state_hash = load_trainable_state(self.model, self.config.source_dir, step)
        if self.record_enabled and self.config.save_pre_step_trainable_state:
            self.pre_step_state_hash = save_trainable_state(self.model, self.artifact_dir, step, self.world)

    def collect_loss_metadata(self, local_loss_scale: int) -> None:
        if not self.enabled:
            return
        self.loss_metadata = _loss_scale_metadata(local_loss_scale, self.parallel_dims.get_mesh("dp"))

    def save_gradients(self, step: int, global_token_count: float | None = None) -> None:
        if not self.enabled:
            return
        gradients = collect_gradients(self.model)
        gradient_hash = _hash_value(gradients)
        if not self.world.is_master:
            return

        path = _step_dir(self.artifact_dir, step)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(gradients, path / "gradients.pt")

        metadata = {
            "step": step,
            "mode": self.config.mode,
            "git_ref": self.git_ref,
            "world_size": self.world.world_size,
            "dp_world_size": self.parallel_dims.get_mesh("dp").size(),
            "dp_replicate": self.parallel_dims.dp_replicate,
            "dp_shard": self.parallel_dims.dp_shard,
            "cp": self.parallel_dims.cp,
            "ep": self.parallel_dims.ep,
            "global_token_count": global_token_count,
            "batch_hash": self.batch_hash,
            "local_batch_hash": self.local_batch_hash,
            "execution_batch_hash": self.execution_batch_hash,
            "execution_local_batch_hash": self.execution_local_batch_hash,
            "replay_padding_micro_batches": self.replay_padding_micro_batches,
            "replay_padding_tokens": self.replay_padding_tokens,
            "pre_step_state_hash": self.pre_step_state_hash,
            "gradient_hash": gradient_hash,
            "parameters": _parameter_metadata(gradients),
            **self.loss_metadata,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
