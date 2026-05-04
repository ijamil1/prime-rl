import torch

from prime_rl.trainer.rl.data import TensorMicroBatch
from prime_rl.trainer.rl.diagnostics import (
    _hash_value,
    _make_dummy_tensor_micro_batch,
    _pad_replay_micro_batches_for_distribution,
)


def _micro_batch(offset: int = 0) -> TensorMicroBatch:
    input_ids = torch.tensor([[offset + 1, offset + 2, offset + 3]], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "position_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "advantages": torch.tensor([[1.0 + offset, 2.0 + offset, 3.0 + offset]], dtype=torch.float),
        "inference_logprobs": torch.tensor([[-0.1, -0.2, -0.3]], dtype=torch.float),
        "teacher_logprobs": torch.tensor([[-0.4, -0.5, -0.6]], dtype=torch.float),
        "loss_mask": torch.tensor([[True, False, True]], dtype=torch.bool),
        "temperatures": torch.ones_like(input_ids, dtype=torch.float),
        "lora_num_tokens": torch.tensor([input_ids.numel()], dtype=torch.int32),
        "routed_experts": None,
        "pixel_values": None,
        "image_grid_thw": None,
        "mm_token_type_ids": None,
        "sft_loss": False,
    }


def test_dummy_tensor_micro_batch_matches_main_dummy_semantics() -> None:
    source = _micro_batch()

    dummy = _make_dummy_tensor_micro_batch(source)

    assert torch.equal(dummy["advantages"], torch.zeros_like(source["advantages"]))
    assert torch.equal(dummy["loss_mask"], torch.zeros_like(source["loss_mask"], dtype=torch.bool))
    assert dummy["loss_mask"].sum().item() == 0
    assert dummy["lora_num_tokens"].sum().item() == dummy["input_ids"].numel()

    unchanged_keys = set(source) - {"advantages", "loss_mask"}
    for key in unchanged_keys:
        source_value = source[key]
        dummy_value = dummy[key]
        if isinstance(source_value, torch.Tensor):
            assert torch.equal(dummy_value, source_value)
        else:
            assert dummy_value == source_value


def test_replay_padding_appends_dummy_micro_batches_before_distribution() -> None:
    micro_batches = [_micro_batch(0), _micro_batch(10), _micro_batch(20)]

    padded = _pad_replay_micro_batches_for_distribution(micro_batches, dp_world_size=2)

    assert len(padded) == 4
    assert all(padded[idx] is micro_batches[idx] for idx in range(3))
    assert torch.equal(padded[3]["advantages"], torch.zeros_like(micro_batches[0]["advantages"]))
    assert torch.equal(padded[3]["loss_mask"], torch.zeros_like(micro_batches[0]["loss_mask"], dtype=torch.bool))

    rank_0_micro_batches = padded[0::2]
    rank_1_micro_batches = padded[1::2]
    assert len(rank_0_micro_batches) == len(rank_1_micro_batches) == 2


def test_replay_padding_leaves_even_batches_unchanged() -> None:
    micro_batches = [_micro_batch(0), _micro_batch(10), _micro_batch(20), _micro_batch(30)]

    padded = _pad_replay_micro_batches_for_distribution(micro_batches, dp_world_size=2)

    assert padded is micro_batches


def test_replay_padding_keeps_original_hash_separate_from_execution_hash() -> None:
    micro_batches = [_micro_batch(0), _micro_batch(10), _micro_batch(20)]
    original_batch_hash = _hash_value(micro_batches)
    original_rank_1_hash = _hash_value(micro_batches[1::2])

    padded = _pad_replay_micro_batches_for_distribution(micro_batches, dp_world_size=2)
    execution_batch_hash = _hash_value(padded)
    execution_rank_1_hash = _hash_value(padded[1::2])

    assert original_batch_hash == _hash_value(micro_batches)
    assert original_rank_1_hash == _hash_value(micro_batches[1::2])
    assert execution_batch_hash != original_batch_hash
    assert execution_rank_1_hash != original_rank_1_hash
