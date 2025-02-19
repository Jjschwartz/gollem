import time

from gollem.data import load_dataset
from gollem.data.loader import DataLoader
from gollem.tokenizer import get_tokenizer


def test_dataloader():
    tokenizer = get_tokenizer("gpt2")
    dataset = load_dataset("tinyshakespeare", tokenizer)
    B, T = 10, 10
    dataloader = DataLoader(
        dataset.train_data_pattern, batch_size=B, seq_len=T, world_size=1, rank=0
    )
    position = dataloader.current_position

    start_time = time.time()
    for batch_num, batch in enumerate(iter(dataloader)):
        end_position = dataloader.current_position
        if end_position > 0:
            assert end_position == (batch_num + 1) * B * T
            assert end_position == position + B * T
        else:
            # we've reached the end of the dataset
            break
        # print(f"loaded [{position}:{end_position}] on shard {shard}")
        position = end_position
        batch_x, batch_y = batch
        assert batch_x.shape == (B, T)
        assert batch_y.shape == (B, T)
        assert (batch_x[:, 1:] == batch_y[:, :-1]).all()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


def test_dataloader_multi_process():
    tokenizer = get_tokenizer("gpt2")
    dataset = load_dataset("tinyshakespeare", tokenizer)
    B, T, W, R = 10, 10, 2, 1
    dataloader = DataLoader(
        dataset.train_data_pattern, batch_size=B, seq_len=T, world_size=W, rank=R
    )
    position = dataloader.current_position

    start_time = time.time()
    for batch_num, batch in enumerate(iter(dataloader)):
        end_position = dataloader.current_position
        if end_position < position:
            # we've reached the end of the dataset
            break
        assert end_position == ((batch_num + 1) * W + R) * B * T
        assert end_position == position + (W * B * T)
        batch_x, batch_y = batch
        assert batch_x.shape == (B, T)
        assert batch_y.shape == (B, T)
        assert (batch_x[:, 1:] == batch_y[:, :-1]).all()
        position = end_position
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


def test_dataloader_state_dict():
    tokenizer = get_tokenizer("gpt2")
    dataset = load_dataset("tinyshakespeare", tokenizer)
    B, T, W, R = 10, 10, 2, 1

    # test saving and loading dataloader with different ranks
    dataloader = DataLoader(
        dataset.train_data_pattern, batch_size=B, seq_len=T, world_size=W, rank=0
    )
    state_dict = dataloader.state_dict()
    assert state_dict["current_step"] == 0
    new_dataloader = DataLoader(
        dataset.train_data_pattern, batch_size=B, seq_len=T, world_size=W, rank=R
    )
    new_dataloader.load_state_dict(state_dict)
    assert new_dataloader.current_step == 0

    total_num_batches = dataloader.ntok_total // (B * T * W)
    for i in range(total_num_batches // 2):
        batch_x, batch_y = dataloader.next_batch()
        assert batch_x.shape == (B, T)

    state_dict = dataloader.state_dict()
    batch_num = total_num_batches // 2
    assert state_dict["current_step"] == batch_num
    new_dataloader.load_state_dict(state_dict)
    assert new_dataloader.current_step == batch_num
    assert new_dataloader.current_shard == 0  # there is only 1 shard
    assert new_dataloader.current_step_in_shard == batch_num
    assert new_dataloader.current_position == (batch_num * W + R) * B * T

    position = new_dataloader.current_position
    for batch_num in range(total_num_batches // 2, total_num_batches):
        batch_x, batch_y = new_dataloader.next_batch()
        end_position = new_dataloader.current_position
        if end_position < position:
            # we've reached the end of the dataset
            break
        assert end_position == ((batch_num + 1) * W + R) * B * T
        assert end_position == position + (W * B * T)
        assert batch_x.shape == (B, T)
        assert batch_y.shape == (B, T)
        assert (batch_x[:, 1:] == batch_y[:, :-1]).all()
        position = end_position
