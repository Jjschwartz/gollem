import time

from gollem.data import load_dataset
from gollem.data.loader import DataLoader
from gollem.tokenizer import get_tokenizer


def test_data_set():
    tokenizer = get_tokenizer("gpt2")
    dataset = load_dataset("tinyshakespeare", tokenizer)
    B, T = 10, 10
    dataloader = DataLoader(
        dataset.train_data_pattern, batch_size=B, seq_len=T, world_size=1, rank=0
    )
    position = dataloader.current_position
    shard = dataloader.current_shard

    start_time = time.time()
    for batch in iter(dataloader):
        end_position = dataloader.current_position
        end_shard = dataloader.current_shard
        if end_position > 0:
            assert end_position == position + B * T
        else:
            # we've reached the end of the dataset
            break
        # print(f"loaded [{position}:{end_position}] on shard {shard}")
        position = end_position
        shard = end_shard
        batch_x, batch_y = batch
        assert batch_x.shape == (B, T)
        assert batch_y.shape == (B, T)
        assert (batch_x[:, 1:] == batch_y[:, :-1]).all()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    test_data_set()
