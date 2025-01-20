import tempfile

import torch
from gollem.models.gpt2.config import GPT2_CONFIG
from gollem.models.gpt2.model import GPT


def test_save_and_load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = GPT2_CONFIG
    with tempfile.TemporaryDirectory() as save_dir:
        model, _ = model_config.get_model_and_optimizer(device)
        model.save_model(model, f"{save_dir}/model.pt")
        loaded_model = GPT.load_model(f"{save_dir}/model.pt", device)
        assert type(loaded_model) is type(model)
        assert loaded_model.cfg == model.cfg

        model_state = model.state_dict()
        loaded_state = loaded_model.state_dict()
        assert model_state.keys() == loaded_state.keys()
        for key in model_state:
            assert torch.allclose(
                model_state[key], loaded_state[key]
            ), f"Mismatch in layer {key}"
