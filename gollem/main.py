import sys
from dataclasses import dataclass
from dataclasses import field
from pprint import pprint

import pyrallis

from gollem.data import load_dataset
from gollem.models import get_model_config
from gollem.train.config import TrainConfig
from gollem.train.core import run


@dataclass
class RunConfig:
    # Name of the dataset to use (see `gollem.data.dataset_registry`)
    dataset: str = field(default="tiny_shakespeare")
    # Name of the model to use (see `gollem.models.model_registry`)
    model: str = field(default="gpt2")
    # Training configuration
    train: TrainConfig = field(default_factory=TrainConfig)


def main():
    try:
        cfg = pyrallis.parse(config_class=RunConfig)
        pprint(cfg)
    except Exception as e:
        sys.exit(1)

    # load model config
    model_cfg = get_model_config(cfg.model)

    # load dataset
    dataset = load_dataset(cfg.dataset, model_cfg.get_tokenizer())

    # Run training
    run(
        dataset_config=dataset,
        model_config=model_cfg,
        train_config=cfg.train,
    )


if __name__ == "__main__":
    main()
