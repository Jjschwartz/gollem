import sys
from dataclasses import dataclass
from dataclasses import field
from pprint import pprint

import pyrallis

from gollem.data import load_dataset
from gollem.models.gpt2.config import GPT2Config
from gollem.train.config import TrainConfig
from gollem.train.core import run


@dataclass
class RunConfig:
    # Name of the dataset to use (see `gollem.data.__init__.py`)
    dataset: str = field(default="tinyshakespeare")
    # GPT2Config
    model: GPT2Config = field(default_factory=GPT2Config)
    # Training configuration
    train: TrainConfig = field(default_factory=TrainConfig)


def main():
    try:
        print("Parsing config")
        cfg = pyrallis.parse(config_class=RunConfig)
        pprint(cfg)
    except Exception as e:
        print("Failed to parse config")
        print(e)
        sys.exit(1)

    model_cfg = cfg.model

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
