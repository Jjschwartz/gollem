import dataclasses
import sys
from dataclasses import Field
from dataclasses import dataclass
from dataclasses import field
from pprint import pprint

import pyrallis

from gollem.models.config import ModelConfig
from gollem.models.gpt2.config import GPT2Config
from gollem.train.config import TrainConfig


@dataclass
class RunConfig:
    dataset: str
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


model_to_cfg_class = {
    "gpt2": GPT2Config,
}


def main():
    # first we get the cfg class from the id using sys.argv
    cfg_id = "gpt2"
    cfg_class = model_to_cfg_class[cfg_id]

    # modify the RunConfig to use the cfg_class
    # this is a a bit of a hack but seems to work
    RunConfig.__dataclass_fields__["model"].default_factory = cfg_class
    RunConfig.__dataclass_fields__["model"].type = cfg_class

    print("hello")

    # then we parse the config
    cfg = pyrallis.parse(config_class=RunConfig)
    pprint(cfg)


if __name__ == "__main__":
    main()
