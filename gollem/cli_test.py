import sys
from dataclasses import dataclass
from pprint import pprint

import pyrallis


@dataclass
class A:
    a: int


@dataclass
class B(A):
    b: str


@dataclass
class C(A):
    c: str


id_to_cfg_class = {
    "a": A,
    "b": B,
    "c": C,
}


def main():
    # first we get the cfg class from the id using sys.argv
    cfg_id = sys.argv[1]
    cfg_class = id_to_cfg_class[cfg_id]

    # then we parse the config
    cfg = pyrallis.parse(config_class=cfg_class)
    pprint(cfg)


if __name__ == "__main__":
    main()
