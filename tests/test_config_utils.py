from argparse import Namespace

from omegaconf import OmegaConf

from config_utils import apply_omegaconf_over_defaults


def test_config_values_fill_in_defaults_only_when_cli_left_them_unchanged():
    defaults = Namespace(iterations=1000, seed=0, source_path="", model_path="")
    args = Namespace(iterations=1000, seed=999, source_path="", model_path="")
    cfg = OmegaConf.create(
        {
            "training": {"iterations": 250},
            "runtime": {"seed": 123},
            "paths": {"source_path": "/tmp/source", "model_path": "/tmp/model"},
        }
    )

    apply_omegaconf_over_defaults(args, defaults, cfg)

    assert args.iterations == 250
    assert args.seed == 999
    assert args.source_path == "/tmp/source"
    assert args.model_path == "/tmp/model"
