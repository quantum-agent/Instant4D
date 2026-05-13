from omegaconf.dictconfig import DictConfig


def apply_omegaconf_over_defaults(args, defaults, cfg):
    """Apply OmegaConf leaf values only where the CLI still matches parser defaults.

    This preserves explicit command-line overrides while still allowing a config file
    to supply defaults for omitted arguments.
    """

    def recursive_merge(host):
        for key, value in host.items():
            if isinstance(value, DictConfig):
                recursive_merge(value)
            else:
                assert hasattr(args, key), key
                if getattr(args, key) == getattr(defaults, key):
                    setattr(args, key, value)

    recursive_merge(cfg)
