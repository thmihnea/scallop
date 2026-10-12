"""Recipe validation settings must override system and preset defaults."""
import unittest
from unittest.mock import Mock, patch

from launchers import learn


def resolve(cfg="configs.aaa", mode="dlogpdt", *flags):
    with patch("sys.argv", ["learn.py", "--cfg_path", cfg,
                           "--mode", mode, "--device", "cpu", *flags]):
        return learn.setup_config_dict(learn.parse_command_line_arguments())


class ValidationCLI(unittest.TestCase):
    def test_explicit_validation_settings_reach_every_stage(self):
        for cfg in ["configs.ace_a_nme", "configs.aaa",
                    "configs.ace_aaa_nme", "configs.aaaaaa"]:
            for mode in ["pretrain_v", "dlogpdt", "hutchinson"]:
                with self.subTest(cfg=cfg, mode=mode):
                    config = resolve(cfg, mode, "--n_sampling_steps", "4",
                                     "--importance_weight_clip", "0.002")
                    self.assertEqual(config.logging.n_sampling_steps, [4])
                    self.assertEqual(config.logging.importance_weight_clip, 0.002)


    def test_validation_flags_override_recipe_and_support_multiple_counts(self):
        config = resolve("configs.aaa", "dlogpdt",
                         "--training_recipe", "branched_muon",
                         "--n_sampling_steps", "4", "8", "--importance_weight_clip", "0")
        self.assertEqual(config.logging.n_sampling_steps, [4, 8])
        self.assertEqual(config.logging.importance_weight_clip, 0)


    def test_invalid_validation_settings_fail_before_training(self):
        for flag, value, message in [
            ("--n_sampling_steps", "0", "must be positive"),
            ("--n_sampling_steps", "-1", "must be positive"),
            ("--importance_weight_clip", "-0.1", "must be finite"),
            ("--importance_weight_clip", "1", "must be finite"),
            ("--importance_weight_clip", "nan", "must be finite"),
            ("--importance_weight_clip", "inf", "must be finite"),
        ]:
            with self.subTest(flag=flag, value=value), self.assertRaisesRegex(ValueError, message):
                resolve("configs.aaa", "dlogpdt", flag + "=" + value)


    def test_omitted_validation_flags_use_readme_defaults(self):
        config = resolve()
        self.assertEqual(config.logging.n_sampling_steps, [4])
        self.assertEqual(config.logging.importance_weight_clip, 0.002)

    def test_wandb_is_opt_in(self):
        config = resolve()
        for flags, enabled in [([], False), (["--wandb"], True),
                               (["--require_wandb"], True), (["--no_wandb"], False),
                               (["--wandb", "--no_wandb"], False)]:
            with self.subTest(flags=flags), patch("sys.argv", ["learn.py", *flags]), \
                    patch.object(learn, "wandb") as wandb:
                run = wandb.init.return_value
                run.settings.mode = "online"
                args = learn.parse_command_line_arguments()
                self.assertIs(learn._init_wandb(config, args), run if enabled else None)
                self.assertEqual(wandb.init.call_count, int(enabled))

    def test_required_wandb_still_fails_when_unavailable_or_disabled(self):
        for wandb, flags in [(None, []), (Mock(), ["--no_wandb"])]:
            with self.subTest(flags=flags), patch.object(learn, "wandb", wandb), \
                    patch("sys.argv", ["learn.py", "--require_wandb", *flags]):
                args = learn.parse_command_line_arguments()
                with self.assertRaisesRegex(RuntimeError, "installed and enabled"):
                    learn._init_wandb(resolve(), args)


if __name__ == "__main__":
    unittest.main()
