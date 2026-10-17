"""CPU-only tests of weight planning and mocked training setup; no real training."""
import contextlib
import io
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from component_lab.cli import main
from component_lab.training import build_training_sample_weights, train_resnet


class TrainingBalanceTests(unittest.TestCase):
    def test_square_root_relative_weights_cap_and_sample_mapping(self):
        result = build_training_sample_weights([0] * 100 + [1] * 25 + [2], 3, 5)
        self.assertEqual(result["class_counts"], [100, 25, 1])
        self.assertEqual(result["class_weights"], [1.0, 2.0, 5.0])
        self.assertEqual(result["sample_weights"], [1.0] * 100 + [2.0] * 25 + [5.0])
        self.assertEqual(build_training_sample_weights([0] * 100 + [1], 2, 1)["class_weights"], [1.0, 1.0])

    def test_parameter_boundaries_and_missing_training_classes(self):
        for cap in (0, -1, .9, float("inf"), float("nan"), True, "5"):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                build_training_sample_weights([0], 1, cap)
        for count in (0, -1, 1.5, True):
            with self.subTest(num_classes=count), self.assertRaises(ValueError):
                build_training_sample_weights([0], count)
        for targets in ([-1], [1], [0.5], [True]):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                build_training_sample_weights(targets, 1)
        for targets, classes in (([], 1), ([0], 2)):
            with self.assertRaisesRegex(ValueError, "zero samples"):
                build_training_sample_weights(targets, classes)

    def mocked_dependencies(self, train_targets, val_targets):
        """Fake optional vision modules so setup cannot load or train a model."""
        torch = ModuleType("torch")
        torch.manual_seed = MagicMock()
        torch.Generator = MagicMock()
        torch.Generator.return_value.manual_seed.return_value = torch.Generator.return_value
        torch.nn = SimpleNamespace(Linear=MagicMock(), CrossEntropyLoss=MagicMock())
        torch.optim = SimpleNamespace(AdamW=MagicMock())
        data = ModuleType("torch.utils.data")
        data.DataLoader, data.WeightedRandomSampler = MagicMock(), MagicMock()
        folder = ModuleType("torchvision.datasets")
        class Dataset:
            def __init__(self, targets):
                self.classes, self.targets = ["C", "R"], targets

            def __len__(self):
                return len(self.targets)
        folder.ImageFolder = MagicMock(side_effect=lambda path, **kw: Dataset(
            train_targets if Path(path).name == "train" else val_targets))
        models = ModuleType("torchvision.models")
        models.resnet18 = MagicMock()
        models.ResNet18_Weights = SimpleNamespace(DEFAULT="unused-pretrained-marker")
        models.resnet18.return_value.fc.in_features = 512
        modules = {"torch": torch, "torch.utils": ModuleType("torch.utils"), "torch.utils.data": data,
                   "torchvision": ModuleType("torchvision"), "torchvision.datasets": folder,
                   "torchvision.models": models}
        return modules, torch, data, models

    @contextlib.contextmanager
    def mocked_setup_files(self):
        # Keep setup tests entirely in memory, including the balance JSON write.
        with patch("component_lab.training.Path.exists", return_value=False), \
                patch("component_lab.training.Path.is_dir", return_value=True), \
                patch("component_lab.training.Path.iterdir", return_value=[Path("C"), Path("R")]), \
                patch("component_lab.training.Path.mkdir"), \
                patch("component_lab.training.Path.write_text") as write:
            yield write

    def run_mock_setup(self, train_targets, val_targets, **options):
        args = SimpleNamespace(data="data", output="output", batch=8,
                               no_pretrained=True, device="cpu", lr=.001, epochs=0, **options)
        modules, torch, data, models = self.mocked_dependencies(train_targets, val_targets)
        with patch.dict("sys.modules", modules), patch("component_lab.models.transform", return_value="transform"), \
                self.mocked_setup_files() as write:
            train_resnet(args)
        record = json.loads(write.call_args.args[0])
        return record, torch, data, models

    def test_sampler_is_seeded_uses_only_train_targets_and_no_loss_weights(self):
        train = [0] * 100 + [1] * 4
        record, torch, data, models = self.run_mock_setup(
            train, [1] * 10000, balance="sqrt_sampler", balance_cap=3)
        sampler_args = data.WeightedRandomSampler.call_args
        self.assertEqual(sampler_args.args[0], [1.0] * 100 + [3.0] * 4)
        self.assertEqual(sampler_args.kwargs["num_samples"], len(train))
        self.assertTrue(sampler_args.kwargs["replacement"])
        self.assertIs(sampler_args.kwargs["generator"], torch.Generator.return_value)
        torch.Generator.return_value.manual_seed.assert_called_once_with(42)
        train_loader = data.DataLoader.call_args_list[0]
        self.assertFalse(train_loader.kwargs["shuffle"])
        self.assertIs(train_loader.kwargs["sampler"], data.WeightedRandomSampler.return_value)
        self.assertFalse(data.DataLoader.call_args_list[1].kwargs["shuffle"])
        torch.nn.CrossEntropyLoss.assert_called_once_with()
        models.resnet18.assert_called_once_with(weights=None)
        self.assertEqual(record["class_counts"], {"C": 100, "R": 4})
        self.assertEqual(record["class_weights"], {"C": 1.0, "R": 3.0})
        self.assertEqual(record["mode"], "sqrt_sampler")
        self.assertEqual(record["seed"], 42)
        self.assertIsNone(record["loss_class_weights"])

    def test_legacy_args_default_to_unweighted_shuffle(self):
        record, _, data, _ = self.run_mock_setup([0, 0, 1], [0, 1])
        data.WeightedRandomSampler.assert_not_called()
        self.assertTrue(data.DataLoader.call_args_list[0].kwargs["shuffle"])
        self.assertIsNone(data.DataLoader.call_args_list[0].kwargs["sampler"])
        self.assertEqual(record["mode"], "none")
        self.assertEqual(record["class_weights"], {"C": 1.0, "R": 1.0})

    def test_empty_val_and_missing_train_class_rejected_before_model_setup(self):
        for train, val, message in (([0, 1], [], "Validation split has no images"),
                                    ([0], [0, 1], "zero samples.*R")):
            modules, _, _, models = self.mocked_dependencies(train, val)
            args = SimpleNamespace(data="data", output="output")
            with patch.dict("sys.modules", modules), patch("component_lab.models.transform", return_value="transform"), \
                    self.mocked_setup_files():
                with self.assertRaisesRegex(ValueError, message):
                    train_resnet(args)
            models.resnet18.assert_not_called()

    def test_cli_balance_defaults_choices_and_cap_validation(self):
        base = ["component-lab", "train-resnet", "--data", "data", "--output", "output"]
        for extra, mode, cap in (([], "none", 5), (["--balance", "sqrt_sampler", "--balance-cap", "3"], "sqrt_sampler", 3)):
            with patch("sys.argv", base + extra), patch("component_lab.training.train_resnet") as train:
                main()
            args = train.call_args.args[0]
            self.assertEqual(args.balance, mode)
            self.assertEqual(args.balance_cap, cap)
        for extra in (["--balance", "invalid"], ["--balance-cap", "0"], ["--balance-cap", "nan"], ["--balance-cap", "inf"]):
            with patch("sys.argv", base + extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()


if __name__ == "__main__":
    unittest.main()
