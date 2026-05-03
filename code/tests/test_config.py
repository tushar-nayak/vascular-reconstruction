from __future__ import annotations

from vascular_reconstruction.config import TrainingConfig


def test_training_config_save_load(tmp_path):
    config = TrainingConfig(experiment_name="test_exp", iterations=100)
    config_path = tmp_path / "config.json"
    config.save(config_path)

    loaded_config = TrainingConfig.load(config_path)
    assert loaded_config.experiment_name == "test_exp"
    assert loaded_config.iterations == 100
    assert loaded_config.learning_rate == 0.001
    assert loaded_config.pinn_learning_rate == 0.0001
    assert loaded_config.physics_loss_weight == 0.05
    assert loaded_config.device == "auto"
