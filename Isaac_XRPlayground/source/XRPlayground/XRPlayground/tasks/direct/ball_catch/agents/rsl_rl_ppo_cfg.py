# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config for Ball Catch (ONNX-export friendly)."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class BallCatchPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 2500
    save_interval = 50
    experiment_name = "xrplayground_ball_catch_direct"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.8),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.012,  # slightly higher — enclosure is sparse early
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.012,
        max_grad_norm=1.0,
    )


@configclass
class BallCatchWrapPPORunnerCfg(BallCatchPPORunnerCfg):
    """Phase Wrap: explore enclose/hold; entropy capped so action std stays usable."""

    max_iterations = 1500
    # Shared log root so Wrap → Throw-A → Throw-B resumes resolve.
    experiment_name = "xrplayground_ball_catch_2phase"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.35),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=2.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class BallCatchThrowAPPORunnerCfg(BallCatchPPORunnerCfg):
    """Throw-A: preserve wrap gripper skill; low entropy; arm frozen in env."""

    num_steps_per_env = 24
    max_iterations = 2500
    experiment_name = "xrplayground_ball_catch_2phase"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.30),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=2.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class BallCatchThrowBPPORunnerCfg(BallCatchPPORunnerCfg):
    """Throw-B: free arm intercept; higher entropy to re-learn tracking."""

    num_steps_per_env = 32
    max_iterations = 3500
    experiment_name = "xrplayground_ball_catch_2phase"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.55),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.010,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.012,
        max_grad_norm=1.0,
    )


@configclass
class BallCatchThrowPPORunnerCfg(BallCatchThrowBPPORunnerCfg):
    """Legacy alias → Throw-B PPO."""

    pass
