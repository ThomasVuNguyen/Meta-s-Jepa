"""
arm3d_env.py — Gym-style environment wrapper for the custom 3-DOF arm.
Uses raw MuJoCo (not dm_control) so the MJCF is loaded directly.

Task: reach a randomly placed target sphere with the end-effector.
Observation: 224×224 RGB frame from fixed camera (fed to V-JEPA).
Action space: 3-dim continuous [-1, 1] (mapped to motor torques).
Reward: 1.0 - tanh(5 * distance) — smooth, peaks near 1.0 when close.
"""

import os
import numpy as np

# MuJoCo rendering backend
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco


class Arm3DEnv:
    """Minimal Gym-style wrapper for the 3-DOF reach task."""

    def __init__(
        self,
        xml_path: str | None = None,
        img_size: int = 224,
        max_steps: int = 100,
        seed: int = 0,
    ):
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "arm3d.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.img_size = img_size
        self.max_steps = max_steps
        self._step_count = 0
        self._rng = np.random.RandomState(seed)

        # Cache site IDs for distance computation
        self._ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        self._target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site"
        )
        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )

        # Camera ID for rendering
        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_cam"
        )

        # Action and observation dimensions
        self.action_dim = self.model.nu  # 3
        self.action_spec_min = -np.ones(self.action_dim, dtype=np.float32)
        self.action_spec_max = np.ones(self.action_dim, dtype=np.float32)

        # Renderer
        self._renderer = mujoco.Renderer(self.model, self.img_size, self.img_size)

    # ── Core API ─────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset environment, randomise target, return initial RGB frame."""
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        # Randomise target position within reachable workspace
        # Arm total reach ~0.24m, so target in a sphere of r ∈ [0.08, 0.22]
        r = self._rng.uniform(0.08, 0.20)
        theta = self._rng.uniform(0, 2 * np.pi)
        phi = self._rng.uniform(0.2, np.pi / 2)  # above ground
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi) + 0.04  # offset above base

        # Set target body position (it's a free-floating visual marker)
        self.model.body_pos[self._target_body_id] = [x, y, z]

        # Forward kinematics to update sites
        mujoco.mj_forward(self.model, self.data)

        return self.render()

    def step(self, action: np.ndarray):
        """
        Apply action, simulate, return (frame, reward, done, info).

        action: np.ndarray of shape (3,) in [-1, 1]
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        self.data.ctrl[:] = action

        # Simulate multiple sub-steps for stability
        n_substeps = 4
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Compute reward
        reward = self._compute_reward()
        done = self._step_count >= self.max_steps

        frame = self.render()

        return frame, reward, done, {"distance": self._get_distance()}

    def render(self) -> np.ndarray:
        """Render 224×224 RGB frame from the fixed camera."""
        self._renderer.update_scene(self.data, camera=self._cam_id)
        frame = self._renderer.render()
        return frame.copy()  # (H, W, 3) uint8

    # ── Internals ────────────────────────────────────────────────────

    def _get_distance(self) -> float:
        """Euclidean distance between end-effector and target."""
        ee_pos = self.data.site_xpos[self._ee_site_id]
        tgt_pos = self.data.site_xpos[self._target_site_id]
        return float(np.linalg.norm(ee_pos - tgt_pos))

    def _compute_reward(self) -> float:
        """Smooth reward: 1.0 when touching target, ~0 when far."""
        dist = self._get_distance()
        return float(1.0 - np.tanh(5.0 * dist))

    def action_spec(self):
        """Mimic dm_control action spec interface."""
        class ActionSpec:
            def __init__(self, minimum, maximum, shape):
                self.minimum = minimum
                self.maximum = maximum
                self.shape = shape
        return ActionSpec(self.action_spec_min, self.action_spec_max, (self.action_dim,))

    def close(self):
        """Clean up renderer."""
        if hasattr(self, '_renderer'):
            del self._renderer


# ── Quick sanity check ───────────────────────────────────────────────
if __name__ == "__main__":
    env = Arm3DEnv(seed=42)
    frame = env.reset()
    print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
    print(f"Action dim: {env.action_dim}")

    total_reward = 0
    for i in range(100):
        action = np.random.uniform(-1, 1, size=3).astype(np.float32)
        frame, reward, done, info = env.step(action)
        total_reward += reward
        if (i + 1) % 25 == 0:
            print(f"  Step {i+1}: reward={reward:.3f}, dist={info['distance']:.4f}")

    print(f"Total reward: {total_reward:.1f}")
    env.close()
    print("✓ Sanity check passed")
