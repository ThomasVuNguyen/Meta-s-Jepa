"""
Phase 4 — Model Predictive Control (MPC) Planner
================================================
This script evaluates the V-JEPA World Model on the dm_control `reacher_easy` task.
Given a goal state (an image), it uses the frozen V-JEPA encoder to get `z_goal`.
At each step `t`:
1. Encode current image to `z_t`
2. Sample N random action trajectories of length H.
3. Use the trained dynamics predictor `f(z, a)` to unroll the trajectory in latent space.
4. Select the action trajectory that minimizes distance to `z_goal` at step H.
5. Execute the first action.
"""
import torch
import numpy as np
from dm_control import suite
from transformers import AutoModel
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import imageio

# Import dynamics predictor from the training script
from train_dynamics import DynamicsPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

vjepa_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_observation_image(time_step):
    pixels = time_step.observation['pixels']
    img = Image.fromarray(pixels)
    return img

@torch.no_grad()
def encode_image(model, img: Image.Image):
    # Match the generate_and_encode_modal.py behavior exactly
    x = vjepa_transform(img) # [3, 224, 224]
    
    # We need 8 frames for the window. We'll just pad the same frame 8 times.
    window = [x] * 8
    clips_t = torch.stack(window).unsqueeze(0).to(DEVICE, dtype=torch.float16) # [1, 8, 3, 224, 224]
    
    # V-JEPA 2 forward pass
    outputs = model(pixel_values_videos=clips_t, return_dict=True)
    hidden_states = outputs.last_hidden_state # [1, 197, 1024]
    
    # Use global average pooling over all patches (excluding CLS if there is one)
    # The model we are using (facebook/vjepa2-vitl-fpc64-256) actually doesn't have a cls token
    # so we just average
    z = hidden_states.mean(dim=1) # [1, 1024]
    return z

class RandomShootingMPC:
    def __init__(self, dynamics_model, num_samples=1000, horizon=15, action_dim=2):
        self.dynamics_model = dynamics_model
        self.num_samples = num_samples
        self.horizon = horizon
        self.action_dim = action_dim
        
        # Action space bounds for reacher are roughly [-1, 1]
        self.action_low = -1.0
        self.action_high = 1.0

    @torch.no_grad()
    def plan(self, z_t, z_goal):
        """
        z_t: [1, 1024]
        z_goal: [1, 1024]
        Returns: best next action [1, 2]
        """
        # Step 1: Sample multiple random action trajectories
        # shape: [num_samples, horizon, action_dim]
        actions = torch.rand(self.num_samples, self.horizon, self.action_dim).to(DEVICE)
        actions = actions * (self.action_high - self.action_low) + self.action_low
        
        # Step 2: Unroll dynamics in latent space
        z_curr = z_t.expand(self.num_samples, -1) # [num_samples, 1024]
        
        for t in range(self.horizon):
            a_curr = actions[:, t, :] # [num_samples, 2]
            z_curr = self.dynamics_model(z_curr, a_curr)
            
        # Step 3: Compute cost (distance to goal at final step)
        # Using negative cosine similarity or L2 norm
        # We'll use L2 distance
        z_goal_expanded = z_goal.expand(self.num_samples, -1)
        distances = torch.norm(z_curr - z_goal_expanded, dim=-1) # [num_samples]
        
        # Step 4: Find best trajectory and return its *first* action
        best_idx = torch.argmin(distances)
        best_action = actions[best_idx, 0, :] # [action_dim]
        
        return best_action.cpu().numpy()

def main():
    print("Loading V-JEPA model...")
    vjepa_model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", trust_remote_code=True).to(DEVICE)
    vjepa_model.eval()

    print("Loading Dynamics Predictor...")
    dynamics_model = DynamicsPredictor(latent_dim=1024, action_dim=2, hidden_dim=512).to(DEVICE)
    dynamics_model.load_state_dict(torch.load("train_robots/models/dynamics_predictor.pt", weights_only=True))
    dynamics_model.eval()

    print("Initializing environment...")
    env = suite.load(domain_name="reacher", task_name="easy", task_kwargs={'random': 42})
    
    # Enable pixel rendering
    from dm_control.suite.wrappers import pixels
    env = pixels.Wrapper(env, pixels_only=False, render_kwargs={'height': 224, 'width': 224, 'camera_id': 0})
    
    # Generate a goal state by running a P-controller script to the end (or grabbing a random successful state)
    # For simplicity, we step env randomly until we get a success, and record that image as target
    print("Finding a goal state...")
    goal_img = None
    env.reset()
    for _ in range(500): # max find steps
        action = np.random.uniform(-1, 1, size=(2,))
        ts = env.step(action)
        if ts.reward and ts.reward > 0.9:
            goal_img = get_observation_image(ts)
            print(f"Goal state found! Reward: {ts.reward}")
            break
            
    if goal_img is None:
        print("Warning: Could not find a high-reward goal state via random exploration. Using last state.")
        goal_img = get_observation_image(ts)
        
    goal_img.save("train_robots/results/mpc_goal_target.png")
    z_goal = encode_image(vjepa_model, goal_img)

    # Initialize MPC
    planner = RandomShootingMPC(dynamics_model, num_samples=1000, horizon=10) # 10 steps lookahead

    print("\nStarting MPC Evaluation Run...")
    time_step = env.reset()
    frames = []
    total_reward = 0.0
    
    max_steps = 100
    for step in range(max_steps):
        img_t = get_observation_image(time_step)
        frames.append(np.array(img_t))
        z_t = encode_image(vjepa_model, img_t)
        
        # Plan!
        action = planner.plan(z_t, z_goal)
        
        # Execute
        time_step = env.step(action)
        reward = time_step.reward or 0.0
        total_reward += reward
        
        if (step+1) % 10 == 0:
            print(f"Step {step+1:3d}/{max_steps} | Action: [{action[0]:5.2f}, {action[1]:5.2f}] | Reward: {reward:.3f}")
            
    print(f"\nEvaluation Complete. Total Reward: {total_reward:.2f}")
    
    # Save video
    out_video = "train_robots/results/mpc_rollout.mp4"
    imageio.mimwrite(out_video, frames, fps=30)
    print(f"Saved MPC rollout to {out_video}")

if __name__ == "__main__":
    main()
