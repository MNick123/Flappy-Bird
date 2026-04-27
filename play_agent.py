"""
Watch your trained DQN agent play Flappy Bird!

Usage:
    python play_agent.py
"""

import torch
import numpy as np
import gymnasium as gym
import yaml
import time
from pathlib import Path

# Import your DQN class
from cudadqn import DuelingDQN as DQN


def play_agent(model_path='models/flappybird_turbo.pth', 
               config_name='flappybird_turbo',
               num_episodes=5,
               render_mode='human',
               fps=30):
    """
    Load and play the trained agent
    
    Args:
        model_path: Path to saved model
        config_name: Config name from hyperparameters.yml
        num_episodes: Number of games to play
        render_mode: 'human' to watch, 'rgb_array' for recording
        fps: Frames per second (lower = slower, easier to watch)
    """
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found at: {model_path}")
        print("Available models:")
        for model in Path('models').glob('*.pth'):
            print(f"  - {model}")
        return
    
    # Load config
    with open('hyperparameters.yml', 'r') as f:
        config = yaml.safe_load(f)[config_name]
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎮 Running on: {device}")
    
    # Create environment with rendering
    try:
        import flappy_bird_gymnasium
        env = gym.make(config['env_id'], 
                      render_mode=render_mode,
                      **config.get('env_make_params', {}))
        print(f"✅ Loaded {config['env_id']}")
    except:
        print("⚠️  FlappyBird not found, using CartPole")
        env = gym.make('CartPole-v1', render_mode=render_mode)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Load the trained model
    policy_net = DQN(state_dim, action_dim, config['fc1_nodes']).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(checkpoint['policy_net'])
    policy_net.eval()
    
    print(f"✅ Loaded model from: {model_path}")
    print(f"📊 Training stats: {checkpoint.get('step_count', 'N/A')} steps, ε: {checkpoint.get('epsilon', 'N/A'):.4f}")
    print("="*60)
    
    # Play episodes
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        print(f"\n🎮 Episode {episode + 1}/{num_episodes}")
        
        while not done:
            # Select action (greedy - no exploration)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()
            
            # Take action
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            # Render at specified FPS
            if render_mode == 'human':
                time.sleep(1.0 / fps)
        
        episode_rewards.append(episode_reward)
        print(f"   ✓ Reward: {episode_reward:.2f} | Steps: {steps}")
    
    env.close()
    
    # Summary
    print("\n" + "="*60)
    print("📈 Performance Summary:")
    print(f"   Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"   Best Reward: {max(episode_rewards):.2f}")
    print(f"   Worst Reward: {min(episode_rewards):.2f}")
    print("="*60)


def record_agent(model_path='models/flappybird_turbo.pth',
                config_name='flappybird_turbo',
                output_path='gameplay.mp4',
                num_episodes=3):
    """
    Record the agent playing and save as video
    """
    try:
        import imageio
        import flappy_bird_gymnasium
    except ImportError:
        print("❌ Install imageio to record: pip install imageio[ffmpeg]")
        return
    
    # Load everything
    with open('hyperparameters.yml', 'r') as f:
        config = yaml.safe_load(f)[config_name]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make(config['env_id'], 
                  render_mode='rgb_array',
                  **config.get('env_make_params', {}))
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy_net = DQN(state_dim, action_dim, config['fc1_nodes']).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(checkpoint['policy_net'])
    policy_net.eval()
    
    print(f"🎥 Recording {num_episodes} episodes to {output_path}")
    
    frames = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            # Render frame
            frame = env.render()
            frames.append(frame)
            
            # Select action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()
            
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    
    env.close()
    
    # Save video
    imageio.mimsave(output_path, frames, fps=30)
    print(f"✅ Video saved to {output_path}")


def compare_before_after():
    """
    Show untrained vs trained agent side by side
    """
    print("🆚 Comparing Random Agent vs Trained Agent\n")
    
    with open('hyperparameters.yml', 'r') as f:
        config = yaml.safe_load(f)['flappybird_turbo']
    
    try:
        import flappy_bird_gymnasium
        env = gym.make(config['env_id'], **config.get('env_make_params', {}))
    except:
        env = gym.make('CartPole-v1')
    
    # Test random agent
    print("🎲 Random Agent (untrained):")
    random_rewards = []
    for _ in range(5):
        state, _ = env.reset()
        reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            state, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward += r
        random_rewards.append(reward)
    print(f"   Average: {np.mean(random_rewards):.2f}")
    
    # Test trained agent
    print("\n🧠 Trained Agent:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy_net = DQN(state_dim, action_dim, config['fc1_nodes']).to(device)
    checkpoint = torch.load('models/flappybird_turbo.pth', map_location=device)
    policy_net.load_state_dict(checkpoint['policy_net'])
    policy_net.eval()
    
    trained_rewards = []
    for _ in range(5):
        state, _ = env.reset()
        reward = 0
        done = False
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = policy_net(state_tensor).argmax().item()
            state, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward += r
        trained_rewards.append(reward)
    print(f"   Average: {np.mean(trained_rewards):.2f}")
    
    env.close()
    
    improvement = (np.mean(trained_rewards) / np.mean(random_rewards) - 1) * 100
    print(f"\n📊 Improvement: {improvement:.1f}% better than random!")


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("🎮 DQN Agent Playback")
    print("="*60 + "\n")
    
    # Check what to do
    if len(sys.argv) > 1:
        if sys.argv[1] == 'record':
            record_agent()
        elif sys.argv[1] == 'compare':
            compare_before_after()
        else:
            print("Usage:")
            print("  python play_agent.py          - Watch agent play")
            print("  python play_agent.py record   - Record video")
            print("  python play_agent.py compare  - Compare vs random")
    else:
        # Default: watch the agent play
        play_agent(
            model_path='models/flappybird_turbo.pth',
            config_name='flappybird_turbo',
            num_episodes=5,
            fps=30  # Adjust this: lower = slower (easier to watch)
        )