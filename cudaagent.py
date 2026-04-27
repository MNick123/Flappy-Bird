"""
Complete DQN Training Script for Flappy Bird
Optimized for RTX 5070 Ti + Ryzen 7 7800X3D

Usage:
    python train_flappybird.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import yaml
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from pathlib import Path


from cudadqn import DuelingDQN as DQN  # Using Dueling for better performance
from cudaexperience_replay import FastReplayMemory


class DQNTrainer:
    def __init__(self, config_name='flappybird_turbo', use_mixed_precision=True):
        # Load hyperparameters
        with open('hyperparameters.yml', 'r') as f:
            config = yaml.safe_load(f)[config_name]
        
        self.config = config
        self.use_mixed_precision = use_mixed_precision
        
   
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {self.device}")
        
        if torch.cuda.is_available():
            print(f" GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_per_process_memory_fraction(0.95)
        
        # Initialize environment
        import gymnasium as gym
        try:
            import flappy_bird_gymnasium
            self.env = gym.make(config['env_id'], **config.get('env_make_params', {}))
        except:
            print("FlappyBird not found, using CartPole for demo")
            self.env = gym.make('CartPole-v1')
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        print(f"📊 State dim: {state_dim}, Action dim: {action_dim}")
        
        # Initialize networks on GPU
        self.policy_net = DQN(state_dim, action_dim, config['fc1_nodes']).to(self.device)
        self.target_net = DQN(state_dim, action_dim, config['fc1_nodes']).to(self.device)
        
        print("ℹ️  Model compilation disabled (install triton to enable)")
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer - AdamW with fused operations
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(),
            lr=config['learning_rate_a'],
            fused=torch.cuda.is_available()
        )
        
        # Mixed precision training
        self.scaler = GradScaler('cuda') if use_mixed_precision else None
        if use_mixed_precision:
            print("Mixed precision (FP16) enabled")
        
        # Experience replay
        self.memory = FastReplayMemory(
            config['replay_memory_size'],
            state_dim,
            device=self.device
        )
        
        # Training parameters
        self.epsilon = config['epsilon_init']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.gamma = config['discount_factor_g']
        self.batch_size = max(config['mini_batch_size'], 256)
        self.sync_rate = config['network_sync_rate']
        self.train_steps_per_env_step = 4
        
        self.step_count = 0
        print(f" Batch size: {self.batch_size}")
        print("="*60)
    
    @torch.no_grad()
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if self.use_mixed_precision:
            with autocast('cuda'):
                q_values = self.policy_net(state_tensor)
        else:
            q_values = self.policy_net(state_tensor)
            
        return q_values.argmax().item()
    
    def train_step(self):
        """Single training step"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        if self.use_mixed_precision:
            with autocast('cuda'):
                current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                
                with torch.no_grad():
                    next_actions = self.policy_net(next_states).argmax(1)
                    next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
                    target_q = rewards + (1 - dones) * self.gamma * next_q
                
                loss = F.smooth_l1_loss(current_q, target_q)
            
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            
            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(1)
                next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
                target_q = rewards + (1 - dones) * self.gamma * next_q
            
            loss = F.smooth_l1_loss(current_q, target_q)
            
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes=2000):
        """Main training loop"""
        episode_rewards = []
        losses = []
        
        print(f"🎯 Starting training for {num_episodes} episodes...")
        print("="*60)
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_loss = []
            done = False
            
            while not done:
                # Select and perform action
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.memory.append((state, action, reward, next_state, float(done)))
                
                # Train multiple times per step
                if len(self.memory) >= self.batch_size:
                    for _ in range(self.train_steps_per_env_step):
                        step_loss = self.train_step()
                        if step_loss is not None:
                            episode_loss.append(step_loss)
                
                # Update target network
                if self.step_count % self.sync_rate == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                # Decay epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                
                state = next_state
                episode_reward += reward
                self.step_count += 1
            
            episode_rewards.append(episode_reward)
            if episode_loss:
                losses.append(np.mean(episode_loss))
            
            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_loss = np.mean(losses[-10:]) if losses else 0
                
                if torch.cuda.is_available():
                    mem_used = torch.cuda.memory_allocated() / 1e9
                    print(f"Ep {episode:4d} | Reward: {avg_reward:6.2f} | ε: {self.epsilon:.4f} | "
                          f"Loss: {avg_loss:.4f} | GPU: {mem_used:.1f}GB | Steps: {self.step_count}")
                else:
                    print(f"Ep {episode:4d} | Reward: {avg_reward:6.2f} | ε: {self.epsilon:.4f} | "
                          f"Loss: {avg_loss:.4f} | Steps: {self.step_count}")
                
                # Early stopping
                if avg_reward >= self.config['stop_on_reward']:
                    print(f"🎉 Solved in {episode} episodes!")
                    break
        
        print("="*60)
        print("Training complete!")
        return episode_rewards, losses
    
    def save(self, path='dqn_model.pth'):
        """Save model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'epsilon': self.epsilon,
        }, path)
        print(f" Model saved to {path}")
    
    def load(self, path='dqn_model.pth'):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f" Model loaded from {path}")


def plot_results(rewards, losses, save_path='training_plot.png'):
    """Plot training results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot rewards
    ax1.plot(rewards, alpha=0.3, label='Episode Reward')
    if len(rewards) > 10:
        rolling_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
        ax1.plot(range(9, len(rewards)), rolling_avg, label='10-Episode Average', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot losses
    if losses:
        ax2.plot(losses, alpha=0.5)
        if len(losses) > 10:
            rolling_avg = np.convolve(losses, np.ones(10)/10, mode='valid')
            ax2.plot(range(9, len(losses)), rolling_avg, linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Training plot saved to {save_path}")
    plt.close()


def main():
    """Main training function"""
    print("\n" + "="*60)
    print(" DQN Training - RTX 5070 Ti Optimized")
    print("="*60 + "\n")
    
    # Create trainer with flappybird_turbo settings
    trainer = DQNTrainer('flappybird_turbo', use_mixed_precision=True)
    
    # Train
    rewards, losses = trainer.train(num_episodes=2000)
    
    # Save model
    Path('models').mkdir(exist_ok=True)
    trainer.save('models/flappybird_turbo.pth')
    
    # Plot results
    Path('plots').mkdir(exist_ok=True)
    plot_results(rewards, losses, 'plots/flappybird_training.png')
    
    # Print summary
    print("\n" + "="*60)
    print("📈 Training Summary:")
    print(f"   Total Episodes: {len(rewards)}")
    print(f"   Final Avg Reward: {np.mean(rewards[-10:]):.2f}")
    print(f"   Best Reward: {max(rewards):.2f}")
    print(f"   Total Steps: {trainer.step_count:,}")
    print("="*60)


if __name__ == "__main__":
    main()
