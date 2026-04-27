import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()
        
        # Deeper network for better feature learning
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Batch normalization for training stability (optional)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Initialize weights for better convergence
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Kaiming initialization for ReLU activation"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Standard forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def forward_with_bn(self, x):
        """Alternative forward with batch normalization for more stable training"""
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        return self.fc3(x)


# Alternative: Dueling DQN architecture for better value estimation
class DuelingDQN(nn.Module):
    """Dueling architecture separates value and advantage streams"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DuelingDQN, self).__init__()
        
        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)
        
        # Combine value and advantage: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# Noisy DQN for better exploration without epsilon-greedy
class NoisyLinear(nn.Module):
    """Noisy linear layer for NoisyNet"""
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / self.in_features ** 0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / self.out_features ** 0.5)
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class NoisyDQN(nn.Module):
    """DQN with noisy layers for parameter space exploration"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(NoisyDQN, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.noisy1 = NoisyLinear(hidden_dim, hidden_dim)
        self.noisy2 = NoisyLinear(hidden_dim, action_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.noisy1(x))
        return self.noisy2(x)
    
    def reset_noise(self):
        """Reset noise for exploration"""
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


if __name__ == "__main__":
    # Auto-detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    state_dim = 12
    action_dim = 2
    batch_size = 512
    
    # Test standard DQN
    print("\n=== Standard DQN ===")
    net = DQN(state_dim, action_dim, hidden_dim=256).to(device)
    state = torch.randn((batch_size, state_dim), device=device)
    output = net(state)
    print(f"Input shape: {state.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # Test Dueling DQN
    print("\n=== Dueling DQN ===")
    dueling_net = DuelingDQN(state_dim, action_dim, hidden_dim=256).to(device)
    output = dueling_net(state)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in dueling_net.parameters()):,}")
    
    # Test Noisy DQN
    print("\n=== Noisy DQN ===")
    noisy_net = NoisyDQN(state_dim, action_dim, hidden_dim=256).to(device)
    output = noisy_net(state)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in noisy_net.parameters()):,}")
    
    # Benchmark speed
    if torch.cuda.is_available():
        print("\n=== Speed Benchmark (1000 forward passes) ===")
        import time
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(1000):
            _ = net(state)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"Standard DQN: {elapsed:.3f}s ({1000/elapsed:.0f} fps)")
        print(f"Throughput: {batch_size * 1000 / elapsed:.0f} samples/sec")