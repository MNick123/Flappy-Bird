import torch
import numpy as np
from collections import deque
import random


class ReplayMemory:
    def __init__(self, maxlen, seed=None, device='cuda'):
        self.memory = deque([], maxlen=maxlen)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        """Sample and return tensors ready for GPU computation"""
        batch = random.sample(self.memory, sample_size)
        
        # Unpack transitions
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to GPU in one operation
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


# Alternative: Faster numpy-based replay buffer
class FastReplayMemory:
    def __init__(self, maxlen, state_dim, seed=None, device='cuda'):
        self.maxlen = maxlen
        self.state_dim = state_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Pre-allocate numpy arrays
        self.states = np.zeros((maxlen, state_dim), dtype=np.float32)
        self.actions = np.zeros(maxlen, dtype=np.int64)
        self.rewards = np.zeros(maxlen, dtype=np.float32)
        self.next_states = np.zeros((maxlen, state_dim), dtype=np.float32)
        self.dones = np.zeros(maxlen, dtype=np.float32)
        
        self.idx = 0
        self.size = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def append(self, transition):
        state, action, reward, next_state, done = transition
        
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done
        
        self.idx = (self.idx + 1) % self.maxlen
        self.size = min(self.size + 1, self.maxlen)
    
    def sample(self, sample_size):
        """Ultra-fast sampling with direct GPU transfer"""
        indices = np.random.randint(0, self.size, size=sample_size)
        
        # Direct conversion to GPU tensors
        states = torch.from_numpy(self.states[indices]).to(self.device)
        actions = torch.from_numpy(self.actions[indices]).to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device)
        dones = torch.from_numpy(self.dones[indices]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return self.size