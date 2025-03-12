import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

# -----------------------------
# 1️⃣ SETUP ENVIRONMENT
# -----------------------------
env = gym.make("CartPole-v1")  # Replace with your custom robot environment if needed
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2️⃣ BUILD DQN MODEL
# -----------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No activation (raw Q-values)

# Create policy network and target network
policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Target network is not trained directly

# -----------------------------
# 3️⃣ EXPERIENCE REPLAY BUFFER
# -----------------------------
class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory = ReplayMemory()

# -----------------------------
# 4️⃣ HYPERPARAMETERS
# -----------------------------
batch_size = 64
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate of epsilon
learning_rate = 0.001
target_update = 10  # Update target network every 10 episodes
num_episodes = 500  # Training episodes

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# -----------------------------
# 5️⃣ CHOOSE ACTION (EPSILON-GREEDY)
# -----------------------------
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_size - 1)  # Explore: random action
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            return torch.argmax(policy_net(state_tensor)).item()  # Exploit: best action

# -----------------------------
# 6️⃣ TRAINING LOOP
# -----------------------------
for episode in range(num_episodes):
    state = env.reset()
    state = state[0] if isinstance(state, tuple) else state  # Handle tuple return
    total_reward = 0

    for t in range(200):
        action = select_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Train only if enough memory is collected
        if len(memory) > batch_size:
            # Sample from replay memory
            batch = memory.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(states, dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.bool).to(device)

            # Compute Q-values
            q_values = policy_net(states).gather(1, actions)

            # Compute target Q-values
            with torch.no_grad():
                max_next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + (gamma * max_next_q_values * (~dones))

            # Loss function
            loss = F.mse_loss(q_values.squeeze(), targets)

            # Optimize model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # Decay exploration rate
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update target network
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Print progress
    print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

# -----------------------------
# 7️⃣ SAVE TRAINED MODEL
# -----------------------------
torch.save(policy_net.state_dict(), "dqn_robot.pth")
print("Training completed and model saved.")

# -----------------------------
# 8️⃣ TEST THE TRAINED AI
# -----------------------------
state = env.reset()
state = state[0] if isinstance(state, tuple) else state  # Handle tuple return
total_reward = 0

for step in range(200):
    with torch.no_grad():
        action = torch.argmax(policy_net(torch.tensor(state, dtype=torch.float32).to(device))).item()
    state, reward, done, _, _ = env.step(action)
    total_reward += reward
    env.render()  # Uncomment this if you want to visualize in Gym

    if done:
        break

print(f"Total Reward in Test Run: {total_reward}")

env.close()
