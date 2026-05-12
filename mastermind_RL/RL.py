# mastermind_dqn.py
# Requires: torch, numpy
# Run: python mastermind_dqn.py
# Or paste into a Colab cell (enable GPU for faster runs)

import random
import itertools
import math
import time
from collections import deque, namedtuple, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# Environment / utilities
# -------------------------
COLORS = list(range(6))    # 6 colors
POSITIONS = 4              # 4 positions
ALL_CODES = list(itertools.product(COLORS, repeat=POSITIONS))  # 6^4 = 1296 codes
N_ACTIONS = len(ALL_CODES)

def feedback(code, guess):
    """
    Returns (black, white) peg counts between secret code and guess.
    black = correct color & position
    white = correct color wrong position (not counting blacks)
    """
    code = list(code)
    guess = list(guess)
    black = sum(c == g for c, g in zip(code, guess))
    cc = Counter(code)
    gc = Counter(guess)
    common = sum(min(cc[col], gc[col]) for col in cc)
    white = common - black
    return black, white

def consistent_with_history(code, history):
    """
    history: list of (guess, (black,white))
    returns True if the candidate code would have produced those feedbacks
    """
    for g, fb in history:
        if feedback(code, g) != fb:
            return False
    return True

class MastermindEnv:
    """
    Simple Mastermind environment.
    State = binary vector of length 1296 indicating which codes are still possible.
    Action = index in ALL_CODES (a guess).
    Reward: -1 per guess until solved; on solve reward = 100 - steps (encourages quick solves).
    Episode ends when solved or steps >= max_steps.
    """
    def __init__(self, max_steps=10):
        self.max_steps = max_steps
        self.reset()

    def reset(self, secret=None):
        self.secret = secret if secret is not None else random.choice(ALL_CODES)
        self.history = []
        self.steps = 0
        return self._state()

    def _state(self):
        # Basic binary vector of possible codes
        possible_vec = np.zeros(N_ACTIONS, dtype=np.float32)
        possible_codes = []
        for i, code in enumerate(ALL_CODES):
            if consistent_with_history(code, self.history):
                possible_vec[i] = 1.0
                possible_codes.append(code)
        
        # Additional features
        num_possibilities = len(possible_codes)
        num_guesses = len(self.history)
        
        # Encode last guess feedback if any
        last_black, last_white = (0, 0) if not self.history else self.history[-1][1]
        
        # Features: possibilities count (normalized), steps taken (normalized), last feedback
        additional_features = np.array([
            num_possibilities / N_ACTIONS,  # Fraction of codes still possible
            num_guesses / self.max_steps,   # Fraction of steps used
            last_black / POSITIONS,         # Last black pegs (normalized)
            last_white / POSITIONS,         # Last white pegs (normalized)
        ], dtype=np.float32)
        
        # Combine binary vector with additional features
        return np.concatenate([possible_vec, additional_features])

    def step(self, action_index):
        guess = ALL_CODES[action_index]
        fb = feedback(self.secret, guess)
        
        # Count possibilities before this guess
        prev_possibilities = sum(1 for code in ALL_CODES if consistent_with_history(code, self.history))
        
        self.history.append((guess, fb))
        self.steps += 1
        
        # Count possibilities after this guess
        new_possibilities = sum(1 for code in ALL_CODES if consistent_with_history(code, self.history))
        
        done = (fb[0] == POSITIONS) or (self.steps >= self.max_steps)
        
        if fb[0] == POSITIONS:
            # Solved! Big reward, bonus for solving quickly
            reward = 100.0 - self.steps
        else:
            # Reward based on how much the guess reduced possibilities
            if prev_possibilities > 1:
                reduction_ratio = (prev_possibilities - new_possibilities) / prev_possibilities
                reward = 10.0 * reduction_ratio - 1.0  # Reward good information gain
            else:
                reward = -1.0
                
        return self._state(), reward, done, {"feedback": fb, "prev_poss": prev_possibilities, "new_poss": new_possibilities}

# -------------------------
# DQN components (PyTorch)
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),  # LayerNorm works with batch size 1
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, output_dim)
        )
    def forward(self, x):
        return self.net(x)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buf = deque(maxlen=capacity)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buf)

# -------------------------
# Hyperparameters
# -------------------------
input_dim = N_ACTIONS + 4  # Binary vector + 4 additional features
output_dim = N_ACTIONS
hidden = 512

policy_net = DQN(input_dim, output_dim, hidden=hidden).to(device)
target_net = DQN(input_dim, output_dim, hidden=hidden).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)  # Increased learning rate
loss_fn = nn.MSELoss()
buffer = ReplayBuffer(capacity=15000)  # Larger buffer

batch_size = 128  # Larger batch size
gamma = 0.95  # Slightly lower discount for faster learning
sync_every = 100  # More frequent target network updates
eps_start = 1.0
eps_end = 0.01  # Lower final epsilon for more exploitation
eps_decay = 8000   # Slower decay to explore more initially

def select_action(state_vec, steps_done):
    eps = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    
    # Get valid actions (only codes consistent with history)
    valid_actions = np.where(state_vec[:N_ACTIONS] > 0)[0]  # Only check binary part
    
    if len(valid_actions) == 0:  # Safety check
        valid_actions = np.arange(N_ACTIONS)
    
    if random.random() < eps:
        # Random exploration from valid actions only
        return random.choice(valid_actions)
    
    with torch.no_grad():
        s = torch.from_numpy(state_vec).unsqueeze(0).to(device)
        q = policy_net(s)
        
        # Mask invalid actions by setting their Q-values to negative infinity
        q_masked = q.clone()
        invalid_mask = torch.ones(N_ACTIONS, dtype=torch.bool)
        invalid_mask[valid_actions] = False
        q_masked[0, invalid_mask] = float('-inf')
        
        return int(q_masked.argmax().cpu().numpy())

# -------------------------
# Training loop
# -------------------------
def train(num_episodes=10000, max_steps_per_episode=10, print_every=50):
    env = MastermindEnv(max_steps=max_steps_per_episode)
    steps_done = 0
    rewards_history = []
    start_time = time.time()
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = select_action(state, steps_done)
            next_state, reward, done, info = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps_done += 1

            # learning step
            if len(buffer) >= batch_size:
                trans = buffer.sample(batch_size)
                state_b = torch.tensor(np.array(trans.state), dtype=torch.float32).to(device)
                action_b = torch.tensor(trans.action, dtype=torch.long).unsqueeze(1).to(device)
                reward_b = torch.tensor(trans.reward, dtype=torch.float32).unsqueeze(1).to(device)
                next_state_b = torch.tensor(np.array(trans.next_state), dtype=torch.float32).to(device)
                done_b = torch.tensor(trans.done, dtype=torch.float32).unsqueeze(1).to(device)

                q_values = policy_net(state_b).gather(1, action_b)
                with torch.no_grad():
                    next_q = target_net(next_state_b).max(1)[0].unsqueeze(1)
                    target_q = reward_b + (1.0 - done_b) * gamma * next_q
                loss = loss_fn(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps_done % sync_every == 0 and steps_done > 0:
                target_net.load_state_dict(policy_net.state_dict())

        rewards_history.append(total_reward)
        if ep % print_every == 0:
            avg = np.mean(rewards_history[-print_every:]) if len(rewards_history) >= print_every else np.mean(rewards_history)
            elapsed = time.time() - start_time
            print(f"Episode {ep}/{num_episodes}  AvgReward({print_every})={avg:.2f}  eps~{eps_end + (eps_start-eps_end)*math.exp(-1.0*steps_done/eps_decay):.3f}  elapsed={int(elapsed)}s")
    
    print("Training finished.")
    
    # Save the trained model
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparameters': {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden': hidden,
            'lr': optimizer.param_groups[0]['lr'],
            'gamma': gamma,
            'eps_start': eps_start,
            'eps_end': eps_end,
            'eps_decay': eps_decay
        },
        'training_info': {
            'episodes': num_episodes,
            'final_avg_reward': np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history),
            'steps_done': steps_done
        }
    }, 'mastermind_dqn_model.pth')
    print("Model saved as 'mastermind_dqn_model.pth'")
    
    return policy_net, env

def load_trained_model(model_path='mastermind_dqn_model.pth'):
    """
    Load a previously trained model from file.
    Returns the loaded policy network and a new environment.
    """
    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Train a model first.")
    
    # Load the saved data with weights_only=False for compatibility
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract hyperparameters
    hyperparams = checkpoint['hyperparameters']
    
    # Create the model with saved hyperparameters
    policy_net = DQN(
        input_dim=hyperparams['input_dim'],
        output_dim=hyperparams['output_dim'],
        hidden=hyperparams['hidden']
    ).to(device)
    
    # Load the trained weights
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    policy_net.eval()  # Set to evaluation mode
    
    # Create environment
    env = MastermindEnv(max_steps=10)
    
    # Print model info
    training_info = checkpoint['training_info']
    print(f"Loaded model trained for {training_info['episodes']} episodes")
    print(f"Final training performance: {training_info['final_avg_reward']:.2f} avg reward")
    
    return policy_net, env

def play_with_trained_model(model_path='mastermind_dqn_model.pth', n_games=10):
    """
    Load a trained model and play multiple games, showing the gameplay.
    """
    policy_net, env = load_trained_model(model_path)
    
    solved = 0
    total_steps = 0
    
    for game_num in range(1, n_games + 1):
        print(f"\n=== Game {game_num} ===")
        s = env.reset()
        print(f"Secret code: {env.secret}")
        
        done = False
        steps = 0
        while not done and steps < env.max_steps:
            with torch.no_grad():
                ss = torch.from_numpy(s).unsqueeze(0).to(device)
                q = policy_net(ss)
                
                # Apply action masking
                valid_actions = np.where(s[:N_ACTIONS] > 0)[0]
                if len(valid_actions) > 0:
                    q_masked = q.clone()
                    invalid_mask = torch.ones(N_ACTIONS, dtype=torch.bool)
                    invalid_mask[valid_actions] = False
                    q_masked[0, invalid_mask] = float('-inf')
                    action = int(q_masked.argmax().cpu().numpy())
                else:
                    action = 0
                    
            guess = ALL_CODES[action]
            s, reward, done, info = env.step(action)
            steps += 1
            
            black, white = info['feedback']
            print(f"  Step {steps}: Guess {guess} -> {black} black, {white} white")
            
            if black == POSITIONS:
                print(f"  🎉 Solved in {steps} steps!")
                solved += 1
                total_steps += steps
                break
        else:
            print(f"  ❌ Failed to solve in {env.max_steps} steps")
    
    avg_steps = total_steps / solved if solved > 0 else float('inf')
    print(f"\n=== Summary ===")
    print(f"Solved: {solved}/{n_games} ({100*solved/n_games:.1f}%)")
    print(f"Average steps (for solved): {avg_steps:.2f}")

# -------------------------
# Evaluation
# -------------------------
def evaluate(policy_net, env, n_games=200):
    solved = 0
    total_steps = 0
    
    # Track the last game for display
    last_game_history = []
    last_game_secret = None
    
    for game_idx in range(n_games):
        s = env.reset()
        if game_idx == n_games - 1:  # Last game
            last_game_secret = env.secret
            last_game_history = []
            
        done = False
        steps = 0
        while not done and steps < env.max_steps:
            with torch.no_grad():
                ss = torch.from_numpy(s).unsqueeze(0).to(device)
                q = policy_net(ss)
                
                # Apply action masking
                valid_actions = np.where(s[:N_ACTIONS] > 0)[0]  # Only check binary part of state
                if len(valid_actions) > 0:
                    q_masked = q.clone()
                    invalid_mask = torch.ones(N_ACTIONS, dtype=torch.bool)
                    invalid_mask[valid_actions] = False
                    q_masked[0, invalid_mask] = float('-inf')
                    action = int(q_masked.argmax().cpu().numpy())
                else:
                    action = 0  # Fallback
                    
            s, reward, done, info = env.step(action)
            steps += 1
            
            # Record last game details
            if game_idx == n_games - 1:
                guess = ALL_CODES[action]
                feedback_info = info['feedback']
                last_game_history.append((guess, feedback_info))
                
        if info['feedback'][0] == POSITIONS:
            solved += 1
            total_steps += steps
            
    avg_steps = total_steps / solved if solved > 0 else float('inf')
    print(f"Evaluation: solved {solved}/{n_games}, avg steps (for solved) = {avg_steps:.2f}")
    
    # Display the last game
    print(f"\n--- Last Game Details ---")
    print(f"Secret code: {last_game_secret}")
    print(f"Guesses and feedback:")
    for step, (guess, feedback) in enumerate(last_game_history, 1):
        black, white = feedback
        print(f"  Step {step}: Guess {guess} -> {black} black, {white} white")
        if black == POSITIONS:
            print(f"  🎉 Solved in {step} steps!")
            break
    else:
        print(f"  ❌ Failed to solve in {env.max_steps} steps")
    
    return solved, avg_steps

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    import os
    model_file = 'mastermind_dqn_model.pth'
    
    # Check if a trained model already exists
    if os.path.exists(model_file):
        print(f"Found existing model: {model_file}")
        print("Choose an option:")
        print("1. Load existing model and play games")
        print("2. Load existing model and evaluate")
        print("3. Train a new model (overwrites existing)")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            print("\n=== Playing with trained model ===")
            play_with_trained_model(model_file, n_games=5)
        elif choice == "2":
            print("\n=== Evaluating trained model ===")
            policy, env = load_trained_model(model_file)
            evaluate(policy, env, n_games=200)
        elif choice == "3":
            print("\n=== Training new model ===")
            policy, env = train(num_episodes=5000, max_steps_per_episode=10, print_every=200)
            evaluate(policy, env, n_games=200)
        else:
            print("Invalid choice. Loading existing model for evaluation.")
            policy, env = load_trained_model(model_file)
            evaluate(policy, env, n_games=200)
    else:
        print("No existing model found. Training new model...")
        policy, env = train(num_episodes=5000, max_steps_per_episode=10, print_every=200)
        evaluate(policy, env, n_games=200)
