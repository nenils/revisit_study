# mastermind_interactive.py
# Interactive Mastermind game where the user plays and can ask for AI suggestions
# Requires: torch, numpy, and the trained model file

import random
import itertools
import numpy as np
import torch
import torch.nn as nn
from RL import DQN, load_trained_model, ALL_CODES, COLORS, POSITIONS, N_ACTIONS, feedback, consistent_with_history

class InteractiveMastermind:
    def __init__(self, use_ai_model=True):
        self.secret = None
        self.history = []
        self.max_steps = 10
        self.steps = 0
        self.ai_model = None
        self.env_for_ai = None
        
        if use_ai_model:
            try:
                self.ai_model, self.env_for_ai = load_trained_model()
                print("✅ AI assistant loaded successfully!")
            except FileNotFoundError:
                print("⚠️  No trained AI model found. Train the model first using RL.py")
                print("   AI suggestions will not be available.")
    
    def start_new_game(self):
        """Start a new game with a random secret code."""
        self.secret = random.choice(ALL_CODES)
        self.history = []
        self.steps = 0
        print("\n" + "="*50)
        print("🎯 NEW MASTERMIND GAME")
        print("="*50)
        print(f"I've chosen a secret code with {POSITIONS} positions.")
        print(f"Each position has a color from 0 to {len(COLORS)-1}.")
        print(f"You have {self.max_steps} guesses to crack the code!")
        print("\nFeedback legend:")
        print("  • Black pegs = correct color in correct position")
        print("  • White pegs = correct color in wrong position")
        print("\nCommands:")
        print("  • Enter guess as: 1 2 3 4")
        print("  • Type 'hint' for AI suggestion")
        print("  • Type 'quit' to exit")
        print("  • Type 'restart' for new game")
        print("-" * 50)
    
    def parse_guess(self, input_str):
        """Parse user input into a valid guess tuple."""
        try:
            parts = input_str.strip().split()
            if len(parts) != POSITIONS:
                return None, f"Please enter exactly {POSITIONS} numbers separated by spaces."
            
            guess = []
            for part in parts:
                num = int(part)
                if num not in COLORS:
                    return None, f"Each number must be between 0 and {len(COLORS)-1}."
                guess.append(num)
            
            return tuple(guess), None
        except ValueError:
            return None, "Please enter only numbers separated by spaces."
    
    def get_ai_suggestion(self):
        """Get a suggestion from the trained AI model."""
        if not self.ai_model:
            return None, "AI model not available."
        
        try:
            # Create the current state representation
            state = self._get_state_vector()
            
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                q_values = self.ai_model(state_tensor)
                
                # Apply action masking (only valid moves)
                valid_actions = np.where(state[:N_ACTIONS] > 0)[0]
                if len(valid_actions) == 0:
                    return None, "No valid moves available (this shouldn't happen)."
                
                q_masked = q_values.clone()
                invalid_mask = torch.ones(N_ACTIONS, dtype=torch.bool)
                invalid_mask[valid_actions] = False
                q_masked[0, invalid_mask] = float('-inf')
                
                best_action = int(q_masked.argmax().cpu().numpy())
                suggested_guess = ALL_CODES[best_action]
                
                return suggested_guess, None
        except Exception as e:
            return None, f"Error getting AI suggestion: {str(e)}"
    
    def _get_state_vector(self):
        """Create state vector like the one used in training."""
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
    
    def make_guess(self, guess):
        """Process a guess and return feedback."""
        if guess in [g for g, _ in self.history]:
            return None, "You already tried that guess!"
        
        fb = feedback(self.secret, guess)
        self.history.append((guess, fb))
        self.steps += 1
        
        black, white = fb
        
        # Show feedback
        print(f"\n📝 Guess {self.steps}: {guess}")
        print(f"📊 Feedback: {black} black, {white} white")
        
        if black == POSITIONS:
            print(f"\n🎉 CONGRATULATIONS! You cracked the code in {self.steps} steps!")
            print(f"🔓 The secret was: {self.secret}")
            return True, "won"
        elif self.steps >= self.max_steps:
            print(f"\n💥 Game Over! You used all {self.max_steps} guesses.")
            print(f"🔒 The secret was: {self.secret}")
            return True, "lost"
        else:
            # Show remaining possibilities
            remaining = sum(1 for code in ALL_CODES if consistent_with_history(code, self.history))
            print(f"🔍 Possibilities remaining: {remaining}/{N_ACTIONS}")
            return False, "continue"
    
    def play(self):
        """Main game loop."""
        print("🎮 Welcome to Interactive Mastermind!")
        
        while True:
            self.start_new_game()
            
            while True:
                print(f"\n--- Turn {self.steps + 1}/{self.max_steps} ---")
                user_input = input("Enter your guess (or 'hint'/'quit'/'restart'): ").strip().lower()
                
                if user_input == 'quit':
                    print("👋 Thanks for playing!")
                    return
                elif user_input == 'restart':
                    break
                elif user_input == 'hint':
                    suggestion, error = self.get_ai_suggestion()
                    if error:
                        print(f"❌ {error}")
                    else:
                        print(f"🤖 AI suggests: {suggestion}")
                        print("   (You can type this guess or choose your own)")
                    continue
                else:
                    guess, error = self.parse_guess(user_input)
                    if error:
                        print(f"❌ {error}")
                        continue
                    
                    game_over, result = self.make_guess(guess)
                    if game_over:
                        break
            
            # Ask if they want to play again
            while True:
                play_again = input("\n🎯 Play another game? (y/n): ").strip().lower()
                if play_again in ['y', 'yes']:
                    break
                elif play_again in ['n', 'no']:
                    print("👋 Thanks for playing!")
                    return
                else:
                    print("Please enter 'y' or 'n'")

def demo_ai_vs_secret(secret_code=None):
    """Demo function to watch the AI solve a specific code."""
    game = InteractiveMastermind(use_ai_model=True)
    
    if not game.ai_model:
        print("❌ No AI model available for demo.")
        return
    
    if secret_code is None:
        secret_code = random.choice(ALL_CODES)
    
    game.secret = secret_code
    print(f"\n🤖 AI Demo: Solving secret code {secret_code}")
    print("-" * 40)
    
    while game.steps < game.max_steps:
        suggestion, error = game.get_ai_suggestion()
        if error:
            print(f"❌ Error: {error}")
            break
        
        print(f"\n🤖 AI chooses: {suggestion}")
        game_over, result = game.make_guess(suggestion)
        
        if game_over:
            if result == "won":
                print(f"🎉 AI solved it in {game.steps} steps!")
            else:
                print(f"💥 AI failed to solve it in {game.max_steps} steps.")
            break

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Play interactive game")
    print("2. Watch AI solve a random code")
    print("3. Watch AI solve a specific code")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        game = InteractiveMastermind()
        game.play()
    elif choice == "2":
        demo_ai_vs_secret()
    elif choice == "3":
        try:
            code_input = input("Enter secret code (e.g., '1 2 3 4'): ").strip()
            parts = code_input.split()
            if len(parts) != POSITIONS:
                print(f"❌ Please enter exactly {POSITIONS} numbers.")
            else:
                secret = tuple(int(x) for x in parts)
                if all(x in COLORS for x in secret):
                    demo_ai_vs_secret(secret)
                else:
                    print(f"❌ Each number must be between 0 and {len(COLORS)-1}.")
        except ValueError:
            print("❌ Invalid input. Please enter numbers only.")
    else:
        print("❌ Invalid choice.")