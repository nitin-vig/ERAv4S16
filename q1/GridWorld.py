import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


class GridWorld:
    """
    A 4x4 GridWorld environment for reinforcement learning.
    
    The agent starts at the top-left corner (state 0) and tries to reach 
    the bottom-right corner (state 15). The agent can move up, down, left, 
    or right with equal probability.
    
    Rewards:
    - -1 for each move
    - 0 for reaching the terminal state (bottom-right)
    
    States are numbered 0-15:
        0   1   2   3
        4   5   6   7
        8   9  10  11
       12  13  14  15
    """
    
    def __init__(self):
        self.grid_size = 4
        self.n_states = self.grid_size * self.grid_size  # 16 states
        self.n_actions = 4  # up, down, left, right
        
        # Action mapping
        self.actions = {
            0: 'UP',
            1: 'DOWN',
            2: 'LEFT',
            3: 'RIGHT'
        }
        
        # Start and terminal states
        self.start_state = 0  # Top-left
        self.terminal_state = self.n_states-1  # Bottom-right
        
        # Current state
        self.current_state = self.start_state
        
        # Reward structure
        self.step_reward = -1
        self.terminal_reward = 0
        
        # Value function - initialize V(s) = 0 for all states
        self.V = np.zeros(self.n_states)
        
        # RL parameters
        self.gamma = 1  # Discount factor
        self.theta = 1e-4  # Convergence threshold for value iteration
        
    def state_to_position(self, state):
        """Convert state number to (row, col) position."""
        row = state // self.grid_size
        col = state % self.grid_size
        return row, col
    
    def position_to_state(self, row, col):
        """Convert (row, col) position to state number."""
        return row * self.grid_size + col
    
    def is_terminal(self, state):
        """Check if a state is terminal."""
        return state == self.terminal_state
    
    def get_next_state(self, state, action):
        """
        Get the next state given current state and action.
        
        Args:
            state: Current state (0-15)
            action: Action to take (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            
        Returns:
            next_state: The resulting state after taking the action
        """
        if self.is_terminal(state):
            return state  # Terminal state stays terminal
        
        row, col = self.state_to_position(state)
        
        # Apply action
        if action == 0:  # UP
            row = max(0, row - 1)
        elif action == 1:  # DOWN
            row = min(self.grid_size - 1, row + 1)
        elif action == 2:  # LEFT
            col = max(0, col - 1)
        elif action == 3:  # RIGHT
            col = min(self.grid_size - 1, col + 1)
        
        next_state = self.position_to_state(row, col)
        return next_state
    
    def get_reward(self, state, action, next_state):
        """
        Get the reward for transitioning from state to next_state via action.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            reward: The reward received
        """
        if self.is_terminal(next_state):
            return self.terminal_reward
        else:
            return self.step_reward
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0-3)
            
        Returns:
            next_state: The new state
            reward: The reward received
            done: Whether the episode is finished
        """
        next_state = self.get_next_state(self.current_state, action)
        reward = self.get_reward(self.current_state, action, next_state)
        done = self.is_terminal(next_state)
        
        self.current_state = next_state
        
        return next_state, reward, done
    
    def reset(self):
        """Reset the environment to the start state."""
        self.current_state = self.start_state
        return self.current_state
    
    def get_transition_probability(self, state, action, next_state):
        """
        Get the probability of transitioning to next_state from state via action.
        In this deterministic environment, probability is 1.0 for valid transitions.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Potential next state
            
        Returns:
            probability: 1.0 if this is the valid transition, 0.0 otherwise
        """
        expected_next_state = self.get_next_state(state, action)
        return 1.0 if next_state == expected_next_state else 0.0
    
    def value_iteration(self, max_iterations=1000, verbose=True):
        """
        Perform value iteration to find the optimal value function.
        
        Uses the Bellman optimality equation:
        V(s) = max_a [ R(s,a) + gamma * sum_s' P(s'|s,a) * V(s') ]
        
        Args:
            max_iterations: Maximum number of iterations
            verbose: Whether to print progress
            
        Returns:
            V: Optimal value function
            num_iterations: Number of iterations until convergence
            delta_history: History of maximum changes per iteration
        """
        delta_history = []
        
        if verbose:
            print("\n" + "="*60)
            print("Starting Value Iteration")
            print("="*60)
            print(f"Discount factor (gamma): {self.gamma}")
            print(f"Convergence threshold (theta): {self.theta}")
            print(f"Max iterations: {max_iterations}")
            print("="*60)
        
        for iteration in range(max_iterations):

            # Track maximum change in values for convergence check
            delta = 0
            
            # Create a copy of current value function
            V_new = self.V.copy()
            
            # For each state s (excluding terminal state)
            for state in range(self.n_states):
                if self.is_terminal(state):
                    # Terminal state value remains 0
                    continue
                
                # Compute new value using Bellman Equation
                action_values = []
                
                # For each action, calculate expected value
                for action in range(self.n_actions):
                    # Get next state s' (handling grid boundaries)
                    next_state = self.get_next_state(state, action)
                    
                    # Get reward for this transition
                    reward = self.get_reward(state, action, next_state)
                    
                    # Expected value update: R + gamma * V(s')
                    # In deterministic environment, P(s'|s,a) = 1.0
                    action_value = reward + self.gamma * self.V[next_state]
                    action_values.append(action_value)
                    # print(f"Iteration {iteration:4d}")
                    # print(f"state {state} action {action} next_state {next_state} action_value {action_value}")
                    
                # Update V_new(s) with maximum over all actions
                V_new[state] = max(action_values)
                self.visualize_grid(self.V,title="GridWorld - Iteration {}".format(iteration+1))
                plt.savefig("gridworld_iteration_{}.png".format(iteration+1))
                plt.close()
                # Track max change (to check if converged)
                delta = max(delta, abs(V_new[state] - self.V[state]))
            
            # Set V = V_new (update value function)
            self.V = V_new
            delta_history.append(delta)
            
            if verbose:
                print(f"Iteration {iteration:4d}: Max delta = {delta:.6f}")
            
            # If threshold reached, then stop
            if delta < self.theta:
                if verbose:
                    print("="*60)
                    print(f"Converged after {iteration + 1} iterations!")
                    print(f"Final max delta: {delta:.6f}")
                    print("="*60)
                return self.V, iteration + 1, delta_history
        
        if verbose:
            print("="*60)
            print(f"Reached maximum iterations ({max_iterations})")
            print(f"Final max delta: {delta:.6f}")
            print("="*60)
        
        return self.V, max_iterations, delta_history
    
    def extract_policy(self):
        """
        Extract the optimal policy from the current value function.
        
        For each state, choose the action that maximizes:
        R(s,a) + gamma * V(s')
        
        Returns:
            policy: Array of optimal actions for each state
        """
        policy = np.zeros(self.n_states, dtype=int)
        
        for state in range(self.n_states):
            if self.is_terminal(state):
                policy[state] = 0  # Arbitrary action for terminal state
                continue
            
            action_values = []
            for action in range(self.n_actions):
                next_state = self.get_next_state(state, action)
                reward = self.get_reward(state, action, next_state)
                action_value = reward + self.gamma * self.V[next_state]
                action_values.append(action_value)
            
            # Choose action with maximum value
            policy[state] = np.argmax(action_values)
        
        return policy
    
    def visualize_grid(self, values=None, policy=None, title="GridWorld"):
        """
        Visualize the grid world with optional values and policy.
        
        Args:
            values: Optional array of state values to display
            policy: Optional array of actions (policy) to display as arrows
            title: Title for the plot
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw grid
        for i in range(self.grid_size + 1):
            ax.axhline(i, color='black', linewidth=2)
            ax.axvline(i, color='black', linewidth=2)
        
        # Color cells based on values if provided
        if values is not None:
            values_grid = values.reshape(self.grid_size, self.grid_size)
            im = ax.imshow(values_grid, cmap='RdYlGn', alpha=0.6, 
                          extent=[0, self.grid_size, self.grid_size, 0])
            plt.colorbar(im, ax=ax, label='State Value')
        
        # Draw state numbers and values
        for state in range(self.n_states):
            row, col = self.state_to_position(state)
            
            # Highlight start and terminal states
            if state == self.start_state:
                rect = Rectangle((col, row), 1, 1, linewidth=3, 
                               edgecolor='blue', facecolor='lightblue', alpha=0.3)
                ax.add_patch(rect)
                ax.text(col + 0.5, row + 0.2, 'START', ha='center', 
                       va='center', fontsize=10, fontweight='bold', color='blue')
            elif state == self.terminal_state:
                rect = Rectangle((col, row), 1, 1, linewidth=3, 
                               edgecolor='green', facecolor='lightgreen', alpha=0.3)
                ax.add_patch(rect)
                ax.text(col + 0.5, row + 0.2, 'GOAL', ha='center', 
                       va='center', fontsize=10, fontweight='bold', color='green')
            
            # Display state number
            ax.text(col + 0.5, row + 0.5, str(state), ha='center', 
                   va='center', fontsize=12, fontweight='bold')
            
            # Display value if provided
            if values is not None:
                ax.text(col + 0.5, row + 0.8, f'{values[state]:.2f}', 
                       ha='center', va='center', fontsize=9, color='red')
        
        # Draw policy arrows if provided
        if policy is not None:
            arrow_props = dict(arrowstyle='->', lw=2, color='black')
            for state in range(self.n_states):
                if not self.is_terminal(state):
                    row, col = self.state_to_position(state)
                    action = policy[state]
                    
                    # Arrow directions
                    dx, dy = 0, 0
                    if action == 0:  # UP
                        dy = -0.3
                    elif action == 1:  # DOWN
                        dy = 0.3
                    elif action == 2:  # LEFT
                        dx = -0.3
                    elif action == 3:  # RIGHT
                        dx = 0.3
                    
                    ax.annotate('', xy=(col + 0.5 + dx, row + 0.5 + dy),
                              xytext=(col + 0.5, row + 0.5),
                              arrowprops=arrow_props)
        
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(self.grid_size, 0)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        return fig, ax
    
    def print_grid_info(self):
        """Print information about the grid world."""
        print("=" * 50)
        print("GridWorld Environment Information")
        print("=" * 50)
        print(f"Grid Size: {self.grid_size}x{self.grid_size}")
        print(f"Number of States: {self.n_states}")
        print(f"Number of Actions: {self.n_actions}")
        print(f"Actions: {self.actions}")
        print(f"Start State: {self.start_state} (Top-Left)")
        print(f"Terminal State: {self.terminal_state} (Bottom-Right)")
        print(f"Step Reward: {self.step_reward}")
        print(f"Terminal Reward: {self.terminal_reward}")
        print("=" * 50)
        print("\nState Layout:")
        print("  0   1   2   3")
        print("  4   5   6   7")
        print("  8   9  10  11")
        print(" 12  13  14  15")
        print("=" * 50)


def demo():
    """Demonstration of the GridWorld environment."""
    # Create environment
    env = GridWorld()
    env.print_grid_info()
    
    # Visualize empty grid with initial values (all zeros)
    env.visualize_grid(values=env.V, title="GridWorld - Initial State (V=0)")
    plt.savefig('gridworld_initial.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Run Value Iteration
    V_optimal, num_iterations, delta_history = env.value_iteration(max_iterations=1000, verbose=True)
    
    # Display optimal values
    print("\nOptimal Value Function:")
    print("="*60)
    for i in range(env.grid_size):
        row_values = []
        for j in range(env.grid_size):
            state = env.position_to_state(i, j)
            row_values.append(f"{V_optimal[state]:7.2f}")
        print("  ".join(row_values))
    print("="*60)
    
    # Extract optimal policy
    optimal_policy = env.extract_policy()
    
    print("\nOptimal Policy:")
    print("="*60)
    for i in range(env.grid_size):
        row_policy = []
        for j in range(env.grid_size):
            state = env.position_to_state(i, j)
            if env.is_terminal(state):
                row_policy.append("  GOAL ")
            else:
                action_name = env.actions[optimal_policy[state]]
                row_policy.append(f"{action_name:>6}")
        print("  ".join(row_policy))
    print("="*60)
    
    # Visualize optimal values and policy
    env.visualize_grid(values=V_optimal, policy=optimal_policy, 
                      title=f"GridWorld - Optimal Policy (Converged in {num_iterations} iterations)")
    plt.savefig('gridworld_optimal.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot convergence history
    plt.figure(figsize=(10, 6))
    plt.plot(delta_history, linewidth=2)
    plt.axhline(y=env.theta, color='r', linestyle='--', label=f'Threshold (θ={env.theta})')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Max Delta (Change in Value)', fontsize=12)
    plt.title('Value Iteration Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('convergence_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved to:")
    print("  - gridworld_initial.png")
    print("  - gridworld_optimal.png")
    print("  - convergence_history.png")



if __name__ == "__main__":
    demo()
