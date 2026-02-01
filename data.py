import torch
from torch.utils.data import Dataset

class HindsightDataset(Dataset):
    def __init__(self, simulator, num_episodes=1000):
        self.sim = simulator
        self.data = []
        self._generate_data(num_episodes)

    def _generate_data(self, num_episodes):
        """The 'Stroll' Phase: Generate random paths."""
        for _ in range(num_episodes):
            self.sim.generate_random_board()
            walls = self.sim.walls.copy()
            
            # 1. Random Walk
            path = self._random_walk(steps=20) # List of robot_positions
            
            if len(path) < 5: continue

            # 2. Hindsight Relabeling
            # Pick a start (i) and a future goal (j)
            i = 0
            j = len(path) - 1
            
            start_pos = path[i]
            final_pos = path[j]
            
            # 3. Identify True Subgoal (Meeting Point)
            mid_idx = (i + j) // 2
            mid_pos = path[mid_idx]
            
            # Identify which robot moved the most or is the target (heuristic)
            # For v1, let's assume Robot 0 is the target for simplicity
            target_robot = 0
            target_cell_at_mid = mid_pos[target_robot] # (r, c)
            
            # Store tuple: (Walls, Start_Robots, Final_Goal_Robot0, True_Mid_Cell)
            self.data.append((
                walls, 
                start_pos, 
                (target_robot, final_pos[target_robot]), # The "Goal" is just Robot 0's final spot
                target_cell_at_mid # The Label for Proposer
            ))

    def _random_walk(self, steps):
        """Helper to act randomly."""
        path = [self.sim.robot_positions.copy()]
        # (Implementation of random sliding using sim.move()...)
        return path

    def __getitem__(self, idx):
        # Return tensors compatible with Tokenizer
        return self.data[idx]

    def __len__(self):
        return len(self.data)