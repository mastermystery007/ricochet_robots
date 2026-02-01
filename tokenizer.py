import torch
import torch.nn as nn
import numpy as np

class RicochetTokenizer(nn.Module):
    def __init__(self, d_model=256, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        self.d_model = d_model
        
        # --- EMBEDDINGS ---
        
        # 1. Wall Embedding: 16 possible wall configurations (0-15)
        self.wall_embedding = nn.Embedding(16, d_model)
        
        # 2. Positional Embedding: Learnable vector for each of the 256 cells
        self.pos_embedding = nn.Embedding(grid_size * grid_size, d_model)
        
        # 3. Robot ID Embedding: 4 robots
        self.robot_id_embedding = nn.Embedding(4, d_model)
        
        # 4. Goal Type Embedding: Distinguish between Goal Token and Robot Token
        self.type_embedding = nn.Embedding(3, d_model) # 0=Wall, 1=Robot, 2=Goal

    def forward(self, walls_batch, robots_batch, goal_batch):
        """
        Input:
            walls_batch:  [B, 16, 16] (Int8 bitmasks)
            robots_batch: [B, 4, 2]   (Coordinates)
            goal_batch:   [B, 3]      (Target_Robot_ID, r, c)
            
        Output:
            tokens: [B, 256 + 4 + 1, d_model]
        """
        B = walls_batch.shape[0]
        device = walls_batch.device

        # --- A. ENCODE WALLS (The Terrain) ---
        # Flatten [B, 16, 16] -> [B, 256]
        flat_walls = walls_batch.view(B, -1).long()
        
        # Embed wall logic (corners/lines)
        wall_tokens = self.wall_embedding(flat_walls) # [B, 256, d]
        
        # Add 2D Positional Information (Crucial for Transformer)
        positions = torch.arange(self.grid_size**2, device=device).unsqueeze(0)
        wall_tokens += self.pos_embedding(positions)
        
        # Add Type Identity
        wall_tokens += self.type_embedding(torch.tensor(0, device=device))

        # --- B. ENCODE ROBOTS (The Agents) ---
        # robots_batch is coords, we need to convert coords to cell indices (0-255)
        # index = r * 16 + c
        robot_indices = robots_batch[:, :, 0] * self.grid_size + robots_batch[:, :, 1]
        
        # Start with position info (Where are they?)
        robot_tokens = self.pos_embedding(robot_indices.long()) # [B, 4, d]
        
        # Add Robot Identity (Who are they?)
        robot_ids = torch.arange(4, device=device).unsqueeze(0).expand(B, -1)
        robot_tokens += self.robot_id_embedding(robot_ids)
        
        # Add Type Identity
        robot_tokens += self.type_embedding(torch.tensor(1, device=device))

        # --- C. ENCODE GOAL (The Objective) ---
        # Goal is (Robot_ID, r, c)
        target_robot_id = goal_batch[:, 0]
        target_indices = goal_batch[:, 1] * self.grid_size + goal_batch[:, 2]
        
        # Start with position info (Where to go?)
        goal_token = self.pos_embedding(target_indices.long()).unsqueeze(1) # [B, 1, d]
        
        # Add Target Robot Identity (Who needs to go there?)
        goal_token += self.robot_id_embedding(target_robot_id.long()).unsqueeze(1)
        
        # Add Type Identity
        goal_token += self.type_embedding(torch.tensor(2, device=device))

        # --- COMBINE ALL ---
        # Result: A single sequence [B, 261, d]
        # The transformer can now attend between Walls, Robots, and the Goal.
        all_tokens = torch.cat([wall_tokens, robot_tokens, goal_token], dim=1)
        
        return all_tokens