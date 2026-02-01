import torch
import torch.nn as nn

class RicochetTokenizer(nn.Module):
    def __init__(self, d_model=256, grid_size=16):
        super().__init__()
        self.grid_size = grid_size

        self.wall_embedding = nn.Embedding(16, d_model)
        self.pos_embedding  = nn.Embedding(grid_size * grid_size, d_model)
        self.robot_id_embedding = nn.Embedding(4, d_model)
        self.type_embedding = nn.Embedding(3, d_model)  # 0 wall, 1 robot, 2 goal

        self.register_buffer("type_wall", torch.tensor(0, dtype=torch.long))
        self.register_buffer("type_robot", torch.tensor(1, dtype=torch.long))
        self.register_buffer("type_goal", torch.tensor(2, dtype=torch.long))

    def forward(self, walls_batch, robots_batch, goal_batch):
        """
        walls_batch:  [B,16,16] long (0..15)
        robots_batch: [B,4,2]   long (r,c)
        goal_batch:   [B,3]     long (target_robot_id, r, c)
        """
        B = walls_batch.shape[0]
        device = walls_batch.device
        HW = self.grid_size * self.grid_size

        # Walls tokens
        flat_walls = walls_batch.view(B, HW).long()
        wall_tokens = self.wall_embedding(flat_walls)
        positions = torch.arange(HW, device=device).unsqueeze(0)
        wall_tokens = wall_tokens + self.pos_embedding(positions)
        wall_tokens = wall_tokens + self.type_embedding(self.type_wall)

        # Robot tokens
        robot_indices = robots_batch[:, :, 0] * self.grid_size + robots_batch[:, :, 1]
        robot_tokens = self.pos_embedding(robot_indices.long())
        robot_ids = torch.arange(4, device=device).unsqueeze(0).expand(B, -1)
        robot_tokens = robot_tokens + self.robot_id_embedding(robot_ids)
        robot_tokens = robot_tokens + self.type_embedding(self.type_robot)

        # Goal token
        target_robot_id = goal_batch[:, 0].long()
        target_indices = goal_batch[:, 1] * self.grid_size + goal_batch[:, 2]
        goal_token = self.pos_embedding(target_indices.long()).unsqueeze(1)
        goal_token = goal_token + self.robot_id_embedding(target_robot_id).unsqueeze(1)
        goal_token = goal_token + self.type_embedding(self.type_goal)

        return torch.cat([wall_tokens, robot_tokens, goal_token], dim=1)  # [B,261,d]
