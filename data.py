import random
import numpy as np
import torch
from torch.utils.data import Dataset

DMAX = 20  # bin 20 means >=20

def cell_id(r, c, size=16):
    return int(r) * size + int(c)

def action_robot_id(a):
    return a // 4

class HindsightDataset(Dataset):
    def __init__(self, env, num_episodes=1000, rollout_len=15, pairs_per_episode=6, grid_size=16):
        self.env = env
        self.grid_size = grid_size
        self.data = []
        self._generate_data(num_episodes, rollout_len, pairs_per_episode)

    def _random_walk_episode(self, steps=15):
        states = [self.env.get_state()]
        actions = []
        for _ in range(steps):
            # sample until a move happens (or give up)
            for _try in range(20):
                a = random.randint(0, 15)
                moved = self.env.step(a)
                if moved:
                    actions.append(a)
                    states.append(self.env.get_state())
                    break
            else:
                break
        return states, actions

    def _choose_blocker(self, target_robot, actions_segment):
        """
        Pseudo-label: pick the non-target robot moved most in this segment.
        Falls back to a deterministic non-target if segment is empty.
        """
        if not actions_segment:
            return (target_robot + 1) % 4

        counts = [0, 0, 0, 0]
        for a in actions_segment:
            counts[action_robot_id(a)] += 1

        best_r = None
        best_c = -1
        for r in range(4):
            if r == target_robot:
                continue
            if counts[r] > best_c:
                best_c = counts[r]
                best_r = r

        return best_r if best_r is not None else (target_robot + 1) % 4

    def _generate_data(self, num_episodes, rollout_len, pairs_per_episode):
        for _ in range(num_episodes):
            self.env.generate_random_board()
            walls = self.env.walls.copy()

            states, actions = self._random_walk_episode(steps=rollout_len)
            T = len(states)
            if T < 4:
                continue

            # sample multiple (i,j) pairs per episode
            for _p in range(pairs_per_episode):
                i = random.randint(0, T - 2)
                j = random.randint(i + 1, T - 1)
                k = (i + j) // 2

                s_i = states[i]
                s_j = states[j]
                s_k = states[k]

                # choose target robot id (goal = its final cell at s_j)
                target_robot = random.randint(0, 3)
                goal_flat = np.array([target_robot, s_j[target_robot, 0], s_j[target_robot, 1]], dtype=np.int64)

                # proposer labels from midpoint state s_k
                mid_cell = cell_id(s_k[target_robot, 0], s_k[target_robot, 1], self.grid_size)

                actions_segment = actions[i:j]  # corresponds to moves from s_i to s_j
                blocker_id = self._choose_blocker(target_robot, actions_segment)
                block_cell = cell_id(s_k[blocker_id, 0], s_k[blocker_id, 1], self.grid_size)

                dist = j - i
                dist_bin = dist if dist < DMAX else DMAX  # 20 means >=20

                self.data.append({
                    "walls": walls,
                    "robots_a": s_i,
                    "robots_b": s_j,
                    "goal": goal_flat,
                    "mid_cell": mid_cell,
                    "block_id": blocker_id,
                    "block_cell": block_cell,
                    "dist": dist_bin,
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            "walls": torch.tensor(d["walls"], dtype=torch.long),
            "robots": torch.tensor(d["robots_a"], dtype=torch.long),
            "final_robots": torch.tensor(d["robots_b"], dtype=torch.long),
            "goal": torch.tensor(d["goal"], dtype=torch.long),
            "mid_cell": torch.tensor(d["mid_cell"], dtype=torch.long),
            "block_id": torch.tensor(d["block_id"], dtype=torch.long),
            "block_cell": torch.tensor(d["block_cell"], dtype=torch.long),
            "dist": torch.tensor(d["dist"], dtype=torch.long),
        }
