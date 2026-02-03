import random
import numpy as np
import torch
from torch.utils.data import Dataset

from seq2seq_model import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        env,
        num_episodes=1000,
        rollout_len=15,
        max_pairs_per_episode=1000,
        min_seg_len=1,
        max_seg_len=None,
        grid_size=16,
    ):
        self.env = env
        self.grid_size = grid_size
        self.data = []
        self.total_pairs = 0
        self.episodes = 0
        self._generate_data(
            num_episodes,
            rollout_len,
            max_pairs_per_episode,
            min_seg_len,
            max_seg_len,
        )

    def _random_walk_episode(self, steps=15):
        states = [self.env.get_state()]
        actions = []
        for _ in range(steps):
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

    def _generate_data(self, num_episodes, rollout_len, max_pairs_per_episode, min_seg_len, max_seg_len):
        for _ in range(num_episodes):
            self.env.generate_random_board()
            walls = self.env.walls.copy()

            states, actions = self._random_walk_episode(steps=rollout_len)
            T = len(states)
            if T < 4:
                continue

            max_len = (T - 1) if max_seg_len is None else min(max_seg_len, T - 1)
            seg_pairs = []
            for i in range(0, T - 1):
                for j in range(i + min_seg_len, T):
                    seg_len = j - i
                    if seg_len < min_seg_len:
                        continue
                    if seg_len > max_len:
                        continue
                    seg_pairs.append((i, j))

            if not seg_pairs:
                continue

            if len(seg_pairs) > max_pairs_per_episode:
                seg_pairs = random.sample(seg_pairs, max_pairs_per_episode)

            for i, j in seg_pairs:
                s_i = states[i]
                s_j = states[j]

                target_robot = random.randint(0, 3)
                goal_flat = np.array([target_robot, s_j[target_robot, 0], s_j[target_robot, 1]], dtype=np.int64)

                action_seq = actions[i:j]
                action_in = [BOS_TOKEN] + action_seq
                action_out = action_seq + [EOS_TOKEN]

                self.data.append({
                    "walls": walls,
                    "robots_a": s_i,
                    "goal": goal_flat,
                    "action_in": action_in,
                    "action_out": action_out,
                })

            self.total_pairs += len(seg_pairs)
            self.episodes += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            "walls": torch.tensor(d["walls"], dtype=torch.long),
            "robots": torch.tensor(d["robots_a"], dtype=torch.long),
            "goal": torch.tensor(d["goal"], dtype=torch.long),
            "action_in": torch.tensor(d["action_in"], dtype=torch.long),
            "action_out": torch.tensor(d["action_out"], dtype=torch.long),
        }


def seq2seq_collate(batch):
    walls = torch.stack([b["walls"] for b in batch], dim=0)
    robots = torch.stack([b["robots"] for b in batch], dim=0)
    goal = torch.stack([b["goal"] for b in batch], dim=0)

    max_len = max(b["action_in"].numel() for b in batch)
    action_in = torch.full((len(batch), max_len), PAD_TOKEN, dtype=torch.long)
    action_out = torch.full((len(batch), max_len), PAD_TOKEN, dtype=torch.long)

    for i, b in enumerate(batch):
        length = b["action_in"].numel()
        action_in[i, :length] = b["action_in"]
        action_out[i, :length] = b["action_out"]

    return {
        "walls": walls,
        "robots": robots,
        "goal": goal,
        "action_in": action_in,
        "action_out": action_out,
    }
