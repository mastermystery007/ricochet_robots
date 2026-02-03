# data_s2s.py
import random
import numpy as np
import torch
from torch.utils.data import Dataset

# must match environment.py
NORTH, EAST, SOUTH, WEST = 1, 2, 4, 8
DIR_LIST = [NORTH, EAST, SOUTH, WEST]

def cell_id(r, c, size=16):
    return int(r) * size + int(c)

class Seq2SeqDataset(Dataset):
    """
    Generates (walls, robots_start, goal, action_seq) from random rollouts.
    Uses COMPLETE SUBSETS: all (i,j) pairs in each episode (or capped),
    and includes all 1-step segments to strengthen last-step prediction.
    """
    def __init__(
        self,
        env,
        num_episodes=2000,
        rollout_len=15,
        grid_size=16,
        max_pairs_per_episode=None,   # None => complete subsets
        include_all_one_step=True,    # strengthens 1-step prediction
        max_seq_len=15                # for padding
    ):
        self.env = env
        self.grid_size = grid_size
        self.max_seq_len = max_seq_len
        self.data = []
        self._generate_data(
            num_episodes=num_episodes,
            rollout_len=rollout_len,
            max_pairs_per_episode=max_pairs_per_episode,
            include_all_one_step=include_all_one_step,
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

    def _generate_data(self, num_episodes, rollout_len, max_pairs_per_episode, include_all_one_step):
        for _ in range(num_episodes):
            self.env.generate_random_board()
            walls = self.env.walls.copy()
            states, actions = self._random_walk_episode(steps=rollout_len)

            T = len(states)
            if T < 3:
                continue

            # choose a target robot per episode (or you can randomize per pair)
            # per your v1: goal = (target_robot, final_cell_of_target) only.
            # We will randomize per pair for diversity.
            pairs = []

            # COMPLETE SUBSETS:
            if max_pairs_per_episode is None:
                for i in range(0, T - 1):
                    for j in range(i + 1, T):
                        pairs.append((i, j))
            else:
                for _p in range(max_pairs_per_episode):
                    i = random.randint(0, T - 2)
                    j = random.randint(i + 1, T - 1)
                    pairs.append((i, j))

            # ensure all 1-step segments are present (strong last-step learning)
            if include_all_one_step:
                for i in range(0, T - 1):
                    pairs.append((i, i + 1))

            # de-dup
            pairs = list(set(pairs))

            for (i, j) in pairs:
                s_i = states[i]
                s_j = states[j]
                actions_segment = actions[i:j]
                if len(actions_segment) == 0:
                    continue
                if len(actions_segment) > self.max_seq_len:
                    # skip or truncate; easiest is skip for v1
                    continue

                target_robot = random.randint(0, 3)
                goal_flat = np.array(
                    [target_robot, s_j[target_robot, 0], s_j[target_robot, 1]],
                    dtype=np.int64
                )

                self.data.append({
                    "walls": walls,
                    "robots": s_i,
                    "goal": goal_flat,
                    "actions": np.array(actions_segment, dtype=np.int64),
                    "seq_len": len(actions_segment),
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        walls = torch.tensor(d["walls"], dtype=torch.long)          # [16,16]
        robots = torch.tensor(d["robots"], dtype=torch.long)        # [4,2]
        goal = torch.tensor(d["goal"], dtype=torch.long)            # [3]

        # pad actions to max_seq_len with -100 (ignore_index for CE)
        acts = torch.full((self.max_seq_len,), -100, dtype=torch.long)
        seq = torch.tensor(d["actions"], dtype=torch.long)
        acts[: seq.shape[0]] = seq

        # causal mask is handled in model; we provide an attention mask (1=real token)
        attn = torch.zeros((self.max_seq_len,), dtype=torch.long)
        attn[: d["seq_len"]] = 1

        return {
            "walls": walls,
            "robots": robots,
            "goal": goal,
            "actions": acts,      # [Lmax] padded with -100
            "attn": attn,         # [Lmax] 0/1
            "seq_len": torch.tensor(d["seq_len"], dtype=torch.long),
        }
