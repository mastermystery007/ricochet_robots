import numpy as np
import random

# --- Constants ---
NORTH, EAST, SOUTH, WEST = 1, 2, 4, 8
DIRS = {NORTH: (-1, 0), EAST: (0, 1), SOUTH: (1, 0), WEST: (0, -1)}

class RicochetEnv:
    def __init__(self, size=16, num_robots=4):
        self.size = size
        self.num_robots = num_robots
        self.walls = np.zeros((size, size), dtype=np.int8)
        self.robot_positions = np.zeros((num_robots, 2), dtype=np.int16)

    def generate_random_board(self, density=0.12):
        self.walls.fill(0)

        # 1) Outer boundary
        for i in range(self.size):
            self.walls[0, i] |= NORTH
            self.walls[self.size - 1, i] |= SOUTH
            self.walls[i, 0] |= WEST
            self.walls[i, self.size - 1] |= EAST

        # 2) Internal random walls
        num_walls = int(self.size * self.size * density)
        for _ in range(num_walls):
            r = random.randint(0, self.size - 1)
            c = random.randint(0, self.size - 1)
            direction = random.choice([NORTH, EAST, SOUTH, WEST])
            self._add_wall(r, c, direction)

        # 3) Place robots (not boxed, no overlap)
        used = set()
        for i in range(self.num_robots):
            while True:
                r = random.randint(0, self.size - 1)
                c = random.randint(0, self.size - 1)
                if (r, c) in used:
                    continue
                if self.walls[r, c] == 15:
                    continue
                self.robot_positions[i] = (r, c)
                used.add((r, c))
                break

    def _add_wall(self, r, c, direction):
        if not (0 <= r < self.size and 0 <= c < self.size):
            return

        self.walls[r, c] |= direction
        if direction == NORTH and r > 0:
            self.walls[r - 1, c] |= SOUTH
        elif direction == SOUTH and r < self.size - 1:
            self.walls[r + 1, c] |= NORTH
        elif direction == EAST and c < self.size - 1:
            self.walls[r, c + 1] |= WEST
        elif direction == WEST and c > 0:
            self.walls[r, c - 1] |= EAST

    def _blocked_by_wall(self, r, c, direction):
        return (self.walls[r, c] & direction) != 0

    def _slide(self, robot_id, direction):
        """
        Correct physics:
        - At each step, first check whether a wall blocks leaving the current cell.
        - Then check the next cell for bounds + robot collision.
        """
        dr, dc = DIRS[direction]
        r, c = int(self.robot_positions[robot_id, 0]), int(self.robot_positions[robot_id, 1])

        occ = {(int(rr), int(cc)) for rr, cc in self.robot_positions}
        occ.remove((r, c))

        while True:
            # 1) wall blocks leaving current cell
            if self._blocked_by_wall(r, c, direction):
                break

            nr, nc = r + dr, c + dc

            # 2) bounds (outer walls should already prevent, but keep safe)
            if not (0 <= nr < self.size and 0 <= nc < self.size):
                break

            # 3) robot collision
            if (nr, nc) in occ:
                break

            # 4) move
            r, c = nr, nc

        return r, c

    def step(self, action):
        """
        action: int 0..15
        robot_id = action // 4
        dir_idx  = action % 4 (0=N,1=E,2=S,3=W)
        """
        robot_id = action // 4
        dir_idx = action % 4
        direction = [NORTH, EAST, SOUTH, WEST][dir_idx]

        old_r, old_c = int(self.robot_positions[robot_id, 0]), int(self.robot_positions[robot_id, 1])
        new_r, new_c = self._slide(robot_id, direction)
        self.robot_positions[robot_id] = (new_r, new_c)

        return not (old_r == new_r and old_c == new_c)

    def get_state(self):
        return self.robot_positions.copy()

    def set_state(self, robot_positions):
        self.robot_positions[:] = robot_positions
