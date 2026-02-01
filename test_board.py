from environment import RicochetEnv

# 1. Initialize
env = RicochetEnv(size=16)

# 2. Generate a random board
env.generate_random_board(density=0.12)

# 3. Visualize it
print("Rendering board... Close the window to exit.")
env.render()