# Given a series of navigation steps, the task is to determine whether or not an agent would end up back at the starting point. The asserts should never fire.

class Coordinates:
    def __init__(self):
        self.x = 0
        self.y = 0

    def make_steps(self, num_steps, direction):
        self.x += num_steps * direction[0]
        self.y += num_steps * direction[1]

    def is_on_starting_point(self):
        return self.x == 0 and self.y == 0

# Example 1
# Setting up initial state. The agent is on a map.
coords = Coordinates()
initial_direction = (0, 1)

# Take 2 steps. Take 3 steps. Turn around. Take 4 steps. Take 1 step. Turn left.
direction = initial_direction
coords.make_steps(2, direction)
coords.make_steps(3, direction)
direction = (-direction[0], -direction[1])
coords.make_steps(4, direction)
coords.make_steps(1, direction)
direction = (-direction[1], direction[0])
assert coords.is_on_starting_point() == True

# Example 2
# Setting up initial state. The agent is on a map.
coords = Coordinates()
initial_direction = (0, 1)

# Always face forward. Take 2 steps backward. Take 2 steps right. Take 2 steps forward.
direction = initial_direction
coords.make_steps(2, (-direction[0], -direction[1]))
coords.make_steps(2, (direction[1], -direction[0]))
coords.make_steps(2, direction)
assert coords.is_on_starting_point() == False

# Example 3
# Setting up initial state. The agent is on a map.
coords = Coordinates()
initial_direction = (0, 1)

# 
