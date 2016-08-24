import numpy as np 
from random import randint
import time

# 2 X 3 grid where rightmost block on the first row is the goal state. 
grid_world = np.zeros((2, 3))
print grid_world

# Initializinf a 6 X 4 matrix to keep track and every state and its corresponding action, Q(s, a) values 
q_matrix = np.zeros((6, 4))
current_state = 0
eta = 0.9
alpha = 0.5

def next_state_(current_state, current_action):
  # 0 -> left, 1 -> down , 2 -> right, 3 -> up 
  if current_state == 0:
    if current_action == 2:
      return 1
    elif current_action == 1:
      return 3
    else:
      return -1
  elif current_state == 1:
    if current_action == 0:
      return 0
    elif current_action == 1:
      return 4
    elif current_action == 2:
      return 3
    else:
      return -1
  elif current_state == 2:
    return 2
  elif current_state == 3:
    if current_action == 2:
      return 4
    elif current_action == 3:
      return 0
    else:
      return -1
  elif current_state == 4:
    if current_action == 0:
      return 3
    elif current_action == 2:
      return 5
    elif current_action == 3:
      return 1
    else:
      return -1
  elif current_state == 5:
    if current_action == 0:
      return 4
    elif current_action == 3:
      return 2
    else:
      return -1 

while True:
  current_state = randint(0, 5)
  while current_state != 2:
    current_action = randint(0, 3)
    reward = 0
    next_state = next_state_(current_state, current_action)
    if next_state == 2:
      reward = 100
    elif next_state == -1:
      continue
    max_entry = max(q_matrix[next_state, :])
    q_matrix[current_state, current_action] = q_matrix[current_state, current_action] + alpha*(reward + eta * max_entry - q_matrix[current_state, current_action])
    current_state = next_state
    print(q_matrix)
