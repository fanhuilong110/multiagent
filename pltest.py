
import numpy as np
import random
import plshown
import matplotlib.pyplot as plt
import numpy as np

# 生成5台行车的数据
num_cranes = 5
crane_states = []
for i in range(num_cranes):
  crane_name = f'crane_{i}'

  # 每台行车生成1-5个时间点的数据
  num_points = random.randint(1, 5)
  time = range(num_points)
  positions = [random.randint(0, 100) for _ in range(num_points)]

  crane_states.append([
    {'crane_name': crane_name, 'cur_time': t, 'cur_location': p}
    for t, p in zip(time, positions)
  ])

# 生成3台连铸机的数据
num_castings = 3
casting_states = []
for i in range(num_castings):
  casting_name = f'casting_{i}'

  # 每台连铸机生成3-6个时间点的数据
  num_points = random.randint(3, 6)
  time = range(num_points)
  volumes = [random.randint(0, 100) for _ in range(num_points)]

  casting_states.append([
    {'casting_name': casting_name, 'cur_time': t, 'volume': v}
    for t, v in zip(time, volumes)
  ])

# 任务完成率数据
completion_rate = [random.random() for _ in range(10)]
time = range(len(completion_rate))

vis = plshown.Visualization()

vis.plot_crane_trajectory(crane_states)
vis.plot_casting_state(casting_states)
vis.plot_task_completion_rate(time, completion_rate)

vis.show()