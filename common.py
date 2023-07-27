import numpy as np  # 引入numpy库，用于处理数值计算
import pandas as pd
from based import *
import os
import random


# 定义奖励参数
reward_params = {
    'select_task': 10,
    'no_task': -10,
    'pick_up_ladle': 20,
    'wrong_action': -20,
    'reach_start': 20,
    'reach_target': 30,
    'conflict': -30,
    'put_down_ladle': 40,
    'late_task': -50,
    'casting_interruption': -100,
    'casting_time_threshold': 60
}


def convert_date_columns(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S")  # 例如，如果你的日期时间的格式为"2023-06-17 12:34:56"

                df[col] = (df[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')  # 转换成Unix时间戳，单位是秒
            except ValueError:
                pass
    return df

# 获取所有过跨线位置
def get_cross_lines(df):
    cross_line_markers = ['R5', 'L5', 'LD5', 'RH5', 'KR5', 'LA5', 'LD5', 'R5', 'GF5']  # 过跨线的标记

    cross_lines_df = df[df['loc_location'].apply(lambda x: any(marker in x for marker in cross_line_markers))]  # 筛选出含有过跨线标记的数据
    return cross_lines_df['pos_x'].unique().tolist()  # 返回所有过跨线的位置列表

# 将所有任务按开始时间排序有早到晚排序，20个一组
def group_tasks(job_data, group_size):
    # 按照最早开始时间排序
    job_data = job_data.sort_values(by='earliest_up_time')

    # 创建一个新的列"group_id"，表示每个任务所在的组
    job_data['group_id'] = np.arange(len(job_data)) // group_size

    # 根据"group_id"进行分组
    task_groups = [group for _, group in job_data.groupby('group_id')]

    return task_groups



# 将所有时间窗有重叠的任务放在一起
def group_tasks_by_overlap(job_data):
    # 按照最早开始时间排序
    job_data = job_data.sort_values(by='earliest_up_time')

    # 初始化任务组列表
    task_groups = []
    # 遍历排序后的每一个任务
    for idx, row in job_data.iterrows():
        if task_groups:
            # 如果当前任务的最早开始时间早于或等于上一个任务组的最晚结束时间，说明有重叠
            if row['earliest_up_time'] <= task_groups[-1][-1]['latest_down_time']:
                # 将当前任务添加到上一个任务组
                task_groups[-1].append(row)
            else:
                # 否则，开始一个新的任务组
                task_groups.append([row])
        else:
            # 如果任务组列表为空，直接添加一个新的任务组
            task_groups.append([row])

    # 转换每个任务组为DataFrame
    task_groups = [pd.DataFrame(group) for group in task_groups]

    return task_groups


def load_data(data_path):  # 定义一个加载数据的函数
    """
         加载数据。

         Parameters
         ----------
         data_path : str
             数据的路径，该路径下应包含所有必要的数据文件。

         Raises
         ------
         FileNotFoundError
             如果在给定的路径下找不到必要的数据文件，则抛出此异常。
     """
    # 确保文件路径存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find the data path: {data_path}")

    # 各个设备的位置----------然后处理缺失值，最后将列名更改
    location_df = pd.read_excel(data_path + r'/MOD_CRANE_LOCATION.xlsx')
    location_df = location_df.dropna(subset=['pos_x'])

    # 行车的具体信息---------------然后更改列名
    crane_df = pd.read_excel(data_path + r'/MOD_CRANE_CRANE.xlsx')

    # 计划号使用钢包的情况--------------然后删除无关的列，最后更改列名
    ladle_data = pd.read_excel(data_path + r'/MOD_CRANE_LADLE.xlsx')

    # 吊运任务----------------------然后删除无关的列，最后更改列名
    initial_capacity = {'11L5': 100, '12L5': 100, '1RH5': 100, '1LD5': 100, '2LD5': 100, '1LA5': 100, '2LA5': 100,
                        'DZ': 0}
    machine_run_time = {"LF": 60 * 60, "RH": 30 * 60, "LATS": 20 * 60, "BOF": 45 * 60, "CC": 60 * 60,
                        "DZ": 10 * 60, "QF": 15 * 60}

    hangjob_df = get_crane_hang_job_from_file()  # 处理下任务数据
    # 转换日期格式
    location_df = convert_date_columns(location_df)
    crane_df = convert_date_columns(crane_df)
    ladle_data = convert_date_columns(ladle_data)
    hangjob_df = convert_date_columns(hangjob_df)

    hangjob_df['prev_end_loc'] = np.nan  # 添加新的属性，初始化为NaN
    crane_df['cur_position'] = crane_df['init_pos_x']
    crane_df['task_list'] = [[] for _ in range(len(crane_df))]

    # Apply this function to the desired columns
    crane_df['boundary_increasing'] = crane_df['boundary_increasing'].apply(replace_6_with_5)
    crane_df['boundary_decreasing'] = crane_df['boundary_decreasing'].apply(replace_6_with_5)

    hangjob_df['start_loc'] = hangjob_df['start_loc'].apply(replace_6_with_5)
    hangjob_df['dest_loc'] = hangjob_df['dest_loc'].apply(replace_6_with_5)
    # 0=未执行，1=正在执行，2=执行完成，3=执行失败
    hangjob_df['status'] = 0  # 设置所有任务的初始状态为 0（未执行）

    location_df['loc_location'] = location_df['loc_location'].apply(replace_6_with_5)

    ladle_data['status']=0
    ladle_data['loc'] = ladle_data['loc'].apply(replace_6_with_5)
    ladle_data['even_time']=0

    return [location_df, crane_df, ladle_data, hangjob_df]

def replace_6_with_5(value):
    if isinstance(value, str) and value[-1] == '6':
        return value[:-1] + '5'
    return value

def divide_tasks(groupsize,hangjob_df):

    tasklist = group_tasks(hangjob_df, groupsize)

    return tasklist

#精炼跨的坐标范围和钢水接收跨的坐标范围
def span_pox_range(data):
    # 从数据中提取行车信息
    # "MOD_CRANE_CRANE" 是数据中的一个键，对应的值是一个包含行车信息的 DataFrame
    crane_df = data[1]

    # 计算跨3的范围
    span3_df = crane_df[crane_df['span_name'] == 3]  # 从行车信息中提取出跨3的行车信息
    span3_range = (
    span3_df['boundary_decreasing'].min(), span3_df['boundary_increasing'].max())  # 计算跨3的范围，即所有跨3行车移动范围的最小和最大值

    # 计算跨4的范围
    span4_df = crane_df[crane_df['span_name'] == 4]  # 从行车信息中提取出跨4的行车信息
    span4_range = (
    span4_df['boundary_decreasing'].min(), span4_df['boundary_increasing'].max())  # 计算跨4的范围，即所有跨4行车移动范围的最小和最大值

    return span3_range, span4_range  # 跨3和跨4的范围

def getpos_val(location_df, crane_state, task):
    # 获取任务的开始位置标记和值
    start_pos_val = 0
    start_pos = ''
    # task['start_loc']的值可能是位置标记，也可能是例如“DZ”的值，location_df的loc_location并不包含此字段值。所以，
    # 如果location_df中存在loc_location等于task['start_loc']的行
    if (location_df['loc_location'] == task['start_loc']).any():
        start_pos = location_df[location_df['loc_location'] == task['start_loc']]['loc_location'].values[0]
        start_pos_val = location_df[location_df['loc_location'] == task['start_loc']]['pos_x'].values[0]
    else:
        # 如果不存在，找出包含task['start_loc']的行
        target_rows = location_df[location_df['loc_location'].str.contains(task['start_loc'])]
        # 如果存在多行，找出距离crane_state['cur_location']最小的行
        if not target_rows.empty:
            target_rows = target_rows.copy()
            target_rows.loc[:, 'distance'] = abs(target_rows['pos_x'] - crane_state['cur_location'])

            start_pos_val = target_rows.loc[target_rows['distance'].idxmin(), 'pos_x']
            first_true_index = (target_rows['pos_x'] == start_pos_val).idxmax()
            start_pos = target_rows.loc[first_true_index, 'loc_location']
    # 获取任务的结束位置标记和值
    dest_pos_val = 0
    dest_pos = ''
    # task['dest_loc']的值可能是位置标记，也可能是例如“QF”的值，location_df的loc_location并不包含此字段值。所以，
    # 如果location_df中存在loc_location等于task['dest_loc']的行
    if (location_df['loc_location'] == task['dest_loc']).any():
        dest_pos = location_df[location_df['loc_location'] == task['dest_loc']]['loc_location'].values[0]
        dest_pos_val = location_df[location_df['loc_location'] == task['dest_loc']]['pos_x'].values[0]
    else:
        # 如果不存在，找出包含task['start_loc']的行
        target_rows = location_df[location_df['loc_location'].str.contains(task['dest_loc'])]
        # 如果存在多行，找出距离crane_state['cur_location']最小的行
        if not target_rows.empty:
            target_rows = target_rows.copy()
            target_rows.loc[:, 'distance'] = abs(target_rows['pos_x'] - crane_state['cur_location'])

            dest_pos_val = target_rows.loc[target_rows['distance'].idxmin(), 'pos_x']
            first_true_index = (target_rows['pos_x'] == dest_pos_val).idxmax()
            dest_pos = target_rows.loc[first_true_index, 'loc_location']

    return start_pos, start_pos_val, dest_pos, dest_pos_val

def select_task(location_df, casting_rows, casting_states, crane_df,crane_state, task_list, timeline):
    """
    输入：
    casting_df: 连铸机列表
    crane_state：行车状态
    task_list：任务列表
    timeline：总时间线
    功能：遍历任务列表，选择一个未被占用且离行车最近的任务作为下一个任务，优先考虑连铸机的任务（如果连铸机剩余浇筑时间少于15分钟）
    输出：被执行的任务 ID
    """
    # 初始化被执行的任务 ID
    task_id = None

    # 创建两个列表：`casting_tasks` 和 `non_casting_tasks`，分别用于存储目标是连铸机的任务和目标不是连铸机的任务。
    casting_tasks = []
    non_casting_tasks = []
    for index, task in task_list.iterrows():
        if task['status'] != 2:
            if 'CC' in task['dest_loc']:
                casting_tasks.append(task)
            else:
                non_casting_tasks.append(task)


    # 处理 `casting_tasks` 列表, 选择快要断浇的连铸机，将行车分配浇筑任务
    for i, task in enumerate(casting_tasks):

        start_pos, start_pos_val, dest_pos, dest_pos_val = getpos_val(location_df, crane_state, task)

        # 检查行车当前位置到任务开始位置是否被占用
        occupied = check_timeline_state(location_df, crane_df, crane_state, start_pos, timeline)
        # 如果未被占用，则该任务可作为候选任务
        if occupied == 0:
            casting_machine = start_pos #任务起始位置标记
            casting_machine_val = start_pos_val #任务起始位置值
            cur_loc_val = crane_state['cur_location'] #行车当前位置
            crane_speed = crane_df.iloc[crane_state['crane_name']]['speed'] #行车速度
            #行车到任务的开始位置所耗的时间
            wast_time = abs(casting_machine_val - cur_loc_val) / crane_speed #行车从当前位置到达连铸机的耗时
            #行车从任务开始位置到任务结束位置所耗的时间
            dest_wast_time = abs(dest_pos_val - cur_loc_val) / crane_speed #行车从当前位置到达连铸机的耗时

            index = casting_rows[casting_rows['loc_location'] == dest_pos].index[0] #连铸机编号


            # 遍历连铸机的状态，找到目标连铸机的完成时间
            for state in casting_states:
                if state['casting_name'] == index:
                    finish_time = state['finish_time']
                    break

            remaining_time = finish_time - crane_state['cur_time'] - wast_time - dest_wast_time
            if remaining_time < 15 * 60:
                task_id = task['plan_no']
                break
            elif i == len(casting_tasks):
                random_task = random.choice(casting_tasks)
                task_id = random_task['plan_no']

    # 如果我们没有在 `casting_tasks` 列表中找到适合的任务，我们就处理 `non_casting_tasks` 列表。
    if task_id is None:
        # 初始化最小距离为无穷大
        min_distance = float('inf')
        # 初始化最近任务
        nearest_task = None


        for task in non_casting_tasks:
            start_pos, start_pos_val, dest_pos, dest_pos_val = getpos_val(location_df, crane_state, task)
            # 检查行车当前位置到任务开始位置是否被占用
            occupied = check_timeline_state(location_df, crane_df, crane_state, start_pos, timeline)

            # 如果未被占用，则该任务可作为候选任务
            if occupied == 0:
                # 计算任务的开始位置和行车当前位置的绝对距离
                start_loc_x = location_df.loc[location_df['loc_location'] == start_pos, 'pos_x'].values[0]

                current_loc_x = crane_state['cur_location']

                distance = abs(start_loc_x - current_loc_x)

                # 如果这个任务的开始位置离行车当前位置更近，则选择这个任务
                if distance < min_distance:
                    min_distance = distance
                    nearest_task = task

        # 如果找到了最近的任务，更新 task_id
        if nearest_task is not None:
            task_id = nearest_task['plan_no']


    # 返回被执行的任务 ID
    return task_id


def check_timeline_state(location_df, crane_df, crane_state, target_pos, timeline):
    # 获取行车当前时间
    cur_time = crane_state['cur_time']

    # 预测行车移动到目标位置的时间
    target_pos_val = location_df[location_df['loc_location'] == target_pos]['pos_x'].values[0]
    predicted_end_time = cur_time + abs(target_pos_val - crane_state['cur_location']) / crane_df.loc[
        crane_state['crane_name'], 'speed']

    # 初始化 occupied 为 0，表示未占用
    occupied = 0

    # 如果时间线不为空
    if len(timeline) != 0:
        # 遍历时间线中的每一段时间
        for [start_time, timeline_end_time, start_pos, end_pos] in timeline:
            # 检查当前时间段是否与目标时间段有交叠，并找到所有交叠时间段
            if max(start_time, cur_time) < min(timeline_end_time, predicted_end_time):
                # 计算所有交叠时间段 subt
                subt_start_times = np.linspace(max(start_time, cur_time), min(timeline_end_time, predicted_end_time), num=100)
                for subt_start_time in subt_start_times:
                    subt_end_time = subt_start_time + 1.0  # assuming each subt is 1 time unit long
                    # 计算 subt 时间段内的位置变化，根据行车移动的方向调整位置变化的计算方式
                    if crane_state['cur_location'] < target_pos_val:  # 从左向右移动
                        subt_start_pos = crane_state['cur_location'] + crane_df.loc[
                            crane_state['crane_name'], 'speed'] * (subt_start_time - cur_time)
                        subt_end_pos = crane_state['cur_location'] + crane_df.loc[
                            crane_state['crane_name'], 'speed'] * (subt_end_time - cur_time)
                    else:  # 从右向左移动
                        subt_start_pos = crane_state['cur_location'] - crane_df.loc[
                            crane_state['crane_name'], 'speed'] * (subt_start_time - cur_time)
                        subt_end_pos = crane_state['cur_location'] - crane_df.loc[
                            crane_state['crane_name'], 'speed'] * (subt_end_time - cur_time)
                    # 检查 subt 时间段内的位置是否与目标位置有交叠
                    pos_overlap = max(min(subt_start_pos, subt_end_pos), start_pos) < min(
                        max(subt_start_pos, subt_end_pos), end_pos)
                    if pos_overlap:
                        # 如果有交叠，那么设置 occupied 为 1，表示已占用
                        occupied = 1
                        # 找到一个已占用的时间段，就可以提前结束循环
                        break
                # 如果已找到占用，提前结束外部循环
                if occupied == 1:
                    break

    # 返回占用状态，0=未占用，1=已占用
    return occupied



def update_crane_state(data, casting_rows, casting_states, crane_state, action, task_list, timeline, reward_params):
    """
    输入：行车状态 crane_states，动作 action，任务列表 task_list，总时间线 timeline，奖励参数 reward_params
    功能：根据动作更新行车状态，如果行车当前没有执行任务，选择一个任务；如果行车正在执行任务，根据动作更新行车状态，并给予奖励或惩罚
    输出：更新后的行车状态和奖励
    """
    # 初始化奖励为 0
    reward = 0
    ladle_data = data[2]
    crane_df = data[1]
    location_df = data[0]

    """
    检查行车是否有任务，若没有则找一个任务。
    """

    task_id = None
    # 如果行车的任务 ID 为 0，表示行车当前没有执行任务
    if crane_state['task_id'] == 0:
        # 选择一个任务
        task_id = select_task(location_df, casting_rows, casting_states, crane_df, crane_state, task_list, timeline)
        # 如果找到了任务
        if task_id is not None:
            # 更新行车的任务 ID
            crane_state['task_id'] = task_id

            #更新任务状态为正在执行
            task_list.loc[task_list['plan_no'] == task_id, 'status'] = 1

            # 给予奖励
            reward += reward_params['select_task']
        # 如果没有找到任务，给予惩罚
        else:
            reward -= reward_params['no_task']

    # 如果行车的任务 ID 不为 0，表示行车当前正在执行任务
    else:
        task_id = crane_state['task_id']
    # 根据任务 ID 找到任务
    task = None
    if task_id!=None:
        # print("task_id",task_id)
        task = next((t for i, t in task_list.iterrows() if t['plan_no'] == task_id), None)
        assert task is not None, "Task with given id not found in the task list.task"

    """
    根据行车动作，处理碰撞、断浇和更新状态。
    """
    pre_time = crane_state['cur_time']

    # 根据动作更新行车状态
    if action == -2:  # 提起钢包
        if crane_state['ladle_id'] == 0:
            # 在钢包数据中寻找与行车当前位置相同且状态为0的钢包
            ladle = ladle_data[(ladle_data['loc'] == crane_state['cur_location']) & (ladle_data['status'] == 0)]
            # 如果找到了符合条件的钢包
            if not ladle.empty:
                # 更新新事件时间：将当前时间+60秒赋值给crane_state的cur_time
                crane_state['cur_time'] += 60  # assuming time is in seconds


                # 更新上一个事件位置：上一个事件的位置 crane_state['pre_location'] = cur_location
                crane_state['pre_location'] = crane_state['cur_location']

                # 更新钢包id：将钢包id赋值给crane_state的ladle_id
                crane_state['ladle_id'] = ladle.iloc[0][
                    'ladle_name']  # assuming 'ladle_name' is the column name for the ladle id in ladle_data

                # 在时间线中添加新的事件
                timeline.append([crane_state['cur_time'] - 60, crane_state['cur_time'], crane_state['cur_location'],
                                 crane_state['cur_location']])


                # 给一个奖励
                reward += reward_params['pick_up_ladle']

            else:
                reward -= reward_params['wrong_action']
        else:
            # 更新新事件时间：将当前时间+60秒赋值给crane_state的cur_time
            crane_state['cur_time'] += 60  # assuming time is in seconds
            # 在时间线中添加新的事件
            timeline.append([crane_state['cur_time'] - 60, crane_state['cur_time'], crane_state['cur_location'],
                             crane_state['cur_location']])



        # 更新连铸机状态，并获取奖励
        casting_states, tmpreward = update_casting_states(casting_rows,casting_states, task, crane_state,
                                                          reward_params)

        reward += tmpreward

    # 向左移动
    elif action == -1:
        # 计算左边的第一个过跨线的位置
        new_pos = crane_df.loc[crane_state['crane_name'], 'boundary_decreasing']

        # 检查是否有冲突
        if check_timeline_state(location_df, crane_df, crane_state, new_pos, timeline) == 0:

            start_time = crane_state['cur_time']

            new_pos_val = location_df.loc[location_df['loc_location'] == new_pos, 'pos_x'].values[0]

            # 计算移动的耗时
            time_to_move = abs(new_pos_val - crane_state['cur_location']) / crane_df.loc[crane_state['crane_name'], 'speed']

            # 更新新事件时间
            crane_state['cur_time'] += time_to_move

            # 更新上一个事件位置
            crane_state['pre_location'] = crane_state['cur_location']

            # 更新当前事件新位置
            if new_pos_val == 13468 or new_pos_val == '13468':
                print("13468:",new_pos_val)
                exit(1)
            

            crane_state['cur_location'] = new_pos_val

            # 添加新的事件到时间线
            timeline.append((start_time, crane_state['cur_time'], crane_state['pre_location'], crane_state['cur_location']))
            if task is not None:
                # 计算任务的目标位置
                target_pos = task['dest_loc']
                start_pos = task['start_loc']

                # 如果左边的第一个过跨线的位置就是目标位置
                if crane_state['task_id'] != 0 and new_pos == target_pos:
                    # 找到当前任务并将其状态设为已完成
                    task_list.loc[task_list['plan_no'] == task['plan_no'], 'status'] = 2

                    # 更新为未执行任务
                    crane_state['task_id'] = 0

                    # 给一个较大的奖励
                    reward += reward_params['reach_target']
                elif crane_state['task_id'] == 0 and new_pos == start_pos:
                    # 如果左边的第一个过跨线的位置就是起始位置

                    # 找到当前任务并将其状态设为正在执行
                    task_list.loc[task_list['plan_no'] == task['plan_no'], 'status'] = 1

                    # 开始工作了，所以更新行车执行的任务id
                    crane_state['task_id'] = task['plan_no']

                    # 给一个奖励
                    reward += reward_params['reach_start']
            else:
                reward += reward_params['no_task_negative']
        else:
            # 给一个冲突的惩罚
            reward -= reward_params['conflict']

        # 更新连铸机状态，并获取奖励
        casting_states, tmpreward = update_casting_states(casting_rows,casting_states, task, crane_state,
                                                          reward_params)

        reward += tmpreward

    # 向右移动
    elif action == 1:
        # 计算右边的第一个过跨线的位置
        new_pos = crane_df.loc[crane_state['crane_name'], 'boundary_increasing']

        # 检查是否有冲突
        if check_timeline_state(location_df, crane_df, crane_state, new_pos, timeline) == 0:

            start_time = crane_state['cur_time']

            new_pos_val = location_df.loc[location_df['loc_location'] == new_pos, 'pos_x'].values[0]

            # 计算移动的耗时
            time_to_move = abs(new_pos_val - crane_state['cur_location']) / crane_df.loc[
                crane_state['crane_name'], 'speed']

            # 更新新事件时间
            crane_state['cur_time'] += time_to_move

            # 更新上一个事件位置
            crane_state['pre_location'] = crane_state['cur_location']

            # 更新当前事件新位置
            if new_pos_val == 13468 or new_pos_val == '13468':
                print("13468:",new_pos_val)
                exit(1)
            crane_state['cur_location'] = new_pos_val

            # 添加新的事件到时间线
            timeline.append((start_time, crane_state['cur_time'], crane_state['pre_location'], crane_state['cur_location']))
            if task is not None:
                # 计算任务的目标位置
                target_pos = task['dest_loc']
                start_pos = task['start_loc']

                # 如果右边的第一个过跨线的位置就是目标位置
                if crane_state['task_id'] != 0 and new_pos == target_pos:
                    # 找到当前任务并将其状态设为已完成
                    task_list.loc[task_list['plan_no'] == task['plan_no'], 'status'] = 2

                    # 更新为未执行任务
                    crane_state['task_id'] = 0

                    # 给一个较大的奖励
                    reward += reward_params['reach_target']
                elif crane_state['task_id'] == 0 and new_pos == start_pos:
                    # 如果右边的第一个过跨线的位置就是起始位置

                    # 找到当前任务并将其状态设为正在执行
                    task_list.loc[task_list['plan_no'] == task['plan_no'], 'status'] = 1

                    # 开始工作了，所以更新行车执行的任务id
                    crane_state['task_id'] = task['plan_no']

                    # 给一个奖励
                    reward += reward_params['reach_start']
            else:
                reward += reward_params['no_task_negative']
        else:
            # 给一个冲突的惩罚
            reward -= reward_params['conflict']

        # 更新连铸机状态，并获取奖励
        casting_states, tmpreward = update_casting_states(casting_rows,casting_states, task, crane_state,
                                                          reward_params)

        reward += tmpreward

    # 放下钢包
    elif action == 2:
        if crane_state['ladle_id'] != 0:
            if task is not None:
                # 计算任务的目标位置
                target_pos = task['dest_loc']
                # 如果行车当前位置不是目标位置
                if crane_state['cur_location'] != target_pos:
                    # 如果行车有执行的任务
                    if crane_state['task_id'] != 0:
                        # 判断行车是否在精炼跨上
                        if crane_df.iloc[crane_state['crane_name']]['span_name'] == 3:
                            # 判断当前行车的位置上是否是倾翻台
                            print("location_df['pos_x']",location_df['pos_x'])
                            print("crane_state['cur_location']",crane_state['cur_location'])
                            if int(location_df['pos_x']) == int(crane_state['cur_location']):
                                if 'QF' in location_df.loc[int(location_df['pos_x']) == int(crane_state['cur_location']), 'loc_location'].values[0]:
                                    # 更新新事件时间：将当前时间+60秒赋值给crane_state的cur_time
                                    crane_state['cur_time'] += 60

                                    # 更新行车上一个事件的位置
                                    crane_state['pre_location'] = crane_state['cur_location']

                                    # 更新钢包id为空字符串
                                    crane_state['ladle_id'] = ''

                                    # 更新为未执行任务
                                    crane_state['task_id'] = 0

                                    # 在时间线中添加新的事件
                                    timeline.append(
                                        [crane_state['cur_time'] - 60, crane_state['cur_time'], crane_state['cur_location'],
                                         crane_state['cur_location']])
                                    # 找到当前任务并将其状态设为已完成
                                    task_list.loc[task_list['plan_no'] == task['plan_no'], 'status'] = 2

                                else:
                                    # 如果当前行车位置不是倾翻台的位置，则给予一个惩罚
                                    reward -= reward_params['wrong_action']
                            else:
                                # 如果行车当前位置为无效位置，不在locaiton_df里面，则给予一个惩罚
                                reward -= reward_params['wrong_action']
                        else:
                            # 如果行车不在精炼跨上，则给予一个惩罚
                            reward -= reward_params['wrong_action']
                    else:
                        # 如果行车没有执行的任务，则给予一个惩罚
                        reward -= reward_params['wrong_action']
            else:
                reward += reward_params['no_task_negative']
        else:
            reward -= reward_params['wrong_action']
        # 更新连铸机状态，并获取奖励
        casting_states, tmpreward = update_casting_states(casting_rows,casting_states, task, crane_state,
                                                          reward_params)

        reward += tmpreward

    elif action == 0:

        # 更新新事件时间：等待10秒
        crane_state['cur_time'] += 10

        # 在时间线中添加新的事件
        timeline.append(
            [crane_state['cur_time']-10, crane_state['cur_time'], crane_state['cur_location'],
             crane_state['cur_location']])

        reward += reward_params['no_task_negative']

    #如果超出任务的最晚执行时间则：1：给出任务失败标记。2：给予惩罚。如果是正在执行或执行完成则给出相应的奖励
    if task is not None:
        if crane_state['cur_time'] > task['latest_down_time']:
            task_list.loc[task_list['plan_no'] == task['plan_no'], 'status'] = 3
            reward += reward_params['false_task']
        elif task['status'] == 2:
            reward += reward_params['false_task']
        elif task['status'] == 1:
            reward += reward_params['going_task']

    timeline.append([pre_time, crane_state['cur_time'], crane_state['pre_location'], crane_state['cur_location']])



    # 返回更新后的行车状态和奖励
    return crane_state, casting_states, timeline, reward,task_list

def check_task_completion(cur_time, task_list, reward_params):
    """
    输入：当前时间 cur_time，任务列表 task_list，奖励参数 reward_params
    功能：遍历所有的任务，如果任务的最晚完成时间小于当前时间，则任务未在规定时间内完成，给予惩罚
    输出：总惩罚
    """
    # 初始化总惩罚为 0
    total_penalty = 0

    # 遍历所有的任务
    for index, task in task_list.iterrows():
        # 如果任务的最晚完成时间小于当前时间，则任务未在规定时间内完成，给予惩罚
        if task['latest_down_time'] < cur_time and task['status'] != 2:
            total_penalty += reward_params['task_delay']

    # 返回总惩罚
    return total_penalty

def check_casting_continuity(cur_time, casting_list, task_list, reward_params):
    """
    输入：
    cur_time，当前时间
    casting_list，连铸机列表
    task_list，任务列表
    reward_params, 奖励参数
    功能：遍历所有的连铸机，找到以该连铸机为目标的所有任务，如果没有找到任务，给予惩罚；计算上一个任务维持的时间，如果上一个任务维持的时间小于连铸机的浇筑时间，给予惩罚
    输出：总惩罚
    """
    # 初始化总惩罚为 0
    total_penalty = 0

    # 遍历所有的连铸机
    for casting_machine in casting_list:
        # 找到以该连铸机为目标的所有任务
        tasks = [task for task in task_list if task['dest_loc'] == casting_machine]
        # 如果没有找到任务，给予惩罚
        if not tasks:
            total_penalty += reward_params['casting_no_task']
            continue

        # 计算上一个任务维持的时间
        last_task = max(tasks, key=lambda task: task['latest_down_time'])
        last_time = last_task['latest_down_time'] - cur_time

        # 如果上一个任务维持的时间小于连铸机的浇筑时间，给予惩罚
        if last_time < casting_df.loc[casting_machine, 'casting_time']:
            total_penalty += reward_params['casting_interval']

    # 返回总惩罚
    return total_penalty

# 定义一个函数来更新连铸机的状态和检查是否存在断浇
def update_casting_states(casting_rows,casting_states, task, crane_state, reward_params):
    """
    输入：
    casting_states：连铸机状态列表
    task：当前完成的任务
    crane_state：行车状态
    reward_params：奖励参数
    功能：更新连铸机的浇筑时间，检查是否存在断浇，给出相应的奖励或惩罚
    输出：连铸机状态列表，奖励
    """

    # 初始化奖励为 0
    reward = 0

    # 如果当前完成的任务的目标位置为连铸机，增加对应连铸机的浇筑时长
    if task is not None:
        if 'CC' in task['dest_loc']:
            casting_machine = task['dest_loc']
            # 找到 loc_location 值为 casting_machine 的行的索引
            casting_machine_index = casting_rows.loc[casting_rows['loc_location'] == casting_machine].index[0]

            for i in range(len(casting_states)):
                if casting_states[i]['casting_name'] == casting_machine_index:
                    casting_states[i]['finish_time'] += 45 * 60

    # 遍历所有的连铸机，检查是否存在断浇
    for i in range(len(casting_states)):
        casting_machine = casting_states[i]
        # 如果连铸机的剩余浇筑时间已经为0，并且当前时间大于连铸机的当前时间，那么表示该连铸机发生了断浇
        if crane_state['cur_time'] > casting_machine['finish_time']:
            # 发生断浇，给予大的惩罚
            reward -= reward_params['casting_interruption']
        else:
            # 如果没有发生断浇，更新连铸机的当前时间和当前状态
            casting_states[i]['cur_time'] = crane_state['cur_time']
            casting_states[i]['cur_state'] = 0

    # 返回连铸机状态列表和奖励
    return casting_states, reward

def select_best_line(job, cross_lines):
    TRAVEL_TIME = 10  # Assume fixed travel time from one line to another
    cross_line_busy_till = [0] * len(cross_lines)  # 初始化每个过跨线的空闲时间为0
    # 计算每个过跨线距离任务开始的时间
    line_times = [max(line_busy_till + TRAVEL_TIME, job['earliest_up_time']) for line_busy_till in
                  cross_line_busy_till]

    # 找到可以在最早时间内完成任务的过跨线
    best_line = line_times.index(min(line_times))

    # 如果选择的过跨线不能在任务的最晚结束时间前完成，那么重新选择过跨线
    if line_times[best_line] > job['latest_down_time']:
        available_lines = [i for i, line_time in enumerate(line_times) if line_time <= job['latest_down_time']]
        # 从可用的过跨线中选择一个空闲时间最长的过跨线
        best_line = available_lines[np.argmax([cross_line_busy_till[i] for i in available_lines])]

    return best_line


def calculate_crane_collisions(timeline):
    """
    计算行车碰撞次数
    """
    # 首先，我们需要对时间线进行排序，以便我们可以按顺序检查每个事件
    timeline.sort(key=lambda x: (x[0], x[1]))  # 根据开始时间和结束时间排序

    collision_count = 0  # 初始化碰撞次数为0
    # 然后，我们遍历时间线，检查每个事件与后面的事件是否有位置交集
    for i in range(len(timeline)):
        for j in range(i + 1, len(timeline)):
            # 如果后面的事件开始时间早于当前事件的结束时间，那么这两个事件在时间上有交集
            if timeline[j][0] < timeline[i][1]:
                # 如果有时间交集的事件在位置上也有交集，那么就发生了碰撞
                if max(timeline[i][2], timeline[j][2]) < min(timeline[i][3], timeline[j][3]):
                    collision_count += 1
            else:
                # 如果后面的事件开始时间晚于当前事件的结束时间，那么我们可以提前结束循环，因为后面的事件不可能与当前事件碰撞
                break

    return collision_count  # 返回碰撞次数

def calculate_casting_interruptions(casting_states, timeline):
    """
    计算连铸机断浇次数
    """
    # 首先，我们需要获取时间线上的最早时间和最晚时间
    min_time = min(event[0] for event in timeline)
    max_time = max(event[1] for event in timeline)

    interruption_count = 0  # 初始化断浇次数为0
    # 然后，我们遍历每个连铸机状态
    for casting_state in casting_states:
        # 如果连铸机在时间线上的有效时间范围内是断浇状态，那么就发生了断浇

        if casting_state['cur_state'] == 0 and casting_state['cur_time'] < max_time and casting_state['finish_time'] > min_time:
            interruption_count += 1

    return interruption_count  # 返回断浇次数

def calculate_task_completion_rate(hangjob_df):
    """
    计算任务完成率
    """
    # 总任务数
    total_tasks = len(hangjob_df)
    # 完成的任务数
    completed_tasks = len(hangjob_df[hangjob_df['status'] == 2])

    # 任务完成率
    completion_rate = completed_tasks / total_tasks

    return completion_rate  # 返回任务完成率
