import copy

import pandas as pd
from gurobipy import *
from datetime import datetime
import gurobipy as gp

basePath = r"D:\paper\code\muldp\data"

# 导入数据 过跨线
lineNameList = ["32R", "31R", "22L", "21L", "4LD", "3LD", "2RH", "1TF", "2LA", "1LA", "2LD", "1LD", "1RH", "1GF", "12L",
                "11L", "0RH"]
line = []
for i in lineNameList:
    line.append(i + "5")
    line.append(i + "6")
line_ALL = copy.deepcopy(line)
line_ALL.append("L")
DZ_ALL = ["3DZ6", "2DZ6", "1DZ6", "DZ"]
DZ = ["3DZ6", "2DZ6", "1DZ6"]

BOF = ["BOF" + str(i + 1) for i in range(4)]
RH = ["RH" + str(i + 1) for i in range(3)]
LF = ["LF" + str(i + 1) for i in range(2)]
LATS = ['LATS' + str(i + 1) for i in range(2)]
CC_temp = [str(i + 1) + "CC" for i in range(4)]
CC = []
for i in CC_temp:
    CC.append(i + str("6"))
QF = [str(i + 1) + "QF5" for i in range(4)]
QF_ALL = copy.deepcopy(QF)
QF_ALL.append("QF")

def get_crane_hang_job_from_file():
    """
    :return: 从excel读取数据，返回一个包含任务信息的字典
    """

    data = pd.read_excel(basePath + r'\MOD_CRANE_HANGJOB.xlsx')
    data["earliest_up_time"] = data["earliest_up_time"].apply(lambda x: change_date_2_second(x))
    data["latest_down_time"] = data["latest_down_time"].apply(lambda x: change_date_2_second(x))
    data["start_loc"] = data["start_loc"].fillna("").astype(str) + data["start_mac"].fillna("").astype(str)
    data["dest_loc"] = data["dest_loc"].fillna("").astype(str) + data["dest_mac"].fillna("").astype(str)

    # 删除只有DZ的QF列,并且钢包位置在QF的任务
    # 获取要删除的行的索引
    data = data.sort_values(['ladle_no', 'plan_no', 'earliest_up_time'], ascending=[True, True, True])
    data = data.reset_index().drop(
        ['index', 'start_mac', 'dest_mac', 'work_loc', 'work_time', 'job_type', 'flag_downline'], axis=1)
    for i, row in data.iterrows():
        if data["start_loc"].get(i) == 'DZ' and data["dest_loc"].get(i) == 'QF':
            # 找到plan_no只出现一次的行
            if data['plan_no'].value_counts()[data["plan_no"].get(i)] == 1:
                ladle_no = data["ladle_no"].get(i)
                if "QF" not in Ladle_loc[ladle_no] and data["ladle_no"].get(i) == data["ladle_no"].get(
                        i + 1) and i + 1 < len(data):
                    data = data.drop(i + 1)
                data = data.drop(i)
    # 重置索引
    data = data.reset_index(drop=True)
    insert_temp_dict = {}
    for i in range(len(data)):
        if data["start_loc"].get(i) == 'DZ' and data["dest_loc"].get(i) == 'QF':
            temp = copy.deepcopy(data.iloc[i])
            temp["start_loc"] = "L"
            temp["dest_loc"] = "QF"
            insert_temp_dict[i] = temp
            column_index = data.columns.get_loc("dest_loc")
            data.iloc[i, column_index] = "L"
    insert_temp_dict = dict(sorted(insert_temp_dict.items()))
    count = 0
    for i, temp in insert_temp_dict.items():
        data = pd.concat([data.iloc[:i + 1 + count], temp.to_frame().T, data.iloc[i + 1 + count:]]).reset_index(
            drop=True)
        count += 1
    # 计算数据框中的最小值
    min_time = data['earliest_up_time'].min()
    # 所有的 time 减去最小值
    data['earliest_up_time'] = data['earliest_up_time'].apply(lambda x: x - min_time)
    # 计算数据框中的最小值
    # min_time = data['earliest_up_time'].min()
    # 所有的 time 减去最小值
    data['latest_down_time'] = data['latest_down_time'].apply(lambda x: x - min_time)
    result = {}
    # 给每一个任务附上一个job id
    for i, row in data.iterrows():
        row["job_id"] = i
        row_key = row["job_id"]
        row_data = row[
            ["plan_no", "start_loc", "dest_loc", "earliest_up_time",
             "latest_down_time", "ladle_no"]].values
        result[row_key] = row_data
    data.to_excel("temp.xlsx")
    before = ""
    ladle_first_job = []
    for i, row in data.iterrows():
        current = row["ladle_no"]
        if current != before:
            ladle_first_job.append(row['plan_no'])
        before = current
    last_job_end_time = data['latest_down_time'].max()
    print(last_job_end_time)
    # return result, ladle_first_job, last_job_end_time
    # //如果需要数据框,返回data
    return data

def change_date_2_second(date):
    """
    将形如 '2022-10-14 10:16:27.000000'的日期转换成秒
    :param date: 日期字符串
    :return:返回日期的秒数
    """
    date_time_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')
    return datetime.timestamp(date_time_obj)


def get_crane_ladle_from_file():
    """
    :return: 从excel读取数据，返回一个包含钢包信息的字典
    """
    result = {}
    data = pd.read_excel(basePath + r'/MOD_CRANE_LADLE.xlsx')
    for i, row in data.iterrows():
        row_key = row["ladle_name"]
        row_data = row[['loc', 'status', "plan_no"]].values
        result[row_key] = row_data
    return result


def get_crane_ladle_from_file():
    """
    :return: 从excel读取数据，返回一个包含钢包信息的字典
    """
    result = {}
    data = pd.read_excel(basePath + r'\MOD_CRANE_LADLE.xlsx')
    for i, row in data.iterrows():
        row_key = row["ladle_name"]
        row_data = row[['loc', 'status', "plan_no"]].values
        result[row_key] = row_data
    return result

def get_crane_type_by_position(start_loc, dest_loc):
    '''
    返回使用什么类型的行车
    :param start_loc: 起点
    :param dest_loc: 终点
    :return:
    '''
    # 只能用钢水接收跨的行车
    if "倒渣到路线" == judge_task_type(start_loc, dest_loc):
        return "钢水"
    # 只能用精炼跨的行车
    elif "路线到倾翻台" == judge_task_type(start_loc, dest_loc):
        return "精炼"
    # 只能用精炼跨的行车
    elif "倾翻台到路线" == judge_task_type(start_loc, dest_loc):
        return "精炼"
    # 只能用钢水接收跨的行车
    elif "连铸到倒渣" == judge_task_type(start_loc, dest_loc):
        return "钢水"
    # 只能使用重行车
    elif "路线到路线" == judge_task_type(start_loc, dest_loc):
        return "重行车"
    # 只能使用重行车
    elif "路线到连铸" == judge_task_type(start_loc, dest_loc):
        return "重行车"
    # 只能使用重行车
    elif "精炼到精炼" == judge_task_type(start_loc, dest_loc):
        return "重行车"

def judge_task_type(start_loc, end_loc):
    # print(start_loc, end_loc)
    # 倾翻台到路线
    if start_loc in QF_ALL and end_loc in line_ALL:
        return "倾翻台到路线"
    # 连铸到倒渣
    elif start_loc in CC and end_loc in DZ_ALL:
        return "连铸到倒渣"
    elif start_loc in DZ_ALL and end_loc in line_ALL:
        return "倒渣到路线"
    elif start_loc in line_ALL and end_loc in QF_ALL:
        return "路线到倾翻台"
    # 路线到路线
    elif start_loc in line_ALL and end_loc in line_ALL:
        return "路线到路线"
    elif start_loc in line_ALL and end_loc in CC:
        return "路线到连铸"
    else:
        print(start_loc,end_loc)
        exit()
    # raise ValueError("没有对应的工作类型")


def get_crane_hang_job_from_file():
    """
    :return: 从excel读取数据，返回一个包含任务信息的字典
    """

    data = pd.read_excel(basePath + r'\MOD_CRANE_HANGJOB.xlsx')
    data["earliest_up_time"] = data["earliest_up_time"].apply(lambda x: change_date_2_second(x))
    data["latest_down_time"] = data["latest_down_time"].apply(lambda x: change_date_2_second(x))
    data["start_loc"] = data["start_loc"].fillna("").astype(str) + data["start_mac"].fillna("").astype(str)
    data["dest_loc"] = data["dest_loc"].fillna("").astype(str) + data["dest_mac"].fillna("").astype(str)

    # 删除只有DZ的QF列,并且钢包位置在QF的任务
    # 获取要删除的行的索引
    data = data.sort_values(['ladle_no', 'plan_no', 'earliest_up_time'], ascending=[True, True, True])
    data = data.reset_index().drop(
        ['index', 'start_mac', 'dest_mac', 'work_loc', 'work_time', 'job_type', 'flag_downline'], axis=1)
    for i, row in data.iterrows():
        if data["start_loc"].get(i) == 'DZ' and data["dest_loc"].get(i) == 'QF':
            # 找到plan_no只出现一次的行
            if data['plan_no'].value_counts()[data["plan_no"].get(i)] == 1:
                ladle_no = data["ladle_no"].get(i)
                if "QF" not in Ladle_loc[ladle_no] and data["ladle_no"].get(i) == data["ladle_no"].get(
                        i + 1) and i + 1 < len(data):
                    data = data.drop(i + 1)
                data = data.drop(i)
    # 重置索引
    data = data.reset_index(drop=True)
    insert_temp_dict = {}
    for i in range(len(data)):
        if data["start_loc"].get(i) == 'DZ' and data["dest_loc"].get(i) == 'QF':
            temp = copy.deepcopy(data.iloc[i])
            temp["start_loc"] = "L"
            temp["dest_loc"] = "QF"
            insert_temp_dict[i] = temp
            column_index = data.columns.get_loc("dest_loc")
            data.iloc[i, column_index] = "L"
    insert_temp_dict = dict(sorted(insert_temp_dict.items()))
    count = 0
    for i, temp in insert_temp_dict.items():
        data = pd.concat([data.iloc[:i + 1 + count], temp.to_frame().T, data.iloc[i + 1 + count:]]).reset_index(
            drop=True)
        count += 1
    # 计算数据框中的最小值
    min_time = data['earliest_up_time'].min()
    # 所有的 time 减去最小值
    data['earliest_up_time'] = data['earliest_up_time'].apply(lambda x: x - min_time)
    # 计算数据框中的最小值
    # min_time = data['earliest_up_time'].min()
    # 所有的 time 减去最小值
    data['latest_down_time'] = data['latest_down_time'].apply(lambda x: x - min_time)
    result = {}
    # 给每一个任务附上一个job id
    for i, row in data.iterrows():
        row["job_id"] = i
        row_key = row["job_id"]
        row_data = row[
            ["plan_no", "start_loc", "dest_loc", "earliest_up_time",
             "latest_down_time", "ladle_no"]].values
        result[row_key] = row_data
    data.to_excel("temp.xlsx")
    before = ""
    ladle_first_job = []
    for i, row in data.iterrows():
        current = row["ladle_no"]
        if current != before:
            ladle_first_job.append(row['plan_no'])
        before = current
    last_job_end_time = data['latest_down_time'].max()
    print(last_job_end_time)
    # return result, ladle_first_job, last_job_end_time
    # //如果需要数据框,返回data
    return data

# 获取数据 钢包信息
Ladle_name, Ladle_loc, Ladle_status, Ladle_plan_no = Ladle = multidict(get_crane_ladle_from_file())