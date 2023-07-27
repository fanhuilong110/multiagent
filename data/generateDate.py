from second.common import *


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

    ladle_data['loc'] = ladle_data['loc'].apply(replace_6_with_5)

    # 合并 DataFrame
    df = pd.concat([hangjob_df, location_df, crane_df, ladle_data])

    # 存储为 HDF5 文件，并添加键值
    with pd.HDFStore('D:\paper\code\muldp\second\data\data.h5') as store:
        store.put('df/location_df', location_df)
        store.put('df/crane_df', crane_df)
        store.put('df/ladle_data', ladle_data)
        store.put('df/hangjob_df', hangjob_df)

    return [location_df, crane_df, ladle_data, hangjob_df]


def main():
    data_path = r"D:\paper\code\muldp\data"  # 指定数据存放的基础路径
    data = load_data(data_path)



if __name__ == "__main__":
    main()