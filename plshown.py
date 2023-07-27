import matplotlib.pyplot as plt
import numpy as np

class Visualization:

    def __init__(self):
        self.fig, self.ax = plt.subplots(3, 1)
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Use colormap to generate 10 different colors

    def plot_crane_trajectory(self, crane_states):
        crane_names = list(set([item['crane_name'] for sublist in crane_states for item in sublist]))

        for i, crane_name in enumerate(crane_names):
            time = [item['cur_time'] for sublist in crane_states for item in sublist if
                    item['crane_name'] == crane_name]
            positions = [item['cur_location'] for sublist in crane_states for item in sublist if
                         item['crane_name'] == crane_name]
            self.ax[0].plot(time, positions, color=self.colors[i % len(self.colors)], label='crane_' + str(crane_name))

        self.ax[0].legend()
        self.ax[0].set_xlabel('Time')
        self.ax[0].set_ylabel('Crane Position')
        self.ax[0].set_title('Crane Trajectories')

    def plot_casting_state(self, casting_states):
        casting_names = list(set([item['casting_name'] for sublist in casting_states for item in sublist]))

        for i, casting_name in enumerate(casting_names):
            casting_states_crane = [item for sublist in casting_states for item in sublist if
                                    item['casting_name'] == casting_name]
            casting_states_crane.sort(key=lambda x: x['cur_time'])  # Sort casting states by cur_time

            time = []
            casting_volume = []
            volume = 0
            for j, casting_state in enumerate(casting_states_crane):
                if j > 0 and casting_states_crane[j]['cur_time'] > casting_states_crane[j - 1]['cur_time']:
                    volume += 45  # Add 45 minutes of casting time when cur_time reaches the next cur_time
                time.append(casting_state['cur_time'])
                casting_volume.append(volume)

                self.ax[1].plot(time, casting_volume, color=self.colors[i], label='casting_' + str(casting_name))

            self.ax[1].legend()
            self.ax[1].set_xlabel('Time')
            self.ax[1].set_ylabel('Casting Volume')
            self.ax[1].set_title('Casting States')

    def plot_task_completion_rate(self, time, completion_rate):
        """
        绘制任务完成率
        输入：
        time：时间列表
        completion_rate：完成率列表
        """
        self.ax[2].clear()
        self.ax[2].plot(time, completion_rate)
        self.ax[2].set_xlabel('Time')
        self.ax[2].set_ylabel('Completion Rate')
        self.ax[2].set_title('Task Completion Rate')

    def show(self):
        # 显示图形
        plt.show()
