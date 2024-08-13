import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
height = 18
#GT位置参数
GT_1 = np.array([28,19]);GT_2 = np.array([27,27])
GT_3 = np.array([39,31]);GT_4 = np.array([34,36])
GT_5 = np.array([26,35])
GT_list = [GT_1,GT_2,GT_3,GT_4,GT_5]
#障碍物位置参数
OP_1 = np.array([13,8]);OP_2 = np.array([11,18])
OP_3 = np.array([10,26]);OP_4 = np.array([14,38])
OP_5 = np.array([26,9]);OP_6 = np.array([29,18])
OP_7 = np.array([20,22]);OP_8 = np.array([32,26])
OP_list = [OP_1,OP_2,OP_3,OP_4,OP_5,OP_6,OP_7,OP_8]

def distance(x, y):
    vector_difference = x - y
    d_i = np.linalg.norm(vector_difference)
    return d_i

def risk(s):
    result_r = 1
    for i in range(8):
        sigma = 2  # 后续可以调整
        d_i = distance(s, OP_list[i])
        r_i = (1/((np.sqrt(2*np.pi))*np.sqrt(sigma)))*np.exp((-(d_i)**2)/(2*(sigma)))
        result_r = result_r * (1 - r_i)
    result_R = 1 - result_r
    return result_R

# 计算热力图数据
heat_map_data = np.zeros((50, 50))
for i in range(50):
    for j in range(50):
        heat_map_data[i, j] = risk(np.array([i, j]))

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(heat_map_data, cmap="Blues")#Blues
plt.title('Environment')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
# 设置坐标轴范围
plt.xlim(0, 50)
plt.ylim(0, 50)
# 设置横轴每隔10个单位标记一次，并显示数字
plt.xticks(np.arange(0, 51, step=10), np.arange(0, 51, step=10))
# 设置纵轴每隔10个单位标记一次，并显示数字
plt.yticks(np.arange(0, 51, step=10), np.arange(0, 51, step=10))
# 标注障碍物位置
# 标注障碍物位置并添加障碍物图例
plt.scatter([op[1] for op in OP_list], [op[0] for op in OP_list], marker='X', color='red', s=100, label='Obstacle')
plt.legend()

# 标注GT位置并添加GT图例
plt.scatter([gt[1] for gt in GT_list], [gt[0] for gt in GT_list], marker='*', color='green', s=100, label='Users')
plt.legend()

plt.legend()

# 加载无人机坐标数据
x_trajectory = np.load('x_out.npy')
y_trajectory = np.load('y_out.npy')

# 确保数据维度一致，这里是假设都是1维数组
trajectory_points = np.column_stack((x_trajectory, y_trajectory))

# 绘制无人机飞行轨迹
plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], marker='o', markersize=2,linestyle='-', linewidth=2, color='orange', label='UAV')

# 添加无人机轨迹图例
plt.legend()
plt.savefig('Heatmap.png')
plt.show()
