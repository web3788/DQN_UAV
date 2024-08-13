import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 加载无人机坐标数据
risk_1 = np.load('cap_out_0.1.npy')
risk_2 = np.load('cap_out_1.npy')
risk_3 = np.load('cap_out_10.npy')
risk_4 = np.load('cap_out_20.npy')


# 取每个数组的前500个数据
risk_1_slice = risk_1[:300]
risk_2_slice = risk_2[:300]
risk_3_slice = risk_3[:300]
risk_4_slice = risk_4[:300]

# 准备绘图
plt.figure(figsize=(13, 7))
plt.title('Capacity Comparison')
plt.xlabel('Time Steps')
plt.ylabel('Capacity')

# 绘制四条曲线
plt.plot(np.arange(300), risk_2_slice, label='K=0.1')
plt.plot(np.arange(300), risk_1_slice, label='K=1')
plt.plot(np.arange(300), risk_3_slice, label='K=10')
plt.plot(np.arange(300), risk_4_slice, label='K=20')




# 添加图例
plt.legend()

plt.savefig('Cap_all.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()