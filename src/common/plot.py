import matplotlib.pyplot as plt
import numpy as np

# 数据
models = ['SRCNN', 'FSRCNN', 'ESPCN', 'EDSR', 'IMDN']
psnr_x2 = [34.40, 34.35, 33.99, 36.58, 36.62]
psnr_x3 = [30.48, 30.40, 30.23, 32.17, 32.24]
psnr_x4 = [28.09, 28.24, 28.04, 29.60, 29.32]

# 设置柱状图的位置
x = np.arange(len(models))  # 模型标签的位置
width = 0.25  # 柱子的宽度

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(12, 7))

# 绘制三组柱状图
rects1 = ax.bar(x - width, psnr_x2, width, label='Scale x2', color='#8DD2C5')
rects2 = ax.bar(x, psnr_x3, width, label='Scale x3', color='#BFBCDA')
rects3 = ax.bar(x + width, psnr_x4, width, label='Scale x4', color='#F47F72')

# 添加标题和标签
ax.set_title('PSNR Comparison on Set5 Dataset', fontsize=25)
ax.set_ylabel('PSNR (dB)', fontsize=15)
ax.set_xlabel('Models', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# 在柱子上方显示数值
def autolabel(rects):
    """在每个柱子上方附加一个文本标签，显示其高度。"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# 添加网格线
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.5)

# 调整Y轴范围以提供更多空间给顶部的标签
ax.set_ylim(0, max(psnr_x2) * 1.1)

# 优化布局
fig.tight_layout()

# 显示图像
plt.show()




