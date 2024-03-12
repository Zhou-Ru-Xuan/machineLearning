import pandas as pd  # 导入Pandas
import matplotlib.pyplot as plt  # 导入Matplotlib的pyplot模块

df_sales = pd.read_csv('易速鲜花订单记录.csv')  # 载入数据
print(df_sales.head())  # 显示头几行数据

# 构建月度的订单数的DataFrame
df_sales['消费日期'] = pd.to_datetime(df_sales['消费日期'], format='%m/%d/%Y %H:%M')  # 转化日期格式
df_orders_monthly = df_sales.set_index('消费日期')['订单号'].resample('ME').nunique()  # 每个月的订单数量
print(df_orders_monthly)
# 设定绘图的画布
ax = pd.DataFrame(df_orders_monthly.values).plot(grid=True, figsize=(15, 6), legend=False)
ax.set_xlabel('月份')  # X轴label
ax.set_ylabel('订单数')  # Y轴Label
ax.set_title('月度订单数')  # 图题
# 设定X轴月份显示格式
plt.xticks(
    range(len(df_orders_monthly.index)),
    [x.strftime('%Y-%m') for x in df_orders_monthly.index],
    rotation=45)
plt.show()  # 绘图
