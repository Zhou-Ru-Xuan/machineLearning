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
# plt.show()  # 绘图

df_sales = df_sales.drop_duplicates()  # 删除重复的数据行

print(df_sales.isna().sum())  # NaN出现的次数

print(df_sales.describe())  # df_sales的统计信息

df_sales = df_sales.loc[df_sales['数量'] > 0]  # 清洗掉数量小于等于0的数据

print(df_sales.describe())  # df_sales的统计信息

df_sales['总价'] = df_sales['数量'] * df_sales['单价']  # 计算每单的总价
print(df_sales.head())  # 显示头几行数据

df_user = pd.DataFrame(df_sales['用户码'].unique())  # 生成以用户码为主键的结构df_user
df_user.columns = ['用户码']  # 设定字段名
df_user = df_user.sort_values(by='用户码', ascending=True).reset_index(drop=True)  # 按用户码排序
print(df_user)  # 显示df_user

df_sales['消费日期'] = pd.to_datetime(df_sales['消费日期'])  # 转化日期格式
df_recent_buy = df_sales.groupby('用户码').消费日期.max().reset_index()  # 构建消费日期信息
df_recent_buy.columns = ['用户码', '最近日期']  # 设定字段名
df_recent_buy['R值'] = (df_recent_buy['最近日期'].max() - df_recent_buy['最近日期']).dt.days  # 计算最新日期与上次消费日期的天数
df_user = pd.merge(df_user, df_recent_buy[['用户码', 'R值']], on='用户码')  # 把上次消费距最新日期的天数（R值）合并至df_user结构
print(df_user.head())  # 显示df_user头几行数据

df_frequency = df_sales.groupby('用户码').消费日期.count().reset_index()  # 计算每个用户消费次数，构建df_frequency对象
df_frequency.columns = ['用户码', 'F值']  # 设定字段名称
df_user = pd.merge(df_user, df_frequency, on='用户码')  # 把消费频率整合至df_user结构
print(df_user.head())  # 显示df_user头几行数据

df_revenue = df_sales.groupby('用户码').总价.sum().reset_index()  # 根据消费总额，构建df_revenue对象
df_revenue.columns = ['用户码', 'M值']  # 设定字段名称
df_user = pd.merge(df_user, df_revenue, on='用户码')  # 把消费金额整合至df_user结构
print(df_user.head())  # 显示df_user头几行数据
