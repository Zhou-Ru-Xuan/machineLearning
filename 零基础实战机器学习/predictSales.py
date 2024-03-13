import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # 导入线性回归模型
from sklearn.tree import DecisionTreeRegressor  # 导入决策树回归模型
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林回归模型
from sklearn.metrics import r2_score  # 导入Sklearn评估模块
import matplotlib.pyplot as plt  # 导入Matplotlib的pyplot模块

df_sales = pd.read_csv('易速鲜花订单记录.csv')  # 载入数据
df_sales = df_sales.drop_duplicates()  # 删除重复的数据行
df_sales = df_sales.loc[df_sales['数量'] > 0]  # 清洗掉数量小于等于0的数据
df_sales['总价'] = df_sales['数量'] * df_sales['单价']  # 计算每单的总价

print('日期范围: %s ~ %s' % (df_sales['消费日期'].min(), df_sales['消费日期'].max()))  # 显示日期范围（格式转换前）
df_sales['消费日期'] = pd.to_datetime(df_sales['消费日期'])  # 转换日期格式
print('日期范围: %s ~ %s' % (df_sales['消费日期'].min(), df_sales['消费日期'].max()))  # 显示日期范围

df_sales = df_sales.loc[df_sales['消费日期'] < '2021-06-01']  # 只保留整月数据
print('日期范围: %s ~ %s' % (df_sales['消费日期'].min(), df_sales['消费日期'].max()))  # 显示日期范围

df_sales_3m = df_sales[(df_sales.消费日期 > '2020-06-01') & (df_sales.消费日期 <= '2020-08-30')]  # 构建仅含头三个月数据的数据集
df_sales_3m.reset_index(drop=True)  # 重置索引

df_user_LTV = pd.DataFrame(df_sales['用户码'].unique())  # 生成以用户码为主键的结构
df_user_LTV.columns = ['用户码']  # 设定字段名
df_user_LTV.head()  # 显示头几行数据
df_R_value = df_sales_3m.groupby('用户码').消费日期.max().reset_index()  # 找到每个用户的最近消费日期，构建df_R_value对象
df_R_value.columns = ['用户码', '最近购买日期']  # 设定字段名
df_R_value['R值'] = (df_R_value['最近购买日期'].max() - df_R_value['最近购买日期']).dt.days  # 计算最新日期与上次消费日期的天数
df_user_LTV = pd.merge(df_user_LTV, df_R_value[['用户码', 'R值']], on='用户码')  # 把上次消费距最新日期的天数（R值）合并至df_user结构
df_F_value = df_sales_3m.groupby('用户码').消费日期.count().reset_index()  # 计算每个用户消费次数，构建df_F_value对象
df_F_value.columns = ['用户码', 'F值']  # 设定字段名
df_user_LTV = pd.merge(df_user_LTV, df_F_value[['用户码', 'F值']], on='用户码')  # 把消费频率(F值)整合至df_user结构
df_M_value = df_sales_3m.groupby('用户码').总价.sum().reset_index()  # 计算每个用户三个月消费总额，构建df_M_value对象
df_M_value.columns = ['用户码', 'M值']  # 设定字段名
df_user_LTV = pd.merge(df_user_LTV, df_M_value, on='用户码')  # 把消费总额整合至df_user结构
print(df_user_LTV)  # 显示用户表结构

df_user_1y = df_sales.groupby('用户码')['总价'].sum().reset_index()  # 计算每个用户整年消费总额，构建df_user_1y对象
df_user_1y.columns = ['用户码', '年度LTV']  # 设定字段名
df_user_1y.head()  # 显示头几行数据
df_LTV = pd.merge(df_user_LTV, df_user_1y, on='用户码', how='left')  # 构建整体LTV训练数据集
print(df_LTV)  # 显示df_LTV

X = df_LTV.drop(['用户码', '年度LTV'], axis=1)  # 特征集
print(X.head())  # 显示特征集

y = df_LTV['年度LTV']  # 标签集
print(y.head())  # 显示标签集

# 先拆分训练集和其它集
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.7, random_state=36)
# 再把其它集拆分成验证集和测试集
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=36)

model_lr = LinearRegression()  # 创建线性回归模型
model_dtr = DecisionTreeRegressor()  # 创建决策树回归模型
model_rfr = RandomForestRegressor()  # 创建随机森林回归模型

model_lr.fit(X_train, y_train)  # 拟合线性回归模型
model_dtr.fit(X_train, y_train)  # 拟合决策树模型
model_rfr.fit(X_train, y_train)  # 拟合随机森林模型

y_valid_preds_lr = model_lr.predict(X_valid)  # 用线性回归模型预测验证集
y_valid_preds_dtr = model_dtr.predict(X_valid)  # 用决策树模型预测验证集
y_valid_preds_rfr = model_rfr.predict(X_valid)  # 用随机森林模型预测验证集

print('验证集上的R平方分数-线性回归: %0.4f' % r2_score(y_valid, y_valid_preds_lr))
print('验证集上的R平方分数-决策树: %0.4f' % r2_score(y_valid, y_valid_preds_dtr))
print('验证集上的R平方分数-随机森林: %0.4f' % r2_score(y_valid, y_valid_preds_rfr))

y_test_preds_rfr = model_rfr.predict(X_test)  # 用模型预随机森林模型预测验证集
plt.scatter(y_test, y_test_preds_rfr)  # 预测值和实际值的散点图
plt.plot([0, max(y_test)], [0, max(y_test_preds_rfr)], color='gray', lw=1, linestyle='--')  # 绘图
plt.xlabel('实际值')  # X轴
plt.ylabel('预测值')  # Y轴
plt.title('实际值 vs. 预测值')  # 标题
plt.show()
