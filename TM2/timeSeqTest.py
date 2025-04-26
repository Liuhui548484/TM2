# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.tsa.api import VAR
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tools.eval_measures import rmse
#
# # 读取数据
# data = pd.read_csv('./data(2.csv', parse_dates=['date'], index_col='date')
#
# # 检查数据是否平稳
# def adf_test(series, title=''):
#     result = adfuller(series)
#     print(f'ADF Statistic for {title}: {result[0]}')
#     print(f'p-value: {result[1]}')
#     if result[1] <= 0.05:
#         print("Data is stationary")
#     else:
#         print("Data is non-stationary")
#
# for col in data.columns:
#     adf_test(data[col], col)
#
# # 如果数据不平稳，进行差分
# data_diff = data.diff().dropna()
#
# # 再次检查差分后的数据是否平稳
# for col in data_diff.columns:
#     adf_test(data_diff[col], f'{col} (differenced)')
#
# # 拟合VAR模型
# model = VAR(data_diff)
# results = model.fit(maxlags=15, ic='aic')
# print(results.summary())
#
# # 预测
# lag_order = results.k_ar
# forecast_input = data_diff.values[-lag_order:]
# fc = results.forecast(y=forecast_input, steps=10)
# forecast = pd.DataFrame(fc, columns=data_diff.columns)
#
# # 反差分以获得原始尺度的预测
# def inverse_diff(original, diff):
#     return diff + original.shift(1)
#
# forecast_orig = forecast.copy()
# for col in data.columns:
#     forecast_orig[col] = inverse_diff(data[col], forecast[col])
#
# # 可视化预测结果
# plt.figure(figsize=(12, 6))
# for col in data.columns:
#     plt.plot(data[col], label=f'Actual {col}')
#     plt.plot(forecast_orig.index, forecast_orig[col], label=f'Forecast {col}', linestyle='--')
# plt.legend()
# plt.show()
#





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse

# 读取数据
data = pd.read_csv('data(2.csv', parse_dates=['date'], index_col='date')

# 检查数据是否为常数
def is_constant(series):
    return series.nunique() == 1

for col in data.columns:
    if is_constant(data[col]):
        print(f'Column {col} is constant and will be dropped.')
        data = data.drop(columns=[col])

# 检查数据是否平稳
def adf_test(series, title=''):
    result = adfuller(series)
    print(f'ADF Statistic for {title}: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("Data is stationary")
    else:
        print("Data is non-stationary")

for col in data.columns:
    adf_test(data[col], col)

# 如果数据不平稳，进行差分
data_diff = data.diff().dropna()

# 再次检查差分后的数据是否平稳
for col in data_diff.columns:
    adf_test(data_diff[col], f'{col} (differenced)')

# 检查数据中的线性依赖关系
def check_linear_dependency(data):
    corr_matrix = data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.95:
                print(f'Warning: High correlation between {corr_matrix.columns[i]} and {corr_matrix.columns[j]}')
                # 检查列是否存在，然后再删除
                if corr_matrix.columns[i] in data.columns:
                    data = data.drop(columns=[corr_matrix.columns[i]])
                elif corr_matrix.columns[j] in data.columns:
                    data = data.drop(columns=[corr_matrix.columns[j]])
    return data

data_diff = check_linear_dependency(data_diff)

# 拟合VAR模型
model = VAR(data_diff)
results = model.fit(maxlags=15, ic='aic')
print(results.summary())

# 预测
lag_order = results.k_ar
forecast_input = data_diff.values[-lag_order:]
fc = results.forecast(y=forecast_input, steps=10)
forecast = pd.DataFrame(fc, columns=data_diff.columns)

# 反差分以获得原始尺度的预测
def inverse_diff(original, diff):
    return diff + original.shift(1)

forecast_orig = forecast.copy()
for col in data.columns:
    forecast_orig[col] = inverse_diff(data[col], forecast[col])

# 可视化预测结果
plt.figure(figsize=(12, 6))
for col in data.columns:
    plt.plot(data[col], label=f'Actual {col}')
    plt.plot(forecast_orig.index, forecast_orig[col], label=f'Forecast {col}', linestyle='--')
plt.legend()
plt.show()