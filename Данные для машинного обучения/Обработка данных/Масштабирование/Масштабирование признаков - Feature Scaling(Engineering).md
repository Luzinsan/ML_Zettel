- Это техника конвертации данных в стандартную форму или структуру, представленных в наборе данных в фиксированном промежутке, которая может быть легко обработана алгоритмом или моделью для анализа.
### Необходимость в масштабировании признаков
1. Feature Normalization - Нормализация признаков - нужна для того, что никакой из признаков не доминировал над любым другим. Осуществляется посредством установления признаков в сопоставимые диапазоны так, чтобы они были эквивалентны и сравнимы между собой.
2. Повышает производительность алгоритма и делает так, чтобы он выполнялся лучше и быстрее сходился.
3. Предотвращает числовую нестабильность - при вычислении расстояний или матричных операциях, когда наличие объектов с радикально отличающимися масштабами может привести к проблемам переобучения или недообучения.
4. Обеспечивает равный вклад каждого признака для избежания искажения/перекосов результатов обучения.
---
## Min-Max Scaling:
- MinMaxScaler масштабирует данные так, чтобы значения признаков находились в промежутке [0,1]  (но также чувствителен к выбросам)$$X_{scaled} = \frac{X_i - X_{min}}{X_{max}-X_{min}}$$
- Необходима для признаков, имеющих различные масштабы, которые далее будут поданы в алгоритмы, чувствительные к масштабам: алгоритм ближайших k соседей (k-nearest neighbors), нейронные сети (neural networks).
- MinMaxScaler имеется в библиотеке scikit-learn.
```run-python
# initialising the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# learning the statistical parameters for each of the data and transforming
rescaledX = scaler.fit_transform(X)
scaled_df = pd.DataFrame(rescaledX,
                         columns=df.columns)
print(scaled_df.head())	
```
Другой пример:
```python
from sklearn.preprocessing import MinMaxScaler

# initialising the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Numerical columns
num_col_ = [col for col in X.columns if X[col].dtype != 'object']
x1 = X
# learning the statistical parameters for each of the data and transforming
x1[num_col_] = scaler.fit_transform(x1[num_col_])
x1.head()
```
---
## Нормализация (Normalization):
$$X_{scaled}=\frac{X_i - X_{mean}}{X_{max}-X_{min}}$$
```python
from sklearn.preprocessing import Normalizer
 
scaler = Normalizer()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data,
                         columns=df.columns)
print(scaled_df.head())
```
---
## Стандартизация (Standardization) - Z-score scaling:
- Это техника преобразования признаков, которые уже имеют гауссово/нормальное распределение (Gaussian distribution) c различными значениями среднего (means, $\mu$) и стандартного отклонения (standard deviation, $\sigma$) в гауссово распределение с $\mu$=0 и $\sigma$=1 $$X_{scaled} = \frac{X_i - \mu}\sigma$$
- Необходима для всех признаков с гауссовым распределением
```run-python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data,
						columns=df.columns)
print(scaled_df.head())

```
---
## Надежное масштабирование - Robust Scaling
- Использует два важных статистических измерения: медиану ($X_{median}$) и межквартильный интервал ($IQR = Q_3 - Q_1$) $$X_{scaled}=\frac{X_i - X_{median}}{IQR }$$
- Используется для масштабирования количественных признаков путем приведения их к обычному масштабу, при этом он гораздо менее чувствителен к наличию __выбросов__ по сравнению с остальными стратегиями масштабирования.
- Устойчивость к распределению фактов: RobustScaler не делает никаких предположений относительно распределения лежащих в основе фактов, что делает его более универсальным и применимым к особым видам признаков. Позволяет легко обрабатывать признаки не в гауссовом распределении.
- Сохраняет целостность фактов: сохраняет _ранг_ и _порядок_ записей, поскольку использует общую статистику на основе рангов (медиана), что делает ее подходящей для порядковых данных или информации со значимыми порядковыми отношениями.
- Обрабатывает перекошенные факты: поскольку он полностью основан на процентилях (медиана и IQR), которые в меньшей степени подвержены экстремальным значениям, что делает его подходящим для наборов данных с перекошенными распределениями.
- Может работать с разреженными матрицами: полезно при работе с наборами данных большой размерности, обладающими множеством возможностей.

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data,
						columns=df.columns)
print(scaled_df.head())

```

```run-python
""" PART 1: Importing Libraries """
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
matplotlib.style.use('ggplot')

""" PART 2: Making the data distributions """

x = pd.DataFrame({

	# Distribution with lower outliers
	'x1': np.concatenate([np.random.normal(20, 1, 2000), 
						  np.random.normal(1, 1, 20)]),
	# Distribution with higher outliers
	'x2': np.concatenate([np.random.normal(30, 1, 2000), 
						  np.random.normal(50, 1, 20)]),
})

""" PART 3: Scaling the Data """
scaler = preprocessing.RobustScaler()
robust_scaled_df = scaler.fit_transform(x)
robust_scaled_df = pd.DataFrame(robust_scaled_df, columns=['x1', 'x2'])

""" PART 4: Visualizing the impact of scaling """
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 5))
ax1.set_title('Before Scaling')
sns.kdeplot(x['x1'], ax=ax1)
sns.kdeplot(x['x2'], ax=ax1)
ax2.set_title('After Robust Scaling')
sns.kdeplot(robust_scaled_df['x1'], ax=ax2)
sns.kdeplot(robust_scaled_df['x2'], ax=ax2)
#fig.show()
```
---
## Абсолютное максимальное масштабирование - Absolute Maximum Scaling
- Значения признака каждой записи лежат в промежутке [-1,1]
- Используется нечасто, так как чувствителен к выбросам.
1. Выбирается максимальное абсолютное значение среди всех наблюдений некоторого признака.
2. Значения этого признака у каждого наблюдения делятся на это максимальное значение: $$X_{scaled}=\frac{X_i - max(|X|)}{max(|X|)}$$
```python
import numpy as np
max_vals = np.max(np.abs(df))
df = (df - max_vals) / max_vals
```