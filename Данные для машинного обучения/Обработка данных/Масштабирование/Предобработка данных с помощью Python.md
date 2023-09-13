Предобработка данных - это техника, которая используется для преобразования необработанных исходных данных в чистый набор данных. Как только данные были собраны из различных источников в грязном формате, их нужно обработать, так как они не подходят для анализа.
![[Pasted image 20230801182446.png]]
##### Требования по обработке:
- Форматы данных должны находиться в правильной форме. Некоторым моделям машинного обучения необходим специфический формат данных: алгоритм рандомного леса (Random Forest) не поддерживает нулевые значения, следовательно, их нужно убрать из исходного набора данных.
- Универсальность. Данные должны быть отформатированы таким образом, чтобы они подходили для разных моделей машинного и глубокого обучения, для того чтобы в последующем выбрать наилучший из моделей. 
---
### Review
- Импорт библиотек
- Загрузка датасета
- Статистический анализ: .describe()
- Проверка на выбросы .boxplot() и удаление выбросов np.percentile()
- Обнаружение корреляции: df.corr() и heatmap()
- Анализ пропорциональности целевых признаков: plt.pie()
- Разделение признаков на входные и целевые 
- Масштабирование признаков - Feature Scaling: стандартизация или нормализация.
---
#### Шаги по предобработке данных:
1. Импортирование необходимых библиотек. Чаще всего используется:
```python{pre}
import pandas as pd
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib
```
---
2. Загрузка датасета: [диабет](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) и проверка информации об этом датасете
```run-python
df = pd.read_csv('~/Downloads/diabetes.csv')
print(df.head())

df.info()
```
Проверка null значений:
```run-python
# check the null values
print(df.isnull().sum())
```
---
3. Статистический анализ. Используется df.describe(), который даёт описательный обзор датасета. В нём можно увидеть первые намёки на наличие выбросов.
```run-python
# descriptive overview of the dataset

print(df.describe())

```
---
4. Проверка выбросов. Из примера можно увидеть, что почти у каждого признака есть хоть какие-то выбросы.
```run-python
# Box Plots
fig, axs = plt.subplots(9,1,dpi=95, figsize=(7,17))
i = 0
for col in df.columns:
	axs[i].boxplot(df[col], vert=False)
	axs[i].set_ylabel(col)
	i+=1
plt.show()
```
Удаление выбросов
```run-python
for feature in ['Insulin','Pregnancies','Age','Glucose',
				'BloodPressure','BMI','DiabetesPedigreeFunction']:
	# Identify the quartiles
	q1, q3 = np.percentile(df[feature], [25, 75])
	# Calculate the interquartile range
	iqr = q3 - q1
	# Calculate the lower and upper bounds
	lower_bound = q1 - (1.5 * iqr)
	upper_bound = q3 + (1.5 * iqr)
	# Drop the outliers
	clean_data = df[(df[feature] >= lower_bound)
					& (df[feature] <= upper_bound)]

```
---
5.  Обнаружение взаимосвязей (корреляции)
```run-python
corr = df.corr()

plt.figure(dpi=130)
sns.heatmap(df.corr(), annot=True, fmt= '.2f')
plt.show()

```
- Сравнение одного признака (например, исхода - болеет ли пациент диабетом или нет) со всеми остальными признаками:
```run-python

print(corr['Outcome'].sort_values(ascending = False))
```
- Проверка пропорциональности целевых признаков:
```run-python

plt.pie(df.Outcome.value_counts(),
		labels= ['Diabetes', 'Not Diabetes'],
		autopct='%.f', shadow=True)
plt.title('Outcome Proportionality')
plt.show()

```
6. Разделение на независимые признаки и целевые переменные
```run-python
# separate array into input and output components
X = df.drop(columns =['Outcome'])
Y = df.Outcome

```
7. [[Масштабирование признаков - Feature Scaling(Engineering)]]