- Это техника, которая используется для представления категориальных переменных в модели машинного обучения.
### Преимущества и недостатки
#### Преимущества
1. позволяет использовать категориальные переменные в моделях, которые требуют только количественные переменные.
2. Может улучшить производительность модели, предоставляя больше информации о категориальных переменных.
3. Может помочь избежать проблемы порядка (ordinality), которая появиться, когда категориальные переменные имеют натуральный порядок.
#### Недостатки
1. Приводит к повышению размерности, так как разделение колонки создает каждую категорию в отдельной переменной. Это делает модель более сложной и медленной для обучения.
2. Приводит к разряжению данных, и большинство наблюдений будут иметь значение 0.
3. Может привести к разряжению данных, особенно если есть много категорий в переменной и размер выборки относительно мал.
4. [[Dummy variable trap in Regression Models]]
#### В процессе One Hot Encoding полезно использовать:
- `df['column'].unique()`
- `df['column'].value_counts()`
### One-Hot Encoding категориальных признаков с помощью Pandas
`one_hot_encoded_data = pd.get_dummies(data, columns = ['Remarks', 'Gender'])` - таким способом признаки кодируются на месте (inplace)
### One-Hot Encoding с помощью Sci-kit - OneHotEncoder - для кодирования категориальных и числовых признаков в бинарный вектор
- Для этого кодирования признаки должны быть сначала закодированы Label Encoder, а затем уже OneHotEncoder - на выходе получатся действительные значения в столбцах.
- Преобразовываем метод enc.fit_transform() в массив, потому что метод fit_transform возвращает разреженную матрицу SpiPy, поэтому преобразование в массив  сэкономит место, когда у нас есть огромное количество категориальных переменных.
```python
from sklearn.preprocessing import OneHotEncoder

# Converting type of columns to category
data['Gender'] = data['Gender'].astype('category')
data['Remarks'] = data['Remarks'].astype('category')

# Assigning numerical values and storing it in another columns
data['Gen_new'] = data['Gender'].cat.codes
data['Rem_new'] = data['Remarks'].cat.codes

# Create an instance of One-hot-encoder
enc = OneHotEncoder()

# Passing encoded columns
enc_data = pd.DataFrame(enc.fit_transform(
	data[['Gen_new', 'Rem_new']]).toarray())

# Merge with main
New_df = data.join(enc_data)
```
- ![[Pasted image 20230808104245.png]]