Для создания датасетов с помощью scikit-learn в python нужно импортировать _sklearn.datasets.samples_generator_.
```python{pre}
# matplotlib for plotting
from matplotlib import pyplot as plt
from matplotlib import style
import pandas as pd

style.use("fivethirtyeight")
```

---
##### Мультиклассовая классификация
---
> make_blobs
```run-python
from sklearn.datasets import make_blobs
# Creating Test DataSets using sklearn.datasets.make_blobs
X, y = make_blobs(n_samples = 1000, centers = 3,
			cluster_std = 1, n_features = 2)

plt.scatter(X[:, 0], X[:, 1], s = 10, c=y)
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
plt.clf()

```
---
>make_classification - для создания сбалансированного датасета, учитывая информативные, избыточные признаки и кол-во классов.
```run-python
from sklearn.datasets import make_classification
 
# generate 2d classification dataset
X, y = make_classification(n_samples = 100,
                           n_features=2,
                           n_redundant=0,
                           n_informative=2,
                           n_repeated=0,
                           n_classes =3,
                           n_clusters_per_class=1)
 
# Plot the generated datasets
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
plt.clf()
```
---
>make_multilabel_classification - для генерации данных со многими метками
```run-python
from sklearn.datasets import make_multilabel_classification
# Generate 2d classification dataset
X, y = make_multilabel_classification(n_samples=500, n_features=2,
                                      n_classes=2, n_labels=2,
                                      allow_unlabeled=True,
                                      random_state=44)
# create pandas dataframe from generated dataset
df = pd.concat([pd.DataFrame(X, columns=['X1', 'X2']),
                pd.DataFrame(y, columns=['Label1', 'Label2'])],
               axis=1)
print(df.head())
 
# Plot the generated datasets
plt.scatter(df['X1'], df['X2'], c=df['Label2'])
plt.show()
plt.clf()
```
##### Бинарная классификация
---
> make_moons - сферическая граница разделения
```run-python
# Creating Test DataSets using sklearn.datasets.make_moon
from sklearn.datasets import make_moons

X, y = make_moons(n_samples = 200, shuffle=True, 
				  noise = 0.1, random_state=42)
plt.scatter(X[:, 0], X[:, 1], s = 10, c=y)
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
plt.clf()

```

---
---
> make_circles
```run-python
# Creating Test DataSets using sklearn.datasets.make_circles
from sklearn.datasets import make_circles

style.use("fivethirtyeight")

X, y = make_circles(n_samples = 200, shuffle=True, noise = 0.1, random_state=42)
plt.scatter(X[:, 0], X[:, 1], s = 10, c=y)
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
plt.clf()
```

##### Регрессия
> make_regression
```run-python
from sklearn.datasets import make_regression

X, y = make_regression(n_samples = 50, n_features=1,noise=20, random_state=23)

# Plot the generated datasets
plt.scatter(X, y)
plt.show()
plt.clf()
```
---
> make_sparce_uncorrelated - генерация разряжённого некоррелированного датасета для нескольких меток
```run-python
from sklearn.datasets import make_sparse_uncorrelated
n_features = 6
X, y = make_sparse_uncorrelated(n_samples = 100, n_features=n_features, random_state=23)
# Plot the generated datasets
plt.figure(figsize=(16,16))

for i in range(n_features):
    plt.subplot(n_features//2,n_features//2, i+1)
    plt.scatter(X[:,i], y)
    plt.xlabel('X'+str(i+1))
    plt.ylabel('Y')
plt.show()
plt.clf()
```
---
> make_friedman2 - генерация датасета для нескольких меток методом Фридмана
```run-python
from sklearn.datasets import make_friedman2

X, y = make_friedman2(n_samples = 100, random_state=23)
print(X[:5])
# Plot the generated datasets
plt.figure(figsize=(12,10))
for i in range(4):
    plt.subplot(2,2, i+1)
    plt.scatter(X[:,i], y)
    plt.xlabel('X'+str(i+1))
    plt.ylabel('Y')
plt.show()
plt.clf()
```
#### [[Преимущества и недостатки создания датасетов с помощью sklearn]]

---
---