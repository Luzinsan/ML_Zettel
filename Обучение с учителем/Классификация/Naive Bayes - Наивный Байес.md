Naive Bayes - Наивный Байес - основан на теореме Байеса ([[Bayes Theorem]])
Наивный - назван так из-за того, что он предполагает независимость между каждой парой признаков в данных.
Так как наивность предполагает независимость для каждой пары $x_1, x_2, ..., x_n$, то формула меняется на: $$P(y|x_1, ..., x_n) = \frac{P(y)\prod^n_{i=1}{(P(x_i|y))}}{P(x_1,...,x_n)}$$
- Вводя пропорциональность путём удаления $P(y|x_1,...,x_n)$ (так как это константа), получаем: $$P(y|x_1,...,x_n)\alpha(P(y)\prod^n_{i=1}{(P(x_i|y))})$$
- Следовательно, метка класса определяется так: $$\hat{y} = \underset{y}{argmax}{P(y)}\prod^n_{i=1}{(P(x_i|y))}$$
- P(y) - это относительная частота метки класса y в обучающем датасете: $$P(x_i|y)=\frac{1}{\sqrt{2\pi\sigma}}exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma\frac{2}{y}}\right)$$
```run-python
# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# loading the iris dataset
iris = datasets.load_iris()

# X -> features, y -> label
X = iris.data
y = iris.target

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# training a Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(X_train, y_train)
gnb_predictions = gnb.predict(X_test)

# accuracy on X_test
accuracy = gnb.score(X_test, y_test)
print(accuracy)

# creating a confusion matrix
cm = confusion_matrix(y_test, gnb_predictions)

```
