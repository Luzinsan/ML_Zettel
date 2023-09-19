- Простой, но мощный (основанный на [[Decision Tree]]) алгоритм классификации, который разделяет датасет на малые подмножества, основываясь на значениях признаков и сооружает модель, подобную дереву, которая в дальнейшем используется для классификации.
- Классификатор решающего дерева представляет собой набор вопросов для датасета относительно атрибутов/признаков.
- Алгоритм можно представить как бинарное дерево, где в корне и каждом внутреннем узле задаётся некоторый вопрос к данным, после чего данные разделяются на подмножество данных и получают некоторые характеристики.
- Листья дерева представляют собой классы, на которые и разделились данные.

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

# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)
print(dtree_predictions)

# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)

```