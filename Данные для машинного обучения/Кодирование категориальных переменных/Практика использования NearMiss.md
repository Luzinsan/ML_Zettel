- Проверка свойств и использование алгоритма NearMiss:
```python
print("Before Undersampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before Undersampling, counts of label '0': {} \n".format(sum(y_train == 0)))

# apply near miss
from imblearn.under_sampling import NearMiss
nr = NearMiss()

X_train_miss, y_train_miss = nr.fit_sample(X_train, y_train.ravel())

print('After Undersampling, the shape of train_X: {}'.format(X_train_miss.shape))
print('After Undersampling, the shape of train_y: {} \n'.format(y_train_miss.shape))

print("After Undersampling, counts of label '1': {}".format(sum(y_train_miss == 1)))
print("After Undersampling, counts of label '0': {}".format(sum(y_train_miss == 0)))
```
- Алгоритм уменьшил количество наблюдений мажорного класса так, что его количество стало равным количеству наблюдений минорного класса
![[Pasted image 20230808121632.png]]
- Обучение и предсказание:
```python
# train the model on train set
lr2 = LogisticRegression()
lr2.fit(X_train_miss, y_train_miss.ravel())
predictions = lr2.predict(X_test)

# print classification report
print(classification_report(y_test, predictions))
```
- Recall второго класса увеличился, однако у первого класса значительно упал, так что в данном случае алгоритм NearMiss не подходит: ![[Pasted image 20230808122627.png]]