#SupervisedLearning 

---
- Модели или алгоритму предоставляются примеры входов и их желаемых выходов. Тогда она находит шаблон и взаимосвязи межу входными и выходными данными. Цель состоит в том, чтобы выявить основные правила, которые сопоставляют вход выходу. Тренировочный процесс продолжается до тех пор, пока модель не достигнет желаемого уровня точности на тренировочных данных (training data). Некоторые реальные примеры:
	- Классификация картинок: тренировка проходит на основе картинок и лейблов. И когда модели на вход подадут новую картинку, она распознает её и классифицирует.
	- Прогнозирование/Регрессия рынка: тренировка проходит на исторических данных рынка, и далее, при запросе на предсказание новой цены, модель предскажет её стоимость.
---------------------------------------------------
# Типы алгоритмов контролируемого обучения
![[Pasted image 20230727100938.png]]
0. [[Data Mining - Интеллектуальный анализ данных]]
	- [[Классификация - Classification]]
2. [[Регрессия - Regression]]
	1. [[Линейная регрессия - Linear Regression]]
	3. [[Polynomial Regression]]
	4. [[Stepwise Regression]]
	5. [[Decision Tree Regression]]
	6. [[Random Forest Regression]]
	7. [[Support Vector Regression (SVR)]]
	8. [[Ridge Regression]]
	9. [[Lasso Regression]]
	10. [[ElasticNet Regression]]
3. [[Линейная классификация - Linear Classification]]
	1. [[Логистическая регрессия - Logistic Regression]]
	2. [[Support Vector Machines having kernel = ‘linear’ - Метод опорных векторов с линейным ядром]]
	3. [[Single-layer Perceptron - Однословный перцептрон]]
	4. [[Stochastic Gradient Descent (SGD) - Стохастических градиентный спуск]]
5. [[Нелинейная классификация - Non-linear Classifiers]]
	1. [[K-Nearest Neighbors - Алгоритм K-ближайших соседей]]
	2. [[Kernel SVM - Ядро метода опорных векторов (МОВ)]]
	3. [[Naive Bayes - Наивный Байес]]
	4. [[Decision Tree - Дерево решений]]
	5. [[Ensemble learning classifiers - Классификаторы коллективного обучения]]
		1. [[Random Forest - Рандомный лес]] 
		2. [[AdaBoost]]
		3. [[Bagging Classifier]]
		4. [[Voting Classifier]]
		5. [[ExtraTrees Classifier]]
	6. [[Multi-layer Artificial Neural Network - Многослойная искусственная нейронная сеть]]