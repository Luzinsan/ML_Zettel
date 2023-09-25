
> Алгоритм K-средних разбивает n наблюдений на k кластеров, где каждое наблюдение принадлежит кластеру с ближайшим средним значением, служащим прототипом кластера. ![[Pasted image 20230925220530.png|500]]

- [[Density-Based Method]]
	- [[DBSCAN – Density-Based Spatial Clustering of Applications with Noise]]
	- [[OPTICS - Ordering Points to Identify Clustering Structure]]
- [[BIRCH – Balanced Iterative Reducing and Clustering using Hierarchies]]
- [[Hierarchical Clustering]]

- _CURE (Clustering Using Representatives)_
- _BIRCH (Balanced Iterative Reducing Clustering and using Hierarchies)_

>> **Методы секционирования - Partitioning Methods**: 
> Эти методы разбивают объекты на k кластеров, и каждый раздел образует один кластер. Этот метод используется для оптимизации функции подобия по объективному критерию, например, когда расстояние является основным параметром: K-means ([[K-Means Clustering]]), [[CLARANS]] и т.д.

>> Методы, основанные на сетке
> Пространство данных формулируется в виде конечного числа ячеек, которые образуют структуру, подобную сетке. 
> Все операции кластеризации, выполняемые на этих сетках, выполняются быстро и не зависят от количества объектов данных
> Например, [[STING]], [[wave cluster]], [[CLIQUE]] и т.д.