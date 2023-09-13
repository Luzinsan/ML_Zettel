Линейная регрессия страдает от переобучения и не может рассматривать коллинеарные данные, когда в наборе данных много объектов, и даже некоторые из них не имеют отношения к прогнозирующей модели. Такая модель не может обобщать данные - из-за высокой дисперсии. 
Чтобы справится с этими проблемами включается регуляризация норм L-2 и L-1, чтобы одновременно получить преимущества как Ridge, так и Lasso.   
Оно выполняет отбор признаков, а также упрощает гипотезу.

Модифицированная функция затрат (cost function): $$\frac{1}{m} [\sum^{m}_{l=1}(y^{(i)} - h(x^{(i)}))^2 + \lambda_1\sum^n_{j=1}w_j+\lambda\sum^{n}_{j=1}w^2_j]$$