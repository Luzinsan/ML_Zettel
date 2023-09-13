- Необходимо для __упорядоченных__ __категориальных__ признаков: уровень дохода (low, medium, high) преобразуют в {0,1,2}.
- Этот тип кодирования сохраняет значимости элементов, таким образом, что, тем выше приоритет элемента, тем больше его вес.
- Суть: значения категориального признака конвертируются в числовой тип, сохраняя при этом приоритет. 
- Наивысший приоритет у значения 0.
##### Пример: датасет iris и его признак породы цветка (species)
```python{pre}
# Import libraries
import numpy as np
import pandas as pd

# Import dataset
df = pd.read_csv('../../data/Iris.csv')

df['species'].unique()
```

```python
# Import label encoder
from sklearn import preprocessing

# label_encoder object knows
# how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
df['species']= label_encoder.fit_transform(df['species'])

df['species'].unique()

```

### Ограничения кодирования меток
- Так как каждому классу данных присваивается уникальный номер (начиная с 0), это может привести к возникновению проблем приоритета во время обучения модели. Метка с высоким рангом (0) имеет более высокий приоритет, чем метка с более низким значением. 
- Поэтому кодирование меток нужно использовать осознанно и только на ранжированных признаках.