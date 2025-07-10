# Classification - nearest neighbor

## Goal: classification 기법의 하나인 nearest neighbor 알고리즘을 이용하여 iris dataset 분류

### Steps

1. 모듈 임포트 & iris dataset loading
```python
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris(as_frame=True)
X = iris.data[["sepal length (cm)", "sepal width (cm)"]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
```

