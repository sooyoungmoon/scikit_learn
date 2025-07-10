# Classification - hand-written-digits

## Goal: 주어진 이미지를 0~9까지의 정수로 분류

### Steps

1. 모듈 임포트
```python
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
```

2. dataset 로딩
```python
digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
```

3. 이미지를 1D array로 변환
```python
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
```
Q.  이미지를 1D array로 변환하는 이유?   
A. scikit-learn에서는 입력 데이터를 (n_samples, # of features)의 2차원 행렬로 간주하기 때문!

4. ML 모델 객체 생성
```python
# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)
```
support vector classifier: classification 문제에 사용되는 알고리즘인 support vector machine (SVM)을 이용하는 모델 
gamma: 

5. 학습 데이터와 테스트 데이터 구분
```python
# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)
```
학습 데이터 (x_train, y_train)와 테스트 데이터 (x_test, y_test)를 각각 50% 비율로 구분 

6. 모델 학습 
```python
clf.fit(X_train, y_train)
```
모델 학습: (입력과 레이블) 사용

6. 추론 
```python
predicted = clf.predict(X_test)
```

7. 추론 결과 시각화
```python
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
```
![](https://scikit-learn.org/stable/_images/sphx_glr_plot_digits_classification_002.png))

8. 모델 성능 평가 (분류 성능)
```python
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
```

```
Classification report for classifier SVC(gamma=0.001):
              precision    recall  f1-score   support

           0       1.00      0.99      0.99        88
           1       0.99      0.97      0.98        91
           2       0.99      0.99      0.99        86
           3       0.98      0.87      0.92        91
           4       0.99      0.96      0.97        92
           5       0.95      0.97      0.96        91
           6       0.99      0.99      0.99        91
           7       0.96      0.99      0.97        89
           8       0.94      1.00      0.97        88
           9       0.93      0.98      0.95        92

    accuracy                           0.97       899
   macro avg       0.97      0.97      0.97       899
weighted avg       0.97      0.97      0.97       899
```
- accuracy: 모든 class 에 대한 예측 중 실제와 일치한 예측의 비율 (single value for a test)
- precision: 각 class 마다 해당 class에 대한 예측이 일치한 비율
- recall: 각 class 마다 해당 class 데이터가 올바르게 예측된 비율
