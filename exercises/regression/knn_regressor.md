# Regression - Predicting Diabetes progress using KNN regression
## Goal: 당뇨 환자들 관련 정보로부터 당뇨 진행 정도 예측 

### 1. 모듈 임포트
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
```
load_diabetes(): diabetes 데이터셋 로딩 
[Diabetes 데이터셋 정보](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)

샘플 수: 442  
속성 수: 10  
타겟 값 (예측 값): 진단 1년 뒤 당뇨 진행 상황 (정수)  

속성 목록  
- age  
- sex  
- bmi body mass index
- bp average blood pressure
- s1 tc, total serum cholesterol
- s2 ldl, low-density lipoproteins
- s3 hdl, high-density lipoproteins
- s4 tch, total cholesterol / HDL
- s5 ltg, possibly log of serum triglycerides level
- s6 glu, blood sugar level
