이 튜토리얼에서는 Seaborn의 타이타닉 데이터셋을 사용하여 **K-최근접 이웃(K-Nearest Neighbors, KNN)** 알고리즘으로 승객의 생존 여부를 예측하는 방법을 단계별로 알아봅니다.

KNN은 특정 데이터 포인트의 클래스를 예측할 때, 그 데이터와 가장 가까운 'K'개의 데이터 포인트를 보고 다수결로 클래스를 결정하는 간단하면서도 강력한 분류 알고리즘입니다.

-----

### 1\. Setup: Import Libraries

먼저 데이터 분석, 시각화, 모델링에 필요한 라이브러리를 가져옵니다.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set plot style
plt.style.use('ggplot')
```

-----

### 2\. Load and Explore Data (EDA)

Seaborn 라이브러리에서 타이타닉 데이터셋을 불러온 후, 데이터의 기본 구조와 통계를 확인합니다.

```python
# Load the dataset
titanic = sns.load_dataset('titanic')

# Display basic info and first 5 rows
print(titanic.info())
titanic.head()
```

데이터 시각화를 통해 변수 간의 관계를 탐색합니다. 

```python
# Plotting survival counts
plt.figure(figsize=(8, 5))
sns.countplot(x='survived', data=titanic, hue='survived', palette='viridis', legend=False)
plt.title('Survival Count (0 = No, 1 = Yes)')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Plotting survival counts by gender
plt.figure(figsize=(8, 5))
sns.countplot(x='survived', hue='sex', data=titanic, palette='plasma')
plt.title('Survival Count by Gender')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()
```

-----

### 3\. Data Preprocessing

KNN 모델을 사용하기 위해 데이터를 깨끗하게 만들고, 기계가 이해할 수 있는 숫자 형태로 변환해야 합니다.

#### 3.1. Handle Missing Values

결측치가 있는 'age', 'deck', 'embarked' 열을 처리합니다.

  * **age**: 평균 나이로 채웁니다.
  * **deck**: 결측치가 너무 많으므로 열을 삭제합니다.
  * **embarked**, **embark\_town**: 결측치가 있는 행을 삭제합니다.

<!-- end list -->

```python
# Fill missing age values with the mean
titanic['age'] = titanic['age'].fillna(titanic['age'].mean())

# Drop the 'deck' column
titanic.drop('deck', axis=1, inplace=True)

# Drop rows with missing 'embarked' or 'embark_town' values
titanic.dropna(inplace=True)
```

#### 3.2. Convert Categorical Features

'sex', 'embarked' 같은 범주형 변수를 원-핫 인코딩을 통해 숫자형으로 변환합니다.

```python
# Convert categorical variables into dummy/indicator variables
sex_dummies = pd.get_dummies(titanic['sex'], drop_first=True, dtype=int)
embark_dummies = pd.get_dummies(titanic['embarked'], drop_first=True, dtype=int)

# Drop original and unnecessary columns
titanic.drop(['sex', 'embarked', 'who', 'class', 'adult_male', 'alive', 'embark_town'], axis=1, inplace=True)

# Concatenate the new dummy variables
titanic = pd.concat([titanic, sex_dummies, embark_dummies], axis=1)
```

#### 3.3. Feature Scaling

KNN은 데이터 포인트 간의 거리를 기반으로 동작하므로, 모든 변수(feature)가 동일한 스케일을 갖도록 **정규화**하는 것이 매우 중요합니다. `StandardScaler`를 사용해 모든 피처를 표준 정규 분포를 따르도록 변환합니다.

```python
# Define features (X) and target (y)
X = titanic.drop('survived', axis=1)
y = titanic['survived']

# Create a scaler object
scaler = StandardScaler()

# Fit the scaler to the features and transform them
scaled_features = scaler.fit_transform(X)

# Create a new DataFrame with the scaled features
X_scaled = pd.DataFrame(scaled_features, columns=X.columns)
```

-----

### 4\. Train and Evaluate the KNN Model

이제 준비된 데이터를 사용하여 KNN 모델을 훈련시키고 평가합니다.

#### 4.1. Split Data

데이터를 훈련용과 테스트용으로 분리합니다.

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=101)
```

#### 4.2. Train and Predict

먼저 `K=5`로 설정하여 모델을 훈련시키고 예측을 수행합니다.

```python
# Initialize KNN with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Make predictions on the test data
predictions = knn.predict(X_test)
```

#### 4.3. Evaluate the Model

정확도, 혼동 행렬, 분류 리포트를 통해 모델의 성능을 평가합니다.

```python
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")
```

-----

### 5\. Find the Optimal K Value

최적의 `K` 값을 찾기 위해, 여러 `K` 값에 대한 오류율을 계산하고 시각화합니다. 이를 \*\*엘보우 방법(Elbow Method)\*\*이라고 합니다.

```python
error_rate = []

# Loop from k=1 to k=40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# Plot the error rate
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
```

위 그래프에서 오류율이 급격히 감소하다가 완만해지는 "팔꿈치(elbow)" 지점이 최적의 `K` 값 후보가 됩니다. 그래프를 보고 새로운 `K` 값을 선택하여 모델을 다시 훈련시키면 성능을 개선할 수 있습니다. 예를 들어, 위 그래프에서 오류율이 가장 낮은 지점인 `K=15`을 선택해 모델의 정확도를 다시 확인할 수 있습니다.

---

# k= 15로 다시 해보기

### K=15로 모델 재훈련 및 평가

이전과 동일한 훈련 및 테스트 데이터를 사용하여 `n_neighbors`만 15로 변경하여 모델을 다시 훈련하고 평가합니다.

```python
# K=15로 새로운 KNN 모델 초기화
knn_optimal = KNeighborsClassifier(n_neighbors=15)

# 새로운 모델 훈련
knn_optimal.fit(X_train, y_train)

# 새로운 예측 생성
new_predictions = knn_optimal.predict(X_test)

# 새로운 모델 평가
print("--- Evaluation with K=15 ---")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, new_predictions))
print("\nClassification Report:")
print(classification_report(y_test, new_predictions))
print(f"\nNew Accuracy: {accuracy_score(y_test, new_predictions):.4f}")
```

-----

### 새로운 평가 결과

위 코드를 실행하면 K=15일 때의 모델 성능을 확인할 수 있습니다. 최적의 K값을 사용함으로써 이전(`K=5`)보다 정확도가 소폭 상승하고, 전반적으로 안정적인 성능을 보이는 것을 기대할 수 있습니다.

titanic_3에서 결과를 비교해보겠습니다.
