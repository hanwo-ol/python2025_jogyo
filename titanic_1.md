### 예측 모델링의 첫걸음: 분류와 회귀의 차이점 이해하기

본격적인 실습에 앞서, 사용자가 요청한 '회귀'와 이 문제의 실제 해결 방법인 '분류'의 차이점을 명확히 짚고 넘어가겠습니다.

  * **회귀 (Regression)**: 연속적인 숫자 값을 예측하는 데 사용됩니다. 예를 들어, 주택 가격, 주가, 또는 사람의 키를 예측하는 경우가 여기에 해당합니다.
  * **분류 (Classification)**: 데이터 포인트를 미리 정해진 두 개 이상의 카테고리(범주) 중 하나로 분류하는 데 사용됩니다. 타이타닉 데이터셋의 '생존' 또는 '사망' 여부 예측, 이메일의 '스팸' 또는 '정상' 분류 등이 대표적인 예시입니다.

따라서 타이타닉 생존자를 예측하는 것은 **분류 문제**이며, 우리는 이 문제에 적합한 로지스틱 회귀 모델을 사용하게 됩니다. 로지스틱 회귀는 이름에 '회귀'가 포함되어 있지만, 실제로는 특정 클래스에 속할 확률을 예측하여 분류를 수행하는 알고리즘입니다.

-----

### 1단계: 개발 환경 설정 및 데이터 불러오기

먼저, 데이터 분석과 모델링에 필요한 라이브러리를 가져옵니다. `pandas`는 데이터 조작, `seaborn`과 `matplotlib`은 시각화, `scikit-learn`은 기계 학습 모델을 위해 사용됩니다.

```python
# 데이터 분석 및 시각화를 위한 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 모델링을 위한 scikit-learn 라이브러리 임포트
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 스타일 설정
%matplotlib inline
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
```

이제 Seaborn 라이브러리에 내장된 타이타닉 데이터셋을 불러와 `titanic`이라는 `DataFrame` 객체에 저장합니다.

```python
# Seaborn에서 타이타닉 데이터셋 불러오기
titanic = sns.load_dataset('titanic') # 여기서 오래 걸린다면 Kernel -> Restart Kernel을 한 후 처음부터 다시 실행해보세요.

# 데이터의 첫 5행 확인
titanic.head()
```

-----

### 2단계: 탐색적 데이터 분석 (EDA)

모델을 만들기 전에 데이터를 깊이 이해하는 과정인 탐색적 데이터 분석(EDA)을 수행합니다. 데이터의 구조를 파악하고, 변수 간의 관계를 시각화하여 인사이트를 얻는 것이 목표입니다.

#### 데이터의 기본 정보 확인

먼저 `info()`와 `describe()` 메소드를 사용하여 데이터셋의 전반적인 정보를 확인합니다.

```python
# 데이터셋의 기본 정보 출력
titanic.info()

# 수치형 데이터의 통계 요약 정보 출력
titanic.describe()
```

`info()`를 통해 'age', 'deck', 'embarked', 'embark\_town' 열에 결측치가 있는 것을 확인할 수 있습니다. `describe()`를 통해서는 수치형 데이터의 분포를 파악할 수 있습니다.

#### 시각화를 통한 데이터 탐색

다양한 플롯을 그려보며 데이터에 숨겨진 패턴을 찾아봅니다.

  * **생존 여부에 따른 승객 수**: `countplot`을 사용하여 생존자와 사망자의 수를 시각적으로 비교합니다.

    ```python
    sns.countplot(x='survived', data=titanic, hue='survived', palette='RdBu_r', legend=False)
    plt.title('survivor vs the dead')
    plt.show()
    ```

  * **성별에 따른 생존율**: 성별이 생존에 어떤 영향을 미쳤는지 확인합니다.

    ```python
    sns.countplot(x='survived', hue='sex', data=titanic, palette='viridis')
    plt.title('Survival by gender')
    plt.show()
    ```

  * **객실 등급에 따른 생존율**: Pclass(객실 등급)가 생존율과 어떤 관련이 있는지 살펴봅니다.

    ```python
    sns.countplot(x='survived', hue='pclass', data=titanic, palette='rainbow')
    plt.title('Survival based on room grade')
    plt.show()
    ```

  * **나이 분포**: `histplot`을 사용하여 승객들의 나이 분포를 확인합니다.

    ```python
    sns.histplot(titanic['age'].dropna(), kde=True, bins=30, color='darkred')
    plt.title('Age Distribution')
    plt.show()
    ```

EDA를 통해 여성과 높은 등급의 객실 승객이 더 높은 생존율을 보였다는 점 등 유의미한 패턴을 발견할 수 있습니다.

-----

### 3단계: 데이터 전처리 및 피처 엔지니어링

기계 학습 모델은 숫자 데이터를 입력으로 받기 때문에, 결측치를 처리하고 범주형 데이터를 수치형으로 변환하는 전처리 과정이 필수적입니다.

#### 결측치 처리

  * **Age (나이)**: 나이의 결측치는 전체 데이터의 평균 나이로 채워 넣겠습니다. 더 정교한 방법으로는 객실 등급별 평균 나이를 계산하여 채울 수도 있습니다.

    ```python
    titanic['age'] = titanic['age'].fillna(titanic['age'].mean())

    ```

  * **Deck (객실)**: 'deck' 열은 결측치가 너무 많아 예측에 사용하기 어렵다고 판단하여 제거합니다.

    ```python
    titanic.drop('deck', axis=1, inplace=True)
    ```

  * **Embarked & Embark Town (탑승 항구)**: 이 두 열은 소수의 결측치를 가지며, 최빈값(가장 자주 나타나는 값)으로 채웁니다.

    ```python
    titanic.dropna(inplace=True)
    ```

#### 범주형 데이터 변환

'sex', 'embarked'와 같은 문자열로 된 범주형 변수는 모델이 이해할 수 있도록 수치형으로 변환해야 합니다. `pandas`의 `get_dummies` 함수를 사용하면 이 과정을 쉽게 처리할 수 있습니다. 이를 \*\*원-핫 인코딩(One-Hot Encoding)\*\*이라고 합니다.

```python
# 범주형 변수들을 원-핫 인코딩으로 변환
sex = pd.get_dummies(titanic['sex'], drop_first=True, dtype=int)
embark = pd.get_dummies(titanic['embarked'], drop_first=True, dtype=int)

# 기존의 범주형 열과 불필요한 열 제거
titanic.drop(['sex', 'embarked', 'who', 'adult_male', 'embark_town', 'alive', 'class'], axis=1, inplace=True)

# 변환된 수치형 데이터프레임과 기존 데이터프레임 병합
titanic = pd.concat([titanic, sex, embark], axis=1)

titanic.head()
```

`drop_first=True` 옵션은 다중공선성 문제를 방지하기 위해 각 범주형 변수의 첫 번째 카테고리를 제거하는 역할을 합니다.

-----

### 4단계: 모델 훈련 및 평가

이제 전처리를 마친 데이터를 사용하여 로지스틱 회귀 모델을 훈련시키고 성능을 평가할 차례입니다.

#### 훈련 데이터와 테스트 데이터 분리

데이터셋을 모델 훈련에 사용할 \*\*훈련 데이터(Training Data)\*\*와 모델 성능 평가에 사용할 \*\*테스트 데이터(Test Data)\*\*로 나눕니다. 일반적으로 7:3 또는 8:2 비율을 사용합니다.

```python
# 독립 변수(X)와 종속 변수(y) 설정
X = titanic.drop('survived', axis=1)
y = titanic['survived']

# 훈련 데이터와 테스트 데이터 분리 (70% 훈련, 30% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

#### 로지스틱 회귀 모델 훈련

`scikit-learn`의 `LogisticRegression` 모델을 생성하고 훈련 데이터를 사용하여 학습시킵니다.

```python
# 로지스틱 회귀 모델 객체 생성
logmodel = LogisticRegression(solver='liblinear')

# 모델 훈련
logmodel.fit(X_train, y_train)
```

#### 모델 예측 및 평가

훈련된 모델을 사용하여 테스트 데이터의 생존 여부를 예측하고, 실제 값과 비교하여 모델의 성능을 평가합니다.

```python
# 테스트 데이터로 예측 수행
predictions = logmodel.predict(X_test)

# 정확도(Accuracy) 평가
accuracy = accuracy_score(y_test, predictions)
print(f'모델 정확도: {accuracy:.4f}\n')

# 혼동 행렬(Confusion Matrix) 출력
print('혼동 행렬:')
print(confusion_matrix(y_test, predictions))
print('\n')

# 분류 리포트(Classification Report) 출력
print('분류 리포트:')
print(classification_report(y_test, predictions))
```

**평가 지표 해석:**

  * **정확도(Accuracy)**: 전체 예측 중 올바르게 예측한 비율입니다. 이 모델은 약 80%의 정확도를 보입니다.
  * **혼동 행렬(Confusion Matrix)**: 모델의 예측이 얼마나 헷갈리는지를 보여주는 행렬입니다. True Positive, False Positive, True Negative, False Negative의 수를 통해 모델의 성능을 구체적으로 파악할 수 있습니다.
  * **분류 리포트(Classification Report)**: 정밀도(Precision), 재현율(Recall), F1-점수(F1-score) 등 각 클래스별 성능 지표를 제공하여 모델을 다각도로 평가할 수 있게 해줍니다.

**여러분과 제 결과는 차이가 있을 수 있습니다. 아래는 조교의 결과입니다.**
``` python

모델 정확도: 0.8202

혼동 행렬:
[[150  13]
 [ 35  69]]


분류 리포트:
              precision    recall  f1-score   support

           0       0.81      0.92      0.86       163
           1       0.84      0.66      0.74       104

    accuracy                           0.82       267
   macro avg       0.83      0.79      0.80       267
weighted avg       0.82      0.82      0.82       267

```


### 모델 성능 요약

* **전체 정확도 (Accuracy): 82.02%**
    * 모델이 전체 267명의 테스트 데이터 중 약 82%의 사람들에 대해 생존 또는 사망 여부를 올바르게 예측했습니다.

---

### 혼동 행렬 (Confusion Matrix) 분석

혼동 행렬은 모델이 어떤 예측을 맞고 틀렸는지 구체적으로 보여주는 표입니다. (0: 사망, 1: 생존)

|               | **예측: 사망(0)** | **예측: 생존(1)** |
| :------------ | :------------------ | :------------------ |
| **실제: 사망(0)** | **150** (TN)        | **13** (FP)         |
| **실제: 생존(1)** | **35** (FN)         | **69** (TP)         |

* **True Negative (TN): 150**
    * 실제 **사망**한 사람을 **사망**했다고 올바르게 예측한 경우입니다.
* **False Positive (FP): 13**
    * 실제 **사망**한 사람을 **생존**했다고 잘못 예측한 경우입니다 (오탐).
* **False Negative (FN): 35**
    * 실제 **생존**한 사람을 **사망**했다고 잘못 예측한 경우입니다 (미탐).
* **True Positive (TP): 69**
    * 실제 **생존**한 사람을 **생존**했다고 올바르게 예측한 경우입니다.

---

### 분류 리포트 (Classification Report) 심층 분석

* **정밀도 (Precision)**: 모델이 "생존했다"고 예측한 것 중, 실제 생존한 사람의 비율입니다.
    * 사망자(0) 예측의 정밀도: **81%** (사망이라고 예측한 185명 중 150명이 실제 사망)
    * 생존자(1) 예측의 정밀도: **84%** (생존이라고 예측한 82명 중 69명이 실제 생존)

* **재현율 (Recall)**: 실제 생존한 사람들 중에서 모델이 "생존했다"고 맞춘 비율입니다. **이 모델의 가장 중요한 특징을 보여줍니다.**
    * 사망자(0)에 대한 재현율: **92%** (실제 사망자 163명 중 150명을 찾아냄)
    * 생존자(1)에 대한 재현율: **66%** (실제 생존자 104명 중 69명만 찾아냄)

* **F1-Score**: 정밀도와 재현율의 조화 평균으로, 클래스별 모델 성능의 균형을 나타냅니다.

---

### 결론 및 해석

* **강점**: 이 모델은 **사망자를 찾아내는 데 매우 뛰어납니다 (재현율 92%)**. 즉, 사망한 사람을 놓칠 확률이 낮습니다.
* **약점**: 반면, **실제 생존자를 놓치는 경우가 많습니다 (재현율 66%)**. 실제 생존자 104명 중 35명을 사망했다고 잘못 예측했습니다.
* **종합 평가**: 모델은 전반적으로 준수한 성능(정확도 82%)을 보이지만, **'생존'보다는 '사망' 예측에 더 치우쳐진(biased) 모델**이라고 할 수 있습니다.

-----

