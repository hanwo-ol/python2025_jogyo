## Seaborn을 활용한 데이터 시각화 자습서

이 자습서에서는 Python의 데이터 시각화 라이브러리인 Seaborn의 다양한 기능을 내장 데이터셋을 사용하여 배워보겠습니다. 
* Seaborn은 Matplotlib을 기반으로 더 아름답고 통계적으로 의미 있는 그래프를 쉽게 만들 수 있게 해줍니다.


### 1. 준비 단계

먼저 필요한 라이브러리를 설치하고 불러옵니다. Seaborn을 설치하면 Matplotlib과 Pandas 등 의존성 있는 라이브러리도 함께 설치됩니다.

```python
# 라이브러리 설치 (아직 설치하지 않은 경우)
# !pip install seaborn

# 라이브러리 불러오기
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

print("Libraries imported successfully!")
```

### 2. 내장 데이터셋 불러오기

Seaborn은 학습용으로 여러 데이터셋을 내장하고 있습니다. `load_dataset()` 함수를 사용하여 쉽게 불러올 수 있습니다. 이 자습서에서는 `tips`, `titanic`, `penguins` 데이터셋을 주로 사용하겠습니다.

```python
# tips 데이터셋 불러오기 (레스토랑 팁 정보)
tips = sns.load_dataset("tips")

# titanic 데이터셋 불러오기 (타이타닉호 승객 정보)
titanic = sns.load_dataset("titanic")

# penguins 데이터셋 불러오기 (펭귄 품종 정보)
penguins = sns.load_dataset("penguins")

# 데이터셋의 첫 5행 확인
print("--- Tips Dataset ---")
print(tips.head())
print("\n--- Titanic Dataset ---")
print(titanic.head())
print("\n--- Penguins Dataset ---")
print(penguins.head())
```

### 3. 관계형 플롯 (Relational Plots)

두 변수 간의 관계를 시각화하는 데 사용됩니다.

#### 3.1. Scatter Plot (산점도)

두 연속형 변수 간의 관계를 점으로 나타냅니다.

```python
plt.figure(figsize=(8, 6)) # 그림 크기 조절
sns.scatterplot(data=tips, x="total_bill", y="tip")

# 제목 및 레이블 추가
plt.title("Relation between Total Bill and Tip")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip ($)")

plt.show()
```

`hue` 파라미터를 사용하면 특정 카테고리에 따라 점의 색상을 다르게 지정할 수 있습니다.

```python
plt.figure(figsize=(8, 6))
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time")

# 제목 및 레이블 추가
plt.title("Relation between Total Bill and Tip by Meal Time")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip ($)")

plt.show()
```

#### 3.2. Line Plot (선 그래프)

x축 변수에 따른 y축 변수의 변화 추이를 선으로 나타냅니다. 주로 시계열 데이터에 사용됩니다.

```python
#flights 데이터셋은 시계열 데이터로 Line Plot에 적합합니다.
flights = sns.load_dataset("flights")
flights_wide = flights.pivot("month", "year", "passengers") # 데이터를 피벗 테이블 형태로 변환

plt.figure(figsize=(10, 6))
sns.lineplot(data=flights_wide)

# 제목 및 레이블 추가
plt.title("Monthly Airline Passengers Over Years")
plt.xlabel("Month")
plt.ylabel("Number of Passengers")

plt.show()
```

### 4. 범주형 플롯 (Categorical Plots)

하나 이상의 범주형 변수와 연속형 변수 간의 관계를 시각화합니다.

#### 4.1. Bar Plot (막대 그래프)

범주에 따른 연속형 변수의 평균(기본값)과 신뢰구간을 막대로 나타냅니다.

```python
plt.figure(figsize=(8, 6))
sns.barplot(data=titanic, x="class", y="fare", hue="sex")

# 제목 및 레이블 추가
plt.title("Average Fare by Passenger Class and Sex")
plt.xlabel("Passenger Class")
plt.ylabel("Average Fare")

plt.show()
```

#### 4.2. Count Plot (개수 플롯)

범주별 데이터의 개수를 막대로 나타냅니다. `barplot`과 달리 y축 변수를 지정할 필요가 없습니다.

```python
plt.figure(figsize=(8, 6))
sns.countplot(data=titanic, x="class", hue="alive")

# 제목 및 레이블 추가
plt.title("Passenger Count by Class and Survival Status")
plt.xlabel("Passenger Class")
plt.ylabel("Count")

plt.show()
```

#### 4.3. Box Plot (상자 그림)

데이터의 분포를 사분위수를 이용하여 시각화합니다. 중앙값, 이상치 등을 파악하기 좋습니다.

```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=tips, x="day", y="total_bill", hue="smoker")

# 제목 및 레이블 추가
plt.title("Distribution of Total Bill by Day and Smoker Status")
plt.xlabel("Day of the Week")
plt.ylabel("Total Bill ($)")

plt.show()
```

#### 4.4. Violin Plot (바이올린 플롯)

Box Plot과 데이터의 분포를 나타내는 커널 밀도 추정(Kernel Density Estimation)을 합친 형태입니다.

```python
plt.figure(figsize=(10, 6))
sns.violinplot(data=penguins, x="species", y="body_mass_g", hue="sex")

# 제목 및 레이블 추가
plt.title("Body Mass Distribution by Penguin Species and Sex")
plt.xlabel("Species")
plt.ylabel("Body Mass (g)")

plt.show()
```

### 5. 분포 플롯 (Distribution Plots)

단일 변수의 분포를 시각화하는 데 사용됩니다.

#### 5.1. Histogram (히스토그램)

데이터를 특정 구간(bin)으로 나누어 각 구간에 속하는 데이터의 빈도를 막대로 나타냅니다.

```python
plt.figure(figsize=(8, 6))
sns.histplot(data=penguins, x="flipper_length_mm", bins=20, kde=True) # kde=True로 밀도 곡선 추가

# 제목 및 레이블 추가
plt.title("Distribution of Penguin Flipper Lengths")
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Frequency")

plt.show()
```

#### 5.2. KDE Plot (커널 밀도 추정 플롯)

히스토그램을 부드러운 곡선으로 나타낸 것으로, 데이터의 연속적인 분포를 보여줍니다.

```python
plt.figure(figsize=(8, 6))
sns.kdeplot(data=penguins, x="bill_length_mm", hue="species", fill=True)

# 제목 및 레이블 추가
plt.title("Density of Bill Length by Penguin Species")
plt.xlabel("Bill Length (mm)")
plt.ylabel("Density")

plt.show()
```

### 6. 매트릭스 플롯 (Matrix Plots)

데이터 행렬을 색상으로 인코딩하여 시각화합니다.

#### 6.1. Heatmap (히트맵)

행렬 형태의 데이터 값을 색상으로 표현합니다. 주로 변수 간의 상관관계를 나타낼 때 사용됩니다.

```python
# 숫자형 데이터만 선택하여 상관관계 행렬 계산
numeric_tips = tips.select_dtypes(include=['number'])
correlation_matrix = numeric_tips.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f") # annot=True로 값 표시

# 제목 추가
plt.title("Correlation Matrix of Tips Dataset")

plt.show()
```

### 7. 다중 플롯 그리드 (Multi-plot Grids)

여러 개의 하위 플롯(subplot)을 생성하여 데이터를 다양한 측면에서 비교 분석할 수 있습니다.

#### 7.1. Pair Plot (페어 플롯)

데이터셋의 모든 숫자형 변수 쌍에 대해 산점도를, 그리고 각 변수 자체에 대해서는 히스토그램(또는 KDE)을 그립니다. 데이터 전체를 조망하기에 매우 유용합니다.

```python
# hue를 사용하여 종(species)에 따라 색상을 구분
sns.pairplot(penguins, hue="species")

# Pair Plot은 자체적으로 제목을 추가하는 기능이 제한적이므로, plt.suptitle을 사용합니다.
plt.suptitle("Pairwise Relationships in the Penguins Dataset", y=1.02) # y=1.02는 제목 위치 조정

plt.show()
```

#### 7.2. FacetGrid

사용자가 지정한 조건에 따라 여러 개의 하위 플롯 그리드를 만들고, 각 그리드에 원하는 종류의 그래프를 매핑할 수 있습니다.

```python
# 'time'과 'sex' 카테고리에 따라 그리드를 생성
g = sns.FacetGrid(tips, col="time", row="sex")

# 각 그리드에 산점도를 매핑
g.map(sns.scatterplot, "total_bill", "tip")

# 각 하위 플롯에 제목 자동 생성됨

plt.show()
```
