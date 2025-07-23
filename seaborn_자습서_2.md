## Seaborn 중급자용 자습서: 고급 기법과 커스터마이징

이 자습서는 Seaborn의 기본 기능에 익숙한 사용자를 대상으로, 더 정교하고 통찰력 있는 시각화를 만들기 위한 고급 기법과 다양한 파라미터 활용법을 다룹니다.

### 1. 준비 단계

먼저 필요한 라이브러리를 불러오고, 이 자습서에서 사용할 데이터셋을 준비합니다. `tips`, `penguins`, 그리고 데이터 포인트가 많은 `diamonds` 데이터셋을 활용하겠습니다.

```python
# 라이브러리 불러오기
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Seaborn의 시각적 스타일 설정 (예: 'darkgrid', 'whitegrid', 'ticks')
sns.set_theme(style="ticks", palette="pastel")

# 데이터셋 불러오기
tips = sns.load_dataset("tips")
penguins = sns.load_dataset("penguins")
diamonds = sns.load_dataset("diamonds").sample(n=2000, random_state=42)

print("Libraries and datasets are ready.")
```

### 2. Figure-level vs. Axes-level 함수 이해하기

Seaborn의 함수는 크게 두 가지 수준으로 나뉩니다. 이 차이를 이해하는 것은 중급 사용자로 나아가는 핵심 열쇠입니다.

*   **Axes-level 함수**: `scatterplot()`, `boxplot()`, `histplot()` 등. 이 함수들은 특정 Matplotlib `Axes` 객체 위에 그림을 그립니다. 따라서 하나의 Figure 안에 여러 그래프를 자유롭게 배치하고 조합할 수 있습니다.
*   **Figure-level 함수**: `relplot()`, `catplot()`, `jointplot()`, `lmplot()` 등. 이 함수들은 자체적으로 Figure와 Axes 그리드를 생성합니다. `row`, `col`, `hue` 파라미터를 사용하여 데이터를 하위 그룹으로 나누고 여러 개의 서브플롯을 쉽게 만들 수 있습니다.

### 3. 고급 관계형 및 분포 플롯

#### 3.1. `jointplot`: 두 변수의 관계와 분포를 동시에

`jointplot`은 두 변수 간의 관계(산점도, 육각 빈도 등)와 각 변수의 분포(히스토그램, KDE)를 한 번에 보여줍니다. 데이터가 밀집된 경우(overplotting)에 특히 유용합니다.

```python
# kind='hex'는 점이 겹치는 문제를 해결하기 위해 육각형으로 밀도를 표현합니다.
# diamonds 데이터셋처럼 데이터가 많을 때 유용합니다.
g = sns.jointplot(data=diamonds, x="carat", y="price", kind="hex", color="#4CB391")

# Figure-level 함수는 plt.title() 대신 suptitle을 사용합니다.
g.fig.suptitle("Hexbin Joint Plot of Diamond Price vs. Carat", y=1.02)

plt.show()
```

`kind` 파라미터를 `kde`로 변경하여 등고선 형태로 밀도를 표현할 수도 있습니다.

```python
# kind='kde'는 커널 밀도 추정을 사용하여 2D 밀도와 1D 분포를 보여줍니다.
g = sns.jointplot(
    data=penguins, 
    x="bill_length_mm", 
    y="bill_depth_mm", 
    hue="species", # 종(species) 별로 색상 구분
    kind="kde"
)

g.fig.suptitle("KDE Joint Plot of Penguin Bill Dimensions by Species", y=1.02)
plt.show()
```

#### 3.2. `lmplot`: 조건부 회귀 모델 시각화

`lmplot`은 `regplot`의 Figure-level 버전으로, `hue`, `col`, `row`를 사용하여 여러 하위 그룹에 대한 선형 회귀 모델을 시각화하는 데 매우 강력합니다.

```python
# 'smoker' 여부에 따라 색상(hue)을, 'time'에 따라 열(col)을 분리하여 회귀선을 그립니다.
sns.lmplot(
    data=tips, 
    x="total_bill", y="tip", 
    col="time", hue="smoker",
    height=5, aspect=0.8 # aspect는 가로/세로 비율
)

plt.suptitle("Regression of Tip on Total Bill by Time and Smoker Status", y=1.03)
plt.show()
```

`order` 파라미터를 사용하여 다항 회귀(polynomial regression)를 시각화할 수도 있습니다.

```python
# 2차 다항 회귀 모델을 적용합니다. ci=None은 신뢰구간을 제거합니다.
sns.lmplot(
    data=diamonds, x="carat", y="price", 
    line_kws={'color': 'red'}, # 라인 색상 지정
    scatter_kws={'alpha': 0.3, 's': 15}, # 점의 투명도와 크기 조절
    order=2, # 2차 회귀
    ci=None, # 신뢰구간 제거
    height=6
)

plt.title("2nd Order Polynomial Regression: Price vs. Carat")
plt.show()
```

### 4. 범주형 플롯 조합과 고급 활용

#### 4.1. Box Plot과 Stripplot 중첩하기

Box Plot은 데이터의 요약 통계를 보여주지만, 실제 데이터 포인트의 분포를 가릴 수 있습니다. `stripplot`이나 `swarmplot`을 겹쳐 그리면 요약 정보와 개별 데이터 포인트를 모두 확인할 수 있습니다.

```python
# Axes-level 함수를 사용하기 위해 먼저 Matplotlib의 Figure와 Axes를 생성합니다.
fig, ax = plt.subplots(figsize=(10, 7))

# 1. Boxplot을 먼저 그립니다.
sns.boxplot(data=tips, x="day", y="total_bill", ax=ax, palette="Set2")

# 2. 같은 ax 위에 Stripplot을 겹쳐 그립니다.
sns.stripplot(data=tips, x="day", y="total_bill", ax=ax, color="0.25", alpha=0.6)

# 제목 및 레이블 설정
ax.set_title("Distribution of Total Bill by Day with Individual Points")
ax.set_xlabel("Day of the Week")
ax.set_ylabel("Total Bill ($)")

plt.show()
```

#### 4.2. `catplot`: 범주형 플롯의 통합 인터페이스

`catplot`은 Figure-level 함수로, `kind` 파라미터를 통해 다양한 종류의 범주형 플롯(`bar`, `box`, `violin`, `swarm` 등)을 동일한 인터페이스로 생성할 수 있습니다. `col`과 `row`를 활용한 서브플롯 생성에 탁월합니다.

```python
# kind='bar'로 막대 그래프를 생성합니다.
# 'sex'에 따라 열(col)을 나누고 'smoker'에 따라 색상(hue)을 구분합니다.
sns.catplot(
    data=tips, 
    x="day", y="total_bill", 
    hue="smoker", col="sex",
    kind="bar", # kind를 'box', 'violin', 'strip' 등으로 변경하며 테스트해보세요.
    height=5, aspect=1.1,
    palette="viridis"
)

plt.suptitle("Average Total Bill by Day, Sex, and Smoker Status", y=1.03)
plt.show()
```

### 5. 시각적 스타일링과 세부 조정

#### 5.1. 다양한 컬러 팔레트(Color Palette) 활용

Seaborn은 목적에 맞는 다양한 컬러 팔레트를 제공합니다.

*   **Qualitative (질적)**: 범주 구분을 위한 팔레트 (`pastel`, `deep`, `Set2`)
*   **Sequential (순차적)**: 값의 크기를 나타내는 팔레트 (`viridis`, `rocket`, `mako`)
*   **Diverging (발산형)**: 중앙값을 기준으로 양쪽으로 갈라지는 값을 나타내는 팔레트 (`coolwarm`, `vlag`, `icefire`)

```python
# 상관관계 히트맵에 발산형 팔레트 사용하기
penguins_numeric = penguins.select_dtypes(include=['number'])
corr = penguins_numeric.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr, 
    annot=True,     # 셀 안에 값 표시
    fmt=".2f",      # 소수점 둘째 자리까지
    cmap="vlag",    # 발산형 팔레트 'vlag' 사용
    linewidths=.5   # 셀 사이의 간격
)

plt.title("Correlation Heatmap of Penguin Measurements")
plt.show()
```

#### 5.2. Matplotlib 객체를 이용한 세부 조정

Seaborn은 Matplotlib을 기반으로 하므로, Matplotlib 함수를 함께 사용하여 그래프를 거의 무한하게 커스터마이징할 수 있습니다.

```python
# Axes-level 함수인 regplot을 사용하여 Axes 객체를 직접 다뤄봅니다.
fig, ax = plt.subplots(figsize=(10, 6))

sns.regplot(data=tips, x="total_bill", y="tip", ax=ax, color='b')

# Matplotlib 함수로 세부 조정
ax.set_title("Customized Regression Plot", fontsize=16, fontweight='bold')
ax.set_xlabel("Total Bill ($)", fontsize=12)
ax.set_ylabel("Tip ($)", fontsize=12)

# 수평선 추가 (평균 팁)
mean_tip = tips['tip'].mean()
ax.axhline(mean_tip, ls='--', color='r', label=f'Mean Tip: ${mean_tip:.2f}')

# 그리드 및 범례 추가
ax.grid(True, which='both', linestyle=':', linewidth=0.5)
ax.legend()

plt.show()
```

이 자습서를 통해 Seaborn의 Figure-level과 Axes-level 함수의 차이를 이해하고, `jointplot`, `lmplot`, `catplot`과 같은 강력한 함수들을 다루는 법을 익혔습니다. 또한, 플롯을 중첩하고, 컬러 팔레트를 선택하며, Matplotlib으로 세부적인 요소를 제어하는 방법을 배웠습니다. 
