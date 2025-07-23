## Seaborn 고급 자습서: 한계를 넘어서는 시각화 기법

이 자습서는 Seaborn의 기본 및 중급 기능에 익숙하며, 데이터로부터 더 깊고 복잡한 인사이트를 얻기 위해 라이브러리의 잠재력을 최대한 활용하고자 하는 고급 사용자를 위해 설계되었습니다. 단순한 플롯 생성을 넘어, 데이터 과학 워크플로우에 통합하고, 복잡한 다변수 관계를 탐색하며, 발표 수준의 고도로 맞춤화된 시각화를 만드는 데 초점을 맞춥니다.

### 1. 준비 단계

고급 기능을 위해 Scikit-learn과 같은 다른 라이브러리와의 연계도 시도해 보겠습니다. 필요한 라이브러리를 설치하고 불러옵니다.

```python
# 필요한 라이브러리 설치 (필요시)
# !pip install scikit-learn

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 시각적 테마 설정
sns.set_theme(style="whitegrid", palette="muted")

# 데이터셋 불러오기
iris = sns.load_dataset("iris")
titanic = sns.load_dataset("titanic")
diamonds = sns.load_dataset("diamonds").sample(n=3000, random_state=42) # 샘플링

print("Advanced tutorial setup is complete.")
```

### 2. 비지도 학습 시각화: `clustermap`의 극한 활용

`clustermap`은 단순히 히트맵에 덴드로그램을 추가한 것이 아닙니다. 데이터의 숨겨진 구조를 발견하는 강력한 비지도 학습 시각화 도구입니다.

**기본 사용법을 넘어:** 클러스터링은 변수의 스케일에 민감합니다. 원본 데이터를 그대로 사용하면 스케일이 큰 변수가 클러스터링 결과를 지배하게 됩니다. 따라서 **데이터 정규화**가 필수적입니다.

```python
# Iris 데이터에서 feature만 선택
iris_features = iris.drop("species", axis=1)

# Scikit-learn의 StandardScaler를 사용하여 데이터 정규화
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_features)
iris_scaled_df = pd.DataFrame(iris_scaled, columns=iris_features.columns)

# 실제 종(species) 정보를 색상으로 매핑하여 클러스터링 결과와 비교
species_colors = iris["species"].map({"setosa": "red", "versicolor": "blue", "virginica": "green"})

# 고급 Clustermap 생성
g = sns.clustermap(
    iris_scaled_df,
    method='ward',          # 클러스터링 연결 방식: Ward's method
    metric='euclidean',     # 거리 측정 방식: 유클리드 거리
    cmap='viridis',         # 히트맵 컬러맵
    row_colors=species_colors, # 행에 실제 종(species) 정보 색상으로 추가
    figsize=(10, 10),
    dendrogram_ratio=(.1, .2) # 덴드로그램이 차지하는 비율 조절
)

# 제목 추가 및 레이아웃 조정
g.fig.suptitle('Hierarchical Clustering of Standardized Iris Data', fontsize=16, y=1.02)
g.ax_heatmap.set_xlabel("Features")
g.ax_heatmap.set_ylabel("Samples")

plt.show()
```

**핵심 인사이트:** `row_colors`를 통해 실제 레이블과 `clustermap`이 찾아낸 클러스터가 얼마나 일치하는지 시각적으로 검증할 수 있습니다. 위 예시에서는 클러스터링이 실제 종을 매우 성공적으로 분리해냈음을 알 수 있습니다.

### 3. 대용량 데이터 분포 탐색: `boxenplot`과 `FacetGrid`의 조합

`boxplot`은 유용하지만 대용량 데이터셋의 미묘한 분포 차이를 표현하는 데 한계가 있습니다. `boxenplot`(Letter-value plot)은 더 많은 분위수를 표시하여 데이터의 꼬리 부분까지 상세하게 보여줍니다.

**극한 활용법:** `boxenplot`은 Axes-level 함수입니다. 이를 Figure-level 객체인 `FacetGrid`와 수동으로 결합하여, `catplot`으로는 만들기 어려운 복잡하고 맞춤화된 다중 플롯을 만들 수 있습니다.

```python
# FacetGrid 객체 생성: 'sex'와 'pclass'에 따라 그리드 분할
g = sns.FacetGrid(
    titanic, 
    row="sex", col="pclass", 
    height=4, aspect=1.2, 
    margin_titles=True # 축 제목을 그리드 바깥쪽으로 이동
)

# 각 그리드(ax)에 boxenplot을 매핑
# palette와 k_depth 파라미터로 시각적 스타일 제어
g.map_dataframe(
    sns.boxenplot, 
    x="age", 
    palette="flare", 
    k_depth="proportion" # 분위수 깊이를 데이터 비율에 따라 조절
)

# 전체 제목 및 레이블 설정
g.fig.suptitle("Detailed Age Distribution by Sex and Passenger Class", y=1.03)
g.set_axis_labels("Age", "Distribution")
g.set_titles(col_template="Class: {col_name}", row_template="Sex: {row_name}")
g.tight_layout(w_pad=1) # 서브플롯 간 간격 조절

plt.show()
```

### 4. 다차원 관계 탐색의 끝판왕: `relplot`의 모든 기능 활용하기

`relplot`은 `scatterplot`과 `lineplot`의 Figure-level 인터페이스로, 최대 4개의 변수(`x`, `y`, `hue`, `size`, `style`)를 동시에 시각화하며 `row`와 `col`로 패싯팅(faceting)할 수 있습니다.

**극한 활용법:** 모든 시각적 변수(semantic variables)를 활용하고, 각 서브플롯에 동적 주석(annotation)을 추가해 봅시다.

```python
# `relplot`으로 다차원 관계 시각화
g = sns.relplot(
    data=diamonds,
    x="carat", y="price",
    hue="clarity",          # 색상으로 명료도 표현
    size="depth",           # 점 크기로 깊이 표현
    col="cut",              # 열(column)을 컷 등급으로 분할
    col_wrap=3,             # 3개의 열마다 줄 바꿈
    kind="scatter",
    palette="YlOrBr",
    sizes=(10, 200),        # 점 크기 범위
    alpha=0.6,
    height=4
)

# 각 서브플롯에 상관계수(correlation) 주석을 추가하는 함수
def annotate_corr(x, y, **kwargs):
    # g.map은 데이터 시리즈를 직접 전달하므로, 추가적인 변환이 필요 없습니다.
    ax = plt.gca()
    corr = x.corr(y)
    ax.text(.1, .9, f'Corr: {corr:.2f}', transform=ax.transAxes,
            ha='left', va='top', fontsize=10, color='black',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.7))

# map_dataframe 대신 map을 사용하여 각 서브플롯에 annotate_corr 함수 적용
# g.map은 'carat'과 'price' 열의 데이터를 뽑아서 annotate_corr의 x, y 인자로 전달합니다.
g.map(annotate_corr, "carat", "price")

# 제목 및 레이아웃 정리
g.fig.suptitle("Multi-dimensional Analysis of Diamond Properties", y=1.03)
g.set_axis_labels("Carat", "Price ($)")
g.tight_layout()

plt.show()
```

### 5. 이변수 분포의 완전한 해부: 수동으로 만드는 `histplot`과 주변부 플롯

`jointplot`은 편리하지만 커스터마이징에 한계가 있습니다. `matplotlib.pyplot.subplots`와 `histplot`을 이용해 완전히 통제 가능한 이변수 분포 플롯을 직접 만들 수 있습니다.

**극한 활용법:** 중앙의 2D 히스토그램과 양쪽의 1D 히스토그램(주변부 분포)을 수동으로 배치하고, 각 플롯의 스타일을 개별적으로 제어합니다.

```python
# Matplotlib을 사용하여 3개의 서브플롯 그리드 생성
fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)

ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

# 중앙의 2D 히트맵 플롯
sns.histplot(data=diamonds, x="carat", y="price", ax=ax,
             bins=(30, 40), cmap="magma", cbar=True, cbar_kws=dict(label="Count"))

# 상단의 X축 주변부 플롯
sns.histplot(data=diamonds, x="carat", ax=ax_histx, color="gray", element="step")
ax_histx.tick_params(axis="x", labelbottom=False) # x축 레이블 숨기기
ax_histx.set_ylabel("Count")

# 우측의 Y축 주변부 플롯
sns.histplot(data=diamonds, y="price", ax=ax_histy, color="gray", element="step")
ax_histy.tick_params(axis="y", labelleft=False) # y축 레이블 숨기기
ax_histy.set_xlabel("Count")

# 전체 제목
fig.suptitle("Bivariate and Marginal Distributions of Diamond Carat and Price", fontsize=16)

plt.show()
```

### 6. 궁극의 커스터마이징: `PairGrid`

`pairplot`은 데이터 전체를 빠르게 훑어보는 데 유용하지만, 모든 플롯이 동일한 종류로 그려집니다. `PairGrid`를 사용하면 그리드의 각 부분(대각, 상단, 하단)에 서로 다른 종류의 플롯을 매핑할 수 있습니다.

```python
# PairGrid 객체 생성
g = sns.PairGrid(iris, hue="species", palette="husl", diag_sharey=False)

# 대각선(Diagonal)에는 KDE 플롯 매핑
g.map_diag(sns.kdeplot, fill=True, alpha=.6)

# 상단(Upper triangle)에는 회귀선이 없는 산점도 매핑
g.map_upper(sns.scatterplot, s=30, alpha=.7)

# 하단(Lower triangle)에는 2D KDE 플롯 매핑
g.map_lower(sns.kdeplot, cmap="Blues_d")

# 범례 추가 및 제목 설정
g.add_legend()
g.fig.suptitle("Customized PairGrid of Iris Dataset", y=1.02)

plt.show()
```

