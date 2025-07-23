## Seaborn 고급 커스터마이징 자습서 

이 자습서는 Seaborn의 시각적 요소를 완벽하게 제어하기 위한 다양한 함수들을 탐구합니다. 플롯의 전체적인 테마부터 색상 팔레트, 세부적인 축과 범례 조정까지, 발표 수준의 아름다운 그래프를 만들기 위한 기법들을 실습해 보겠습니다.

### 1. 준비 단계

먼저 필요한 라이브러리를 불러오고, 실습에 사용할 데이터셋을 준비합니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 데이터셋 불러오기
tips = sns.load_dataset("tips")
penguins = sns.load_dataset("penguins")

print("Setup complete. Ready to customize seaborn plots.")
```

---

### 2. 테마와 컨텍스트 설정 (Theming)

플롯의 전반적인 스타일과 크기를 제어하는 함수들입니다.

#### 2.1. `set_theme()`: 한번에 테마 설정하기

`set_theme()`은 스타일과 팔레트 등 여러 시각적 요소를 한 번에 설정하는 가장 현대적이고 권장되는 방법입니다. (`set()`은 `set_theme()`의 별칭(alias)입니다.)

```python
# 'darkgrid' 스타일과 'viridis' 팔레트로 테마 설정
sns.set_theme(style="darkgrid", palette="viridis")

# 테마가 적용된 플롯 확인
sns.scatterplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species")
plt.title("Plot with set_theme(style='darkgrid', palette='viridis')")
plt.show()
```

#### 2.2. `set_style()` 과 `axes_style()`: 플롯 배경과 그리드 제어

`set_style()`은 플롯의 배경, 축, 그리드 등 전반적인 스타일을 결정합니다. `axes_style()`은 현재 스타일의 상세 파라미터를 딕셔너리 형태로 보여줍니다.

*   스타일 종류: `darkgrid`, `whitegrid`, `dark`, `white`, `ticks`

```python
# 현재 스타일 파라미터 확인
print("Current style parameters:")
print(sns.axes_style())

# 스타일을 'white'로 변경
sns.set_style("white")

sns.histplot(data=tips, x="total_bill", kde=True)
plt.title("Plot with set_style('white')")
plt.show()
```

#### 2.3. `set_context()` 과 `plotting_context()`: 플롯 요소 크기 조절

`set_context()`는 플롯이 사용될 맥락(예: 논문, 발표)에 맞게 폰트 크기, 선 굵기 등 모든 요소의 크기를 일괄적으로 조절합니다.

*   컨텍스트 종류: `notebook` (기본), `paper`, `talk`, `poster`

```python
# 'poster' 컨텍스트로 변경하여 모든 요소를 크게 만듦
sns.set_context("poster")

sns.stripplot(data=tips, x="day", y="total_bill")
plt.title("Plot with set_context('poster')")
plt.show()

# 다시 기본값인 'notebook'으로 복원
sns.set_context("notebook")
```

#### 2.4. `reset_defaults()`: 모든 설정 초기화

지금까지 변경한 모든 테마, 스타일, 컨텍스트 설정을 Seaborn의 기본값으로 되돌립니다.

```python
# 모든 설정을 Seaborn 기본값으로 초기화
sns.reset_defaults()

sns.boxplot(data=tips, x="day", y="total_bill")
plt.title("Plot after reset_defaults()")
plt.show()
```

---

### 3. 색상 팔레트 다루기 (Color Palettes)

Seaborn의 강점은 정교한 색상 팔레트에 있습니다. 데이터의 특성에 맞는 팔레트를 선택하고 생성하는 방법을 배웁니다.

#### 3.1. `set_palette()` 와 `color_palette()`

`set_palette()`는 모든 플롯에 적용될 기본 색상 순환(color cycle)을 설정합니다. `color_palette()`는 팔레트를 색상 리스트로 반환하며, `sns.palplot()`으로 시각화할 수 있습니다.

```python
# 'pastel' 팔레트를 기본값으로 설정
sns.set_palette("pastel")
sns.stripplot(data=tips, x="day", y="total_bill")
plt.title("Plot with set_palette('pastel')")
plt.show()

# 'deep' 팔레트를 색상 리스트로 가져와 시각화
deep_palette = sns.color_palette("deep")
print("RGB values for 'deep' palette:")
print(deep_palette)
sns.palplot(deep_palette)
plt.title("Visualizing the 'deep' palette")
plt.show()
```

#### 3.2. 다양한 팔레트 생성 함수

Seaborn은 특정 목적에 맞는 팔레트를 생성하는 여러 함수를 제공합니다.

```python
# HUSL 색상 시스템 기반 (균일한 밝기/채도)
sns.palplot(sns.husl_palette(n_colors=8, h=0.5, s=0.8, l=0.6))
plt.title("HUSL Palette")
plt.show()

# Cubehelix 시스템 기반 (순차적 데이터용)
sns.palplot(sns.cubehelix_palette(n_colors=8, start=2, rot=0, dark=0.2, light=0.8))
plt.title("Cubehelix Palette")
plt.show()

# 어두운 색에서 밝은 색으로 (순차적 데이터용)
sns.palplot(sns.dark_palette("purple", n_colors=8))
plt.title("Dark Sequential Palette")
plt.show()

# 두 색 사이의 발산형 팔레트 (중앙점을 기준으로 데이터가 나뉠 때)
sns.palplot(sns.diverging_palette(220, 20, n=7))
plt.title("Diverging Palette")
plt.show()

# xkcd 색상 설문조사 이름으로 팔레트 만들기
sns.palplot(sns.xkcd_palette(["windows blue", "amber", "greyish"]))
plt.title("xkcd Palette")
plt.show()
```

#### 3.3. 팔레트 위젯 (Palette Widgets) - Jupyter 환경에서만 동작

**주의:** 아래 함수들은 Jupyter Notebook 또는 Jupyter Lab과 같은 대화형 환경에서만 동작하는 인터랙티브 위젯을 실행합니다.

```python
# 이 코드는 Jupyter 환경에서 실행해야 위젯이 나타납니다.
# 주석을 해제하고 실행해보세요.

# sns.choose_colorbrewer_palette("sequential")
# sns.choose_cubehelix_palette(as_cmap=False)
# sns.choose_diverging_palette(as_cmap=False)
```
위젯을 실행하면 슬라이더나 색상 선택기를 통해 실시간으로 팔레트를 만들어보고, 결과 팔레트를 복사하여 코드에 붙여넣을 수 있습니다.

---

### 4. 유용한 보조 함수들 (Utility Functions)

플롯을 미세 조정하고 마무리하는 데 유용한 함수들입니다.

#### 4.1. `despine()`: 축 테두리 제거하기

기본적으로 플롯의 위쪽과 오른쪽 축 테두리(spine)를 제거하여 깔끔하게 만듭니다.

```python
# 'ticks' 스타일은 despine과 잘 어울립니다.
sns.set_style("ticks")

fig, ax = plt.subplots()
sns.boxplot(data=tips, x="day", y="total_bill", ax=ax)
plt.title("Before despine()")

# despine() 적용 후
fig, ax = plt.subplots()
sns.boxplot(data=tips, x="day", y="total_bill", ax=ax)
sns.despine() # 위쪽과 오른쪽 테두리 제거
plt.title("After despine()")
plt.show()

# 왼쪽 테두리까지 제거
fig, ax = plt.subplots()
sns.boxplot(data=tips, x="day", y="total_bill", ax=ax)
sns.despine(left=True)
plt.title("After despine(left=True)")
plt.show()

# 모든 설정 초기화
sns.reset_defaults()
```

#### 4.2. `move_legend()`: 범례 위치 옮기기

플롯이 그려진 후 범례의 위치를 조정하거나 제목을 추가할 수 있습니다.

```python
ax = sns.scatterplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species")
plt.title("Original Legend Location")

# move_legend를 사용하여 범례 위치 변경
sns.move_legend(
    ax, "lower right",
    bbox_to_anchor=(.5, .5), # 위치를 미세 조정 (그래프 중앙으로)
    title="Penguin Species" # 범례에 제목 추가
)
plt.title("After move_legend()")
plt.show()
```

#### 4.3. 색상 채도 조절 (`saturate`, `desaturate`)

기존 색상의 채도를 조절하여 새로운 색을 만듭니다.

```python
# 'muted' 팔레트의 첫 번째 색상 가져오기
original_color = sns.color_palette("muted")[0]
saturated_color = sns.saturate(original_color)
desaturated_color = sns.desaturate(original_color, 0.5) # 채도를 50% 낮춤

sns.palplot([original_color, saturated_color, desaturated_color])
plt.xticks([0, 1, 2], ["Original", "Saturated", "Desaturated"])
plt.title("Color Saturation Utilities")
plt.show()
```

#### 4.4. 데이터셋 관련 함수

Seaborn에 내장된 예제 데이터셋을 관리합니다.

```python
# 사용 가능한 모든 데이터셋 이름 출력
print("Available datasets:")
print(sns.get_dataset_names())

# 데이터셋이 캐시되는 로컬 디렉토리 경로 확인
print("\nDataset cache directory:")
print(sns.get_data_home())
```

이 자습서를 통해 Seaborn의 다양한 커스터마이징 함수들을 실습해 보았습니다.
