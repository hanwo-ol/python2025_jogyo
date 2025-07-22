다음은 `matplotlib.pyplot` 모듈의 `plot()` 함수에 대해 모든 주요 옵션을 실습해볼 수 있도록 구성한 **자습서**입니다!! 

이 자습서는 각 옵션의 의미와 함께 실습 코드를 제공하여 직접 실행하고 시각적 변화를 확인할 수 있게 작성해봤어요.

---

# Matplotlib `plot()` 함수 자습서

## 1. 기본 사용법

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

plt.plot(x, y)
plt.title("기본 선 그래프")
plt.xlabel("x 값")
plt.ylabel("y 값")
plt.grid(True)
plt.show()
```

---

## 2. 주요 옵션 실습

### 2.1 `color` / `c` - 선 색상

```python
plt.plot(x, y, color='green')  # 또는 c='green'
plt.title("선 색상 설정")
plt.show()
```

* 예시: `'blue'`, `'red'`, `'#00FF00'`, `'0.5'` (회색조)

---

### 2.2 `linestyle` / `ls` - 선 스타일

```python
plt.plot(x, y, linestyle='--')  # 또는 ls='--'
plt.title("점선 스타일")
plt.show()
```

* 주요 스타일:

  * `'-'`: 실선 (기본)
  * `'--'`: 점선
  * `'-.'`: 점선+실선 혼합
  * `':'`: 점만

---

### 2.3 `linewidth` / `lw` - 선 두께

```python
plt.plot(x, y, linewidth=3)  # 또는 lw=3
plt.title("선 두께 조정")
plt.show()
```

---

### 2.4 `marker` - 데이터 포인트 표시

```python
plt.plot(x, y, marker='o')
plt.title("원형 마커 표시")
plt.show()
```

* 주요 마커:

  * `'o'`: 원
  * `'s'`: 정사각형
  * `'^'`: 위쪽 삼각형
  * `'x'`, `'+'`, `'D'` 등

---

### 2.5 `markersize` / `ms` - 마커 크기

```python
plt.plot(x, y, marker='o', markersize=10)
plt.title("마커 크기 조정")
plt.show()
```

---

### 2.6 `markerfacecolor` / `mfc`, `markeredgecolor` / `mec`, `markeredgewidth` / `mew`

```python
plt.plot(x, y, marker='o', mfc='white', mec='red', mew=2)
plt.title("마커 스타일 조정")
plt.show()
```

---

### 2.7 `label` - 범례 이름 설정

```python
plt.plot(x, y, label='제곱')
plt.legend()
plt.title("범례 표시")
plt.show()
```

---

### 2.8 `alpha` - 투명도

```python
plt.plot(x, y, color='purple', alpha=0.3)
plt.title("투명도 설정")
plt.show()
```

---

### 2.9 `zorder` - 앞뒤 순서 지정

```python
plt.plot(x, y, zorder=2)
plt.plot(x, [10]*5, color='gray', linewidth=10, zorder=1)
plt.title("zorder로 순서 조정")
plt.show()
```

---

### 2.10 `drawstyle` - 선 그리는 방식

```python
plt.plot(x, y, drawstyle='steps-post')
plt.title("계단형 그래프 (steps-post)")
plt.show()
```

* `'default'`, `'steps'`, `'steps-pre'`, `'steps-mid'`, `'steps-post'`

---

## 3. 여러 옵션을 조합한 예시

```python
plt.plot(x, y,
         color='orange',
         linestyle='-.',
         linewidth=2,
         marker='D',
         markersize=8,
         markerfacecolor='blue',
         markeredgecolor='black',
         alpha=0.8,
         label='복합 스타일')
plt.legend()
plt.title("복합 스타일 적용 예시")
plt.grid(True)
plt.show()
```

---

## 4. 복수 선 그리기 실습

```python
y2 = [25, 16, 9, 4, 1]
plt.plot(x, y, label='y = x^2', color='blue')
plt.plot(x, y2, label='y = 반전', color='red', linestyle='--')
plt.legend()
plt.title("두 개의 선 그리기")
plt.show()
```

---
