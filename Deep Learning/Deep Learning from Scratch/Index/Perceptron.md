# 퍼셉트론

퍼셉트론은 1957년에 *프랑크 로젠블라트*가 고안한 알고리즘이다.

퍼셉트론을 통해 신경망 알고리즘이 탄생했기 때문에 퍼셉트론 구조를 배우는 것은 신경망과 딥러닝의 기초가 된다.

## 1. 퍼셉트론

```markdown
다수의 신호를 입력으로 받아 하나의 신호 출력

1. 입력 신호가 뉴련에 보내질 때는 가중치가 곱해지고 현향이 더해짐
2. 신호의 총합이 임계값을 넘어가면 뉴련 활성화
> weight(가중치) 크기 = 해당 신호의 중요도
> bias(편향)의 크기 = 뉴런의 활성화 난이도
```

y = 0(*b + w<sub>1</sub>w<sub>1</sub>*+ ... + *w<sub>n</sub>w<sub>n</sub>* <= 0) or 1(*b + w<sub>1</sub>w<sub>1</sub>*+ ... + *w<sub>n</sub>w<sub>n</sub>* > 1)

## 2. 단순한 논리회로

### 2.1. AND 게이트

```python
import numpy as np

def AND(x1, x2):
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.7
  tmp = np.sum(w*x) + b
  
  if tmp <= 0:
    return 0
  elif tmp > 1:
    return 1
```

### 2.2. NAND 게이트

```python
import numpy as np

def AND(x1, x2):
  x = np.array([x1, x2])
  w = np.array([-0.5, -0.5])
  b = -0.7
  tmp = np.sum(w*x) + b
  
  if tmp <= 0:
    return 0
  elif tmp > 1:
    return 1
```

### 2.3. OR 게이트

```python
import numpy as np

def AND(x1, x2):
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.2
  tmp = np.sum(w*x) + b
  
  if tmp <= 0:
    return 0
  elif tmp > 1:
    return 1
```

## 3. 퍼셉트론의 한계

퍼셉트론은 직선 하나로 영역을 나눈다. 그렇기 때문에 직선 하나로 영역을 구분할 수 없는 문제는 퍼셉트론으로 풀 수 없다.

## 4. 다층 퍼셉트론

```markdown
퍼셉트론의 층을 쌓아 만든 알고리즘
```

### 5.1. 기존 게이트 조합

### 5.2. XOR 게이트 구현

XOR의 경우 AND, NAND, OR처럼 직선 하나로 구분할 수 없다. 따라서 다음과 같이 AND, NAND, OR을 활용한 다층 퍼셉트론으로 구현할 수 있다.
```python
def XOR(x1, x2):
  s1 = NAND(x1, x2) // 1층
  s2 = OR(x1, x2)   // 2층
  y = AND(s1, s2)
  return y
```

XOR은 다층 퍼셉트론 중 2층 퍼셉트론에 해당한다. 
