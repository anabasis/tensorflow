# pydot

## [01] pydot, graphviz 설정

- 네트워크의 구조를 시각적으로 표현하고 이미지로 저장 지원

1. pydot 설정

    ```bash
   (base) C:\WINDOWS\system32>activate ai
   (ai) C:\WINDOWS\system32>pip install pydot
    ```

2. Graphviz 설정

   ```bash
   (ai) C:\WINDOWS\system32>conda install python-graphviz
    ```

3. Jupyter notebook 재시작
4. 사용

    ```python
    # plot graph 이미지 생성
    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot
    from keras.utils import plot_model
    plot_model(model, to_file='cnn_ahspe_graph.png')

    SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
    ```

## [02]  수치 예측 모델의 구현(relu, adam, mse 활용), validation_split 적용

- 1차원 선형회귀를 구하는 공식: 분산, 공분산, 평균을 이용하여 산출 가능
   y = wX + b
   f(x) = wX + b
- 기울기 w와 편향 b값을 예측하는 곳이 목표
  최적의 목표값: y = 2X + 0.16

### 1. Script

>> /ws_python/notebook/machine/keras/Basic1.ipynb

```python
# 데이터
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15,16,17,18,19,20])
y_train = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40])

print(x_train)
print(y_train)
```

```python
# 모델 사용
x_use = np.array([51, 52, 53, 54, 55])
y_use = np.array([102, 104, 106, 108, 110])
```

```python
import matplotlib.pyplot as plt
plt.scatter(x_use, y_use, color='g')  # 실제값: 초록색
plt.plot(x_use, y_use, color='g')
plt.scatter(x_use, y_predict, color='r')  # 예측값: 빨간색
plt.plot(x_use, y_predict, color='r')
plt.grid(True)  # 그리드 출력
plt.show()
```

## [03] 2개의 수치로 구성된 입력, 1개의 scala 출력 처리 실습

1. Keras 실습

>> /ws_python/notebook/machine/keras/Basic2.ipynb

```python
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
```

```python
# 데이터, 10행 2열
x_train = []
for i in range(1, 101, 1):  # 1 ~ 100
    x_train.append([i, 10]) # list에 list 추가: 2차원 배열
    # print(i)
x_train = np.array(x_train) # list를 ndarray로 변환
print(x_train[0:5])
print(x_train.shape)
```

```python
y_train = []  # 실제값 저장용 list
for i in range(len(x_train)):
    val = (x_train[i][0] * x_train[i][1])  # 다양한 수식을 적용
    y_train.append([val]) # 각행의 0열과 1열을 곱함

y_train = np.array(y_train)  # list -> ndarray  
print(y_train[0:5])
print(y_train.shape)
```

```python
# 모델 사용
x_use = np.array([[11, 10], [12, 10], [13, 10], [14, 10], [15, 10]])
y_use = np.array([[110], [120], [130], [140], [150]]) # 1 * 10
```

## [03] 3개의 수치로 구성된 입력, 1개의 scala 출력 처리 실습

>> /ws_python/notebook/machine/keras/Basic3.ipynb

```python
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
```

```python
# 데이터, 1 * 5 * 10
x_train = np.array([[1, 5, 10], [2, 5, 10], [3, 5, 10], [4, 5, 10], [5, 5, 10],
                    [6, 5, 10], [7, 5, 10], [8, 5, 10], [9, 5, 10], [10, 5, 10]])
y_train = np.array([[50], [100], [150], [200], [250], [300], [350], [400], [450], [500]])
print(x_train)
print(y_train)
```

```python
# 모델 사용
x_use = np.array([[11, 5, 10], [12, 5, 10], [13, 5, 10], [14, 5, 10], [15, 5, 10]])
y_use = np.array([[550], [600], [650], [700], [750]]) # 1 * 10
```
