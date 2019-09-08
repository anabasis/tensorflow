# 이항분포2

## [01] 과적합 피하기, 초음파 광물 예측하기

- 1988년 존스홉킨스대학교의 세즈노프스키(Sejnowski) 교수는 2년 전 힌튼 교수가 발표한
  역전파 알고리즘에 관심을 가지고 있었음
  그는 역전파가 얼마나 큰 효과가 있는지를 직접 실험해 보기 위해 광석과 일반 돌을 갖다 놓고
  음파 탐지기를 쏜 후 그 결과를 데이터로 정리
![광물](./images/01_2.jpg)

    (1) LabelEncoder()
      - 문자열을 숫자로 그룹화해서 변경, 문자 코드순서 적용
      예)

      ```python
      from sklearn.preprocessing import LabelEncoder

      e = LabelEncoder() # 문자열을 숫자로 그룹화해서 변경, 문자 코드순서 적용
      e.fit(Y_obj)  # 
      Y = e.transform(Y_obj)  # 0, 1, 2 정수로 변환
      print(Y)
      ```

    (2) 난수 기반 훈련/검증 데이터 추출

        ```python
        from sklearn.model_selection import train_test_split # 학습셋과 테스트셋의 분리 지원

        seed = 0
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=seed)
        print(y_val)
        print(y_val.shape)
        ```

1. 데이터

   1) 구조
   - Index가 208개이므로 총 샘플의 수는 208개이고, 컬럼 수가 61개이므로 60개의 속성과 1개의 클래스로 이루어져 있음을 짐작할 수 있음
   - 모든 컬럼이 실수형(float64)인데, 맨 마지막 컬럼만 객체형인 것으로 보아 마지막에 나오는 컬럼은 클래스이며 데이터형 변환이 필요한 것을 알 수 있음
    >> /ws_python/notebook/machine/sonar/sonar.csv

2. script
    - 훈련용, 검증용
    >> /ws_python/notebook/machine/sonar/Sonar1.ipynb

        ```python
        # 훈련용, 검증용
        df = pd.read_csv('./sonar.csv', header=None)
        # print(df.info())
        df.head()
        ```

        ```python
        data = df.values
        print(type(data))
        X = data[:, 0:60].astype(float)  # 0 ~ 59
        print(X[0:5, 0:4])
        Y_obj = data[:, 60]  # 1차원 배열
        print(Y_obj[0:5])
        ```

## [02] 과적합 피하기(훈련데이터, 검증 데이터의 분할)

- 과적합(over fitting)이란 모델이 학습 데이터셋 안에서는 일정 수준 이상의 예측 정확도를 보이지만, 새로운 데이터에 적용하면 잘 맞지 않는 것을 말함
![과적합](./images/05.jpg)  

- 학습 데이터와 테스트 데이터, 사용 데이터의 분리
![학습](./images/06_1.jpg)  

- 학습횟수를 늘린다고 모든 데이터의 정확도가 증가하지는 않음.
![학습횟수](./images/07.jpg)
  
- 세즈노프스키 교수가 논문에 포함한 실험 결과의 일부분
![논문](./images/08.jpg)  

- 세즈노프스키 교수가 논문에 포함한 실험 결과의 일부분 정리
- 학습셋의 훈련과 테스트셋의 검증 결과가 다름으로 학습시 데이터를 다양하게 사용할 필요가 있음.
![학습셋](./images/09.jpg)  

## [03] k겹 교차 검증(5-fold cross validation)

1. 데이터 셋을 여러 조각으로 나누어 검증하면 학습과 함께 모든 데이터를 검증 데이터로 이용하게되어 정확한 검증이 발생함.
    ![교차](./images/10.jpg)  

2. Script
    >> /ws_python/notebook/machine/sonar/Sonar2.ipynb
