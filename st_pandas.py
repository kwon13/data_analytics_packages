# 머신러닝에서 가장 중요한 데이터의 표현을 numpy를 통해 알 수 있었습니다.
# Pandas는 행과 열로 이루어진 2차원의 데이터를 쉽게 처리할 수 있도록 다양한 기능을 제공합니다. 넘파이에 대해 공부하고 보시는걸 추천드립니다!
# 실행하기 전 어떤 결과가 나올지 생각해 보세요!

# 우선 Pandas 모듈을 불러옵니다, pandas는 너무 길기 때문에 pd로 축약해서 표현합니다.
import pandas as pd

# 만약 모듈이 없다고 뜨면 터미널 PS C:\Users\파일이름> pip install pandas를 쳐주세요 

print('-----------------------------------------------------\n')
print('화이팅!!')

# 판다스는 다양한 파일을 데이터프레임으로 로딩할 수 있는 편리한 API 제공합니다.
# API가 무엇인지 고민하지 말고 코드를 통해 알아봅시다.

# read_csv(): 괄호 안에 파일 경로를 적으면 해당하는 csv파일을 로딩합니다.
# 예시로 깃허브에 있는 타이타닉 데이터를 로딩해 봅시다. 
titanic_df = pd.read_csv('https://raw.githubusercontent.com/HanXiaoyang/Kaggle_Titanic/master/train.csv')
print('titanic_df 변수 type: ', type(titanic_df))

print('-----------------------------------------------------\n')

# DataFrame.head(): 괄호 안의 인자 수만큼 순서대로 행을 출력합니다.
print(titanic_df.head(4))

print('-----------------------------------------------------\n')

# DataFrame.shape: 데이터프레임의 행과 열을 튜플 형태로 출력합니다.
print(titanic_df.shape)

print('-----------------------------------------------------\n')

# DataFrame.info(): 데이터프레임의 총 데이터건수, 데이터 타입, Null건수를 출력합니다.
# null은 비어있는 곳을 의미합니다.
print(titanic_df.info()) 

print('-----------------------------------------------------\n')

# DataFrame.describe(): 숫자형 데이터 값의 분포, 평균, 최댓값, 최솟값을 알려줍니다. (문자형은 출력에서 제외됩니다.)
# count값는 non-null개수를 의미합니다.
print(titanic_df.describe())

print('-----------------------------------------------------\n')

# Series.value_counts() : 지정된 컬럼의 데이터 값의 개수를 알려줍니다.
# -> value_counts()는 Series(데이터프레임의 하나의 열) 형태의 데이터에만 사용 가능합니다.
print(titanic_df['Survived'].value_counts())

print('-----------------------------------------------------\n')

# pandas의 데이터 프레임은 리스트와 넘파이의 ndarray와 다르게 column(열) 이름을 가지고 있습니다. 
# ex) 타이타닉 데이터의 Survived칼럼, Pclass칼럼...
# 하지만 리스트와 ndarray도 pandas를 이용하면 칼럼명을 생성할 수 있습니다.

# 넘파이의 ndarray로 예시
import numpy as np
col_name1 = ['col 1']
array1 = np.arange(1,4)
print('원래의 데이터:\n',array1)

# 이제 이 1차원의 ndarray를 이용해 데이터 프레임을 생성합니다.
df_array1 = pd.DataFrame(array1, columns=col_name1)
print('\n1차원 ndarray로 만든 DataFrame: \n', df_array1)

# 만약 2차원 형태의 ndarray를 이용하면 더 많은 칼럼을 지정할 수 있습니다.
col_name2 = ['col 1', 'col 2', 'col 3', 'col 4', 'col 5']
array2 = np.arange(1, 11)
array_2d = array2.reshape(2,5)
df_array_2d = pd.DataFrame(array_2d, columns=col_name2)
print('\n2차원 ndarray로 만든 DataFrame:\n', df_array_2d)

print('-----------------------------------------------------\n')

# 이번에는 반대로 DataFrame을 넘파이 ndarray로 바꿀 수 있습니다.

# DataFrame.values: 데이터 프레임을 ndarray로 바꿔줍니다.
# 방금 만든 df_arryay_2d를 바꿔보겠습니다. 
array3 = df_array_2d.values
print('df_array_2d.values shape: ', array3.shape)
print(array3)

print('-----------------------------------------------------\n')

# pandas를 이용하면 데이터프레임에 새로운 칼럼을 추가할 수 있습니다.
# 타이타닉 데이터에 Age_0이라는 새로운 칼럼을 추가하고, 일괄적으로 값을 0으로 할당해 보겠습니다.
titanic_df['Age_0'] = 0
print(titanic_df.head(3))

print('-----------------------------------------------------\n')

# 하지만 머신러닝에 필요하지 않는 데이터를 사용하면 효율이 떨어집니다.
#  방금 Age_0칼럼을 추가한 것처럼 삭제할 수도 있습니다.

# drop(): 데이터를 삭제하는데 사용됩니다. 
# DaataFrame.drop(labels="A", axis="B", index= , columns= , level= , inplace="C", errors=, )

# "A"의 값은 삭제할 컬럼명 지정, 여러개를 지울 때는 리스트로 넣을 수 있습니다.
# "B"의 값으로 0과 1이 있습니다.
# --> axis = 0: 해당하는 행(row)을 삭제합니다. 
# --> axis = 1: 해당하는 열(column)을 삭제합니다.
# "C"의 값으로 False와 True가 있습니다.
# --> inplace = False: 원본 데이터를 그대로 유지한 채 삭제합니다.
# --> inplace = True: 원본 데이터의 값을 삭제합니다.

# 코드를 보면서 이해해 보세요!

# 방금 추가한 Age_0 칼럼을 삭제하겠습니다.
titanic_drop_df = titanic_df.drop('Age_0', axis=1)
print(titanic_drop_df.head(3))

print('-----------------------------------------------------\n')

# 하지만 원본 titanic_df를 보면 Age_0칼럼이 삭제되지 않았습니다.
print(titanic_df.head(3))

print('-----------------------------------------------------\n')

# inplace = True 로 두어 원본 데이터프레임에서도 삭제합니다.
titanic_df.drop('Age_0', axis=1, inplace=True)
print(titanic_df.head(3))

print('-----------------------------------------------------\n')

# DataFrame.index 또는 Series.index 를 통해 인덱스 객체만 추출할 수 있습니다.

# Index 객체 추출
index = titanic_df.index

# reset_index(): 새로운 index를 연속 숫자형으로 할당하고, 기존 index는 index라는 새로운 컬럼명으로 추가합니다.
#-->기존 index가 연속형 int 숫자형이 아닐 경우 이를 다시 연속형 int 숫자로 만들 때 주로 사용합니다.
titanic_reset_df = titanic_df.reset_index(inplace=False)
print(titanic_reset_df.head(3))

print('-----------------------------------------------------\n')

# DataFrame은 여러 방법을 통해서 원하는 데이터를 선택할 수 있습니다!

# 1. 가장 기본적인 DataFrame[]방식
# 단일 칼럼 데이터를 추출합니다. 
print(titanic_df['Pclass'].head(8))
print()
# 여러 칼럼 데이터를 추출합니다. 여러 칼럼명을 리스트 형태로 작성합니다:['Survived', 'Pclass', 'Age']
print(titanic_df[['Survived', 'Pclass', 'Age']].head(8))
print()

# 2. DataFrame.iloc[로우 인덱스,칼럼 인덱스]방식
print(titanic_df.iloc[0, 3]) #Braund, Mr. Owen Harris출력
# ':'을 이용하면 여러 값도 불러올 수 있습니다.
print(titanic_df.iloc[0:4, 3:6])
print()

# 3. DataFrame.loc[로우 인덱스, 칼럼명]방식
# 특정 칼럼의 값도 불러올 수 있습니다.
print(titanic_df.loc[4, 'Name']) #Allen, Mr. William Henry출력
# 콜론을 이용하면 여러 값도 불러올 수 있지만 인덱스 값을 조심해야 합니다.
# 보통[2:7]-> 2부터 6까지, loc[2:7]-> 2부터 7까지
print(titanic_df.loc[2:7, 'Survived'])
print()

# 4. 불린 인덱싱 방식
print(titanic_df.loc[titanic_df['Age']>60,['Age', 'Survived']].head(3))
print()
# 여러 개의 복합 조건을 이용해서도 불린 인덱싱이 가능
#    and 조건.: &
#    or 조건 : |
#    Not 조건 : ~
print(titanic_df[ (titanic_df['Age']>60) & (titanic_df['Pclass']==1) & (titanic_df['Sex']=='female')])

print('-----------------------------------------------------\n')

# 넘파이의 sort()처럼 DataFrame을 sort_values()를 이용하여 정리할 수 있습니다.

titanic_sorted = titanic_df.sort_values(by=['Pclass'])
print(titanic_sorted.head(3)) # Pclass칼럼을 확인해 주세요!
print()
# 만약 내림차순으로 정리하고 싶으면 ascending=False 추가합니다.
titanic_sorted_down = titanic_df.sort_values(by=['Pclass'], ascending=False)
print(titanic_sorted_down.head(3))

print('-----------------------------------------------------\n')

# Aggregation 함수를 이용하면 각 칼럼의  평균, 최댓값, 총합, non-null값의 개수 집계를 알 수 있습니다.
# Aggregation 함수 종류 : min(), max(), sum(), count()
print('평균 나이:',titanic_df['Age'].mean(), '\n각 칼럼의 non-null값: \n',titanic_df[['Cabin', 'Embarked']].count())

print('-----------------------------------------------------\n')

# groupby(): 특정 칼럼에 대해 데이터를 정리하는 방법입니다.

# 좌석등급에 따라 데이터를 정리해 보겠습니다.
titanic_groupby = titanic_df.groupby('Pclass')[['Cabin', 'Embarked']].count()
print(titanic_groupby)
print()

# 여러개의 Aggregation함수를 적용하기 위해서는 agg를 이용합니다.
print(titanic_df.groupby('Pclass')['Age'].agg(['max', 'min']))
print()
# 여러개의 칼럼에 Aggregation함수를 적용하기 위해서는 agg인자에 딕셔너리를 사용합니다.
agg_format = {'Age':'max', 'Fare':'mean'}
print(titanic_df.groupby('Pclass').agg(agg_format))

print('-----------------------------------------------------\n')

# 넘파이에서 NaN은 판다스에서null값을 의미합니다. (null은 비어있는 값을 의미합니다.)
# 머신러닝에서 null은 문제가 되기 때문에 다른 값으로 대체해야합니다.

# isnull() 또는 isna()를 이용해 NaN값 여부를 확인합니다.
print(titanic_df.isnull().head(3))
print()
# True가 Null값이 있음을 의미합니다.

# sum()을 이용하여 한번에 확인합니다.
print(titanic_df.isnull().sum())

# Age와 Cabin에 비어있는 값이 있다는것을 확인할 수 있습니다.

print('-----------------------------------------------------\n')

# fillna()를 이용하여 비어있는 값을 채워줄 수 있습니다. (값을 변경하는 함수는 모두 inplace=True파라미터를 추가해야 원본데이터가 바뀝니다.)

# Age컬럼의 NaN를 평균값으로 대체하겠습니다.
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
print(titanic_df.isnull().sum())

# Age의 null값이 모두 채워진 것을 확인할 수 있습니다!

print('-----------------------------------------------------\n')

# pandas는 람다(lambda)를 사용할 수 있습니다.
# 타이타닉데이터에서 람다를 이용해 데이터 나이에 따른 분류 칼럼을 만들어 보겠습니다.

titanic_df['Child_Adult'] = titanic_df['Age'].apply(lambda x : 'Child' if x < 15 else 'Adult')
print(titanic_df[['Age', 'Child_Adult']].head(8))
# 여기서 if의 위치가 값 뒤에 있다는 것을 주의하세요! 

# head(), info(), drop(), DataFrame.loc[], Aggregation함수, groupby(), isnull(), fillna() 이것들이 판다스에서 어떤 역할을 하는지 확인해 보세요!