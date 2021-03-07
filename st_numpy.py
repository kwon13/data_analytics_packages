# 머신러닝 데이터는 배열(array)로 표현됩니다. 보편적으로 넘파이 배열로 표현되기 때문에 알면 좋은 패키지입니다.
# 파이썬의 리스트와 상당히 비슷하기 때문에 리스트에 대해서 먼저 공부하고 보는것을 권장합니다. 추가로 튜플이 무엇인지 알면 더 좋습니다! 
# 실행하기 전 어떤 결과가 나올지 생각해 보세요!

# 우선 넘파이 모듈을 불러옵니다, numpy는 너무 길기 때문에 np로 축약해서 표현합니다.
import numpy as np

# 만약 모듈이 없다고 뜨면 터미널 PS C:\Users\파일이름> pip install numpy를 쳐주세요 

print('-----------------------------------------------------\n')
print('화이팅!!')

# array()함수: 파이썬의 리스트와 같은 다양한 인자를 ndarray로 변환해 줍니다.
# -> 생성된 ndarray 배열의 형태는 행과 열의 수를 튜플 형태로 가지고 있고, 데이터 값은 자료, 숫자, 불 모두 가능합니다. 

array1 = np.array([1, 2])
#1차원 array
print('array1 type: ', type(array1))
print('array1의 array 형태', array1.shape)

array2 = np.array([[1, 2], [2, 3, 4]])
#2차원 array
print('array2 type: ', type(array2))
print('array2의 array 형태', array2.shape)

array3 = np.array([[1, 2, 3]])
#2차원 array
print('array3 type: ', type(array3))
print('array3의 array 형태', array3.shape)

array4 = np.array([[[1, 2, 3], [1, 2, 0], [8, 9, 7]]])
#3차원 array
print('array4 type: ', type(array4))
print('array4의 array 형태', array4.shape)

print('-----------------------------------------------------\n')

# 각 array의 차원을 ndim을 이용해 확인 
print('array1: {:0}차원'.format(array1.ndim))
print('array2: {:0}차원'.format(array2.ndim))
print('array3: {:0}차원'.format(array3.ndim))

# 가장 바깥 [](대괄호)의 개수에 따라 차원이 달라지는 것을 알 수 있습니다.
# 추후 다룰 CNN알고리즘의 이미지 데이터의 차원, 최적화를 위한 데이터 차원 낮추기 등 데이터의 차원 수를 인지하는 것은 매우 중요합니다!

print('-----------------------------------------------------\n')

# astype()를 이용하면 ndarray내의 데이터 타입 변경도 가능합니다.

number_array = np.array([1.0, 2.0, 3.0])
print('int로 바꾸기 전 \nnumber_array의 타입: {}'.format(number_array.dtype))
print('number_array의 형태: {}'.format(number_array))

print()
array_int = number_array.astype('int')
print('int로 바꾼 후 \nnumber_array의 타입: {}'.format(array_int.dtype))
print('number_array의 형태: {}'.format(array_int))

# 의미없는 소수점을 지워 메모리 절약 가능!

print('-----------------------------------------------------\n')

# arange()함수: 파이썬의 range함수와 비슷하게 값을 순차적으로 ndarray값으로 변환합니다.

zero_to_nine = np.arange(10)
print(zero_to_nine)
print(type(zero_to_nine))
print()
five_to_nine = np.arange(5, 10)
print(five_to_nine)
print(type(five_to_nine))

print('-----------------------------------------------------\n')

# zeros()함수, ones()함수: 튜플형태의 shape 값을 입력하면 모든 값을 (zeros함수는 0) (ones함수는 1)로 채운 ndarray로 변환합니다.

zero_array = np.zeros((5,), dtype='int')
print(zero_array)
print(zero_array.dtype)
print(zero_array.shape)
print()
#dtype값을 지정하지 않으면 float값으로 채워집니다
ones_array = np.ones((5,))
print(ones_array)
print(ones_array.dtype)
print(ones_array.shape)
print()

# 위의 zero_to_nine의 값을 다 0으로 바꿉니다
rp_o_array = np.zeros(zero_to_nine.shape, dtype='int')
print(rp_o_array)

print('-----------------------------------------------------\n')

# reshape()함수: ndarray의 차원과 크기를 변경합니다.

array_ex1 = np.arange(10)
print('array_ex1:\n', array_ex1)

# 쉽게 10 개의 값을 5개씩 2세트로 이해
array_ex2 = array_ex1.reshape(2, 5)
print('\narray_ex2:\n', array_ex2)
# 쉽게 10 개의 값을 2개씩 5세트로 이해
array_ex3 = array_ex1.reshape(5, 2)
print('\narray_ex3:\n', array_ex3)

# 만약 array_ex1.reshape(4, 2)처럼 변경이 불가능하면 오류를 발생시킵니다.

# -1은 알아서 호환되는 값으로 변경되어 ndarray로 만들어줍니다.
array_ex4 = array_ex1.reshape(-1, 5)
print('\narray_ex4:\n', array_ex4)
print('--->',array_ex4.shape, '\n여기서 -1이 2로 변경')
array_ex5 = array_ex1.reshape(2, -1)
print('\narray_ex5:\n', array_ex5)
print('--->',array_ex5.shape, '\n여기서 -1이 5로 변경')

# 마찬가지로 array_ex1.reshape(4, -1)처럼 변경이 불가능하면 오류를 발생시킵니다.

print('-----------------------------------------------------\n')

# ndarray의 데이터세트 선택하기

#-> ndarray에 해당하는 위치의 인덱스 값을 []안에 입력
# 3(0부터 시작이기 때문에 2)번째 값 추출
print(array_ex1[2])
print()

# array_ex1의 값을 수정할 수 있습니다. (첫 번째 값을 1로 바꾸기)
array_ex1[0]=8
print(array_ex1)
print()
array_ex1[0]=0

# 2차원의 ndarray의 경우 [row위치, column위치]로 로우와 칼럼의 위치 인덱스를 통해 접근이 가능합니다.
print(array_ex3)
print('(row=1, col=0) index 가리키는 값: ',array_ex3[1, 0])
print('(row=3, col=1) index 가리키는 값: ',array_ex3[3, 1])
print('(row=4, col=1) index 가리키는 값: ',array_ex3[4, 1])
print()

# ':'(콜론)을 이용하여 선택
zero_to_one = array_ex1[:2]
print(zero_to_one)
one_to = array_ex1[1:]
print(one_to)
print()

# 마찬가지로 2차원의 ndarray의 경우 [row위치, column위치]로 로우와 칼럼의 위치 인덱스를 통해 접근이 가능합니다.
print(array_ex3)
print('(row=1~2, col=0~1) index 가리키는 값: \n', array_ex3[1:3, 0:2])

print('-----------------------------------------------------\n')

# np.sort(), ndarray.sort()함수: 행렬을 정리하는 함수입니다.
org_array = np.array([3, 1, 8, 4])
print('원본 행렬:', org_array)
print()

# np.sort()함수를 사용하면 원래의 행렬은 그대로 유지한 채 정렬합니다
print(np.sort(org_array))
print('org_array가 바뀌지 않았습니다:',org_array)
print()

# ndarray.sort()함수를 사용하면 원래의 행렬을 정렬합니다.
print(org_array.sort())
print('org_array가 바뀌었습니다:',org_array)
print()

# 내림차순으로 정렬하기 위해서는 [::-1]을 적용하면 됩니다.
down_array = np.sort(org_array)[::-1]
print('내림차순으로 정렬되었습니다:',down_array)

print('-----------------------------------------------------\n')

# 행렬에 대해 이 블로그(https://blog.naver.com/hyunmonn/221989353051)에서 간단하게 알아보세요!

# np.dot()함수: 행렬의 내적을 계산합니다
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
dot_product = np.dot(A, B)
print('행렬 A와 B의 내적:\n',dot_product)

print('-----------------------------------------------------\n')

# np.transpose()함수: 원래의 행렬을 열 위치를 바꿉니다.

C = np.array([[1, 3], [4, 6]])
print('바꾸기 전:\n', C)
trans_C = np.transpose(C)
print('\n바꾼 후:\n', trans_C)
