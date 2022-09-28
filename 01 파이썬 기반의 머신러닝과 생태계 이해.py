#!/usr/bin/env python
# coding: utf-8

# ## [04] 데이터핸들링 - 판다스

# - 판다스(Pandas)는 파이썬에서 데이터 처리를 위해 존재하는 가장 인기 있는 라이브러리이다.
# - 일반적으로 대부분의 데이터 세트는 2차원 데이터이다. 즉, 행(Row)X열(Column)로 구성돼 있다.
# - 행과 열의 2차원 데이터가 인기 있는 이유는 바로 인간이 가장 이해하기 쉬운 데이터 구조이면서도 효과적으로 데이터를 담을 수 있는 구조이기 때문이다.
# - 넘파이는 저수준 API로 데이터 핸들링이 편하지 않다. 그러나 판다스는 많은 부분이 넘파이 기반으로 작성됐는데도 넘파이보다 훨씬 유연하고 편리하게 데이터 핸들링을 가능하게 해준다.
# - 판다사는 파이썬의 리스트, 컬렉션, 넘파이 등의 내부 데이터뿐만 아니라 CSV등의 파일을 쉽게 DataFrame으로 변경해 데이터의 가공/분석을 편리하게 수행할 수 있게 만들어준다.
# - 판다스의 핵심 객체는 DataFrame이다. DataFrame은 여러 개의 행과 열로 이뤄진 2차원 데이터를 담는 데이터 구조체이다.
# - Index는 RDBMS의 PK처럼 개별 데이터를 고유하게 식별하는 Key값이다.
# - Series와 DataFrame은 모두 Index를 Key값으로 가지고 있다.
# - Series와 DataFrame의 가장 큰 차이는 Series는 칼럼이 하나뿐인 데이터 구조체이고, DataFrame은 칼럼이 여러 개인 데이터 구조체이다.
# - DataFrame은 여러 개의 Series로 이뤄졌다고 할 수 있다.

# ### 판다스 시작 - 파일을 DataFrame으로 로딩, 기본 API

# In[1]:


import pandas as pd


# - pandas를 pd로 에일리어스(alias)해 임포트하는 것이 관례이다.

# - 판다스는 다양한 포맷으로 된 파일을 DataFrame으로 로딩할 수 있는 편리한 API를 제공한다.
# - 대표적으로 read_csv(), read_table(), read_fwf()가 있다.
# - read_csv()는 이름에서도 알 수 있듯이 CSV(칼럼을 ','로 구분한 파일 포맷) 파일 포맷 변환을 위한 API이다.
# - read_table()과 read_csv()의 가장 큰 차이는 필드 구분 문자(Delimeter)가 콤마(',')냐, 탭('\t')이냐의 차이다. read_table()의 디폴트 필드 구분 문자는 탭 문자이다.
# - read_csv()는 CSV뿐만 아니라 어떤 필드 구분 문자 기반의 파일 포맷도 DataFrame으로 변환이 가능하다. 
# - read_csv()의 인자인 sep에 해당 구분 문자를 입력하면 된다. 가령 탭으로 필드가 구분돼 있다면 read_csv('파일명',sep='\t')처럼 쓰면 된다.
# - read_csv()에서 sep 인자를 생략하면 자동으로 콤마로 할당한다(sep=','). 
# - read_fwf()는 Fixed Width, 즉 고정 길이 기반의 칼럼 포맷을 dataFrame으로 로딩하기 위한 API이다.
# - read_csv(filepath_or_buffer, sep=', ',...)함수에서 가장 중요한 인자는 filepath이다. 나머지 인자는 지정하지 않으면 디폴트 값으로 할당된다. 
# - Filepath에는 로드하려는 데이터 파일의 경로를 포함한 파일명을 입력하면 된다.

# In[2]:


titanic_df = pd.read_csv('/Users/hansohyeon/Desktop/Data/train.csv')
print('titanic 변수 type:', type(titanic_df))
titanic_df


# - pd.read_csv()는 호출 시 파일명 인자로 들어온 파일을 로딩해 DataFrame 객체로 반환한다.
# - 데이터 파일의 첫 번째 줄에 있던 칼럼 문자열이 DataFrame의 칼럼으로 할당됨.
# - read_csv()는 별다른 파라미터 지정이 없으면 파일의 맨 처음 로우를 칼럼명으로 인지하고 칼럼으로 변환한다. 그리고 콤마로 분리된 데이터값들이 해당 칼럼에 맞게 할당됐다.
# - 맨 왼쪽, 데이터값이 로우 순으로 0,1,2,3,...칼럼명도 표시되지 않는 데이터들은 판다스의 Index 객체 값이다.
# - 모든 DataFrame 내의 데이터는 생성되는 순간 고유의 Index 값을 가지게 된다.

# In[3]:


titanic_df.head(3)


# - DataFrame.head()는 DataFrame의 맨 앞에 있는 N개의 로우를 반환한다. head(3)은 맨 앞 3개의 로우를 반환한다. (Default는 5개이다.)

# In[4]:


print('DataFrame 크기: ',titanic_df.shape)


# - DataFrame 객체의 shape 변수를 이용하면 DataFrame의 행과 열 크기를 알 수 있다. 
# - shape는 DataFrame의 행과 열을 튜플 형태로 반환한다.

# - 생성된 DataFrame 객체인 titanic_df는 891개의 로우와 12개의 칼럼으로 이뤄짐.
# - DataFrame은 데이터뿐만 아니라 칼럼의 타입, Null 데이터 개수, 데이터 분포도 등의 메타 테이터 등도 조회가 가능하다.
# - 대표적인 메서드로 info()와 describe()가 있다.

# In[5]:


titanic_df.info()


# - info()메서드를 통해서 총 데이터 건수와 데이터 타입, Null 건수를 알 수 있다.

# - describe()메서드는 오직 숫자형(int, float 등) 칼럼의 분포도만 조사하며 자동으로 object 타입의 칼럼은 출력에서 제외시킨다.
# - 데이터의 분포도를 아는 것은 머신러닝 알고리즘의 성능을 향상시키는 중요한 요소이다.
# - 가령 회귀에서 결정 값이 정규 분포를 이루지 않고 특정 값으로 왜곡돼 있는 경우, 또는 데이터값에 시상치가 많을 경우 예측 성능이 저하된다.
# - describe() 메서드만으로 정확한 분포도를 알기는 무리지만, 개략적인 수준의 분포도를 확인할 수 있어 유용하다.
# - describe()메서드를 이용하면 숫자형 칼럼에 대한 개략적인 데이터 분포도를 확인할 수 있다.

# In[6]:


titanic_df.describe()


# - count는 Not Null인 데이터 건수, mean은 전체 데이터의 평균값, std는 표준편차, min은 최솟값, max는 최댓값이다. 그리고 25%는 25 percentile 값, 50%는 50 percentile 값, 75%는 75 percentile 값을 의미한다.
# - describe() 해당 숫자 칼럼이 숫자형 카테고리 칼럼인지를 판단할 수 있게 도와준다.
# - 카테고리 칼럼은 특정 범주에 속하는 값을 코드화한 칼럼이다.
# - ex)성별 칼럼의 경우 '남','여'가 있고 '남'을 1로 '여'를 2와 같이 표현한 것.
# - -> 이러한 카테고리 칼럼을 describe()를 이용해 숫자로 표시할 수 있다.
# - PassengerID 칼럼은 승객 ID를 식별하는 칼럼이므로 1~891까지 숫자가 할당되어서 분석을 위한 의미있는 속성이 아니다.
# - Survived의 경우 min 0, 25~75%도 0, max도 1이므로 0과 1로 이뤄진 숫자형 카테고리 칼럼일 것이다.
# - Pclass의 경우도 min이 1, 25~75%가 2와 3, max가 3이므로 1,2,3으로 이뤄진 숫자형 카테고리 칼럼일 것이다.

# - DataFrame의 [] 연산자 내부에 칼럼명을 입력하면 Series 형태로 특정 칼럼 데이터 세트가 반환된다.
# - 반환된 Series 객체에 value_counts() 메서드를 호출하면 해당 칼럼 값의 유형과 건수를 확인할 수 있다.
# - value_counts()는 지정된 칼럼의 데이터값 건수를 반환한다.
# - value_counts()는 데이터의 분포도를 확인하는 데 매우 유용한 함수이다.

# In[7]:


value_counts=titanic_df['Pclass'].value_counts()
print(value_counts)


# - value_counts()의 반환 결과는 Pclass값 3이 491개, 1이 216개, 2가 184개이다.
# - value_counts()는 많은 건수 순서로 정렬되어 값을 반환한다.
# - DataFrame의 [] 연산자 내부에 칼럼명을 입력하면 해당 칼럼에 해당하는 Series 객체를 반환한다.

# In[8]:


titanic_pclass = titanic_df['Pclass']
print(type(titanic_pclass))


# - Series는 Index와 단 하나의 칼럼으로 구성된 데이터 세트이다.
# - 전체 891개의 데이터를 추출하지 않고 head()메서드를 이용해 앞의 5개만 추출

# In[9]:


titanic_pclass.head()


# - 한 개 칼럼의 데이터만 출력되지 않고 맨 왼쪽에 0부터 시작하는 순차 값이 있다.
# - 이것은 DataFrame의 인덱스와 동일한 인덱스이다.
# - 오른쪽은 Series의 해당 칼럼의 데이터값이다.
# - 모든 DataFrame과 Series는 인덱스를 반드시 가진다.
# - 단일 칼럼으로 되어 있는 Series 객체에서 value_counts() 메서드를 호출하는 것이 칼럼별 데이터 값의 분포도를 좀 더 명시적으로 파악하기 쉽다.

# In[11]:


value_counts = titanic_df['Pclass'].value_counts()
print(type(value_counts))
print(value_counts)


# - value_counts()를 titanic_df의 Pclass 칼럼만을 값으로 가지는 Series 객체에서 호출한 반환값.
# - value_counts()가 반환하는 데이터 타입 역시 Series 객체이다.
# - 맨 왼쪽이 인덱스 값, 오른쪽은 데이터 값
# - 인덱스는 단순히 0부터 시작하는 순차 값이 아닌 Pclass 칼럼 값 (3,1,2)를 나타내고 있다.
# - 인덱스는 단순히 순차값과 같은 의미 없는 식별자만 할당하는 것이 아니라 고유성이 보장된다면 의미 있는 데이터값 할당도 가능하다.
# - value_counts()는 칼럼 값별 데이터 건수를 반환하므로 고유 칼럼 값을 식별자로 사용할 수 있다.
# - 인덱스는 DataFrame, Series가 만들어진 후에도 변경할 수 있다.
# - 인덱스는 또한 숫자형뿐만 아니라 문자열도 가능하다. 단, 모든 인덱스는 고유성이 보장돼야 한다.

# - value_counts() 메서드를 사용할 때는 Null 값을 무시하고 결괏값을 내놓기 쉽다는 점 유의해야한다.
# - value_counts()는 Null 값을 포함하여 개별 데이터 값의 건수를 계산할지를 dropna 인자로 판단한다.
# - dropna의 기본값은 True이며 Null 값을 무시하고 개별 데이터 값의 건수를 계산한다.

# In[12]:


print('titanic_df 데이터 건수:', titanic_df.shape[0])
print('기본 설정인 dropna=True로 value_counts()')
# value_counts()는 디폴트로 dropna=True이므로 value_counts(dropna=True)와 동일
print(titanic_df['Embarked'].value_counts())
print(titanic_df['Embarked'].value_counts(dropna=True))


# - 타이타닉 데이터 세트의 Embarked 칼럼의 데이터 값 분포를 value_counts()로 확인
# - 타이타닉 데이터 세트의 전체 건수 -> titanic_df.shape[0]으로 확인
# - Embarked 칼럼은 전체 891건 중에 2건의 데이터가 Null
# - Null값을 포함하여 value_counts()를 적용하고자 한다면 value_counts(cropna=False)

# ---

# ### DataFrame과 리스트, 딕셔너리, 넘파이 ndarray 상호 변환

# - DataFrame은 파이썬의 리스트, 딕셔너리 그리고 넘파이 ndarray 등 다양한 데이터로부터 생성될 수 있다.
# - DataFrame은 반대로 파이썬의 리스트, 딕셔너리 그리고 ndarray 등으로 변환될 수 있다.
# - 특히 사이킷런의 많은 API는 DataFrame을 인자로 입력받을 수 있지만, 기본적으로 넘파이 ndarray를 입력 인자로 사용하는 경우가 대부분이다.

# #### 넘파이 ndarray, 리스트, 딕셔너리를 DataFarame으로 변환하기

# - DataFrame은 리스트와 넘파이 ndarray와 다르게 칼럼명을 가지고 있다.
# - 이 칼럼명으로 인하여 리스트와 넘파이 ndarray보다 상대적으로 편하게 데이터 핸들링이 가능하다.
# - 일반적으로 DataFrame으로 변환 시 이 칼럼명을 지정해준다(지정하지 않으면 자동으로 칼럼명 할당).
# - 판다스 DataFrame 객체의 생성 인자 data는 리스트나 딕셔너리 또는 넘파이 ndarray를 입력받고, 생성 인자 columns는 칼럼명 리스트를 입력받아서 쉽게 DataFrame을 생성할 수 있다.
# - DataFrame은 기본적으로 행과 열을 가지는 2차원 데이터이다.
# - 2차원 이하의 데이터들만 DataFrame으로 변환될 수 있다.

# In[14]:


import numpy as np

col_name1=['col1']
list1 = [1,2,3]
array1 = np.array(list1)
print('array1 shape: ',array1.shape)
#리스트를 이용해 DataFrame 생성
df_list1 = pd.DataFrame(list1,columns=col_name1)
print('1차원 리스트로 만든 DataFrame:\n',df_list1)
#넘파이 ndarray를 이용해 DataFrame 생성
df_array1 = pd.DataFrame(array1,columns=col_name1)
print('1차원 리스트로 만든 DataFrame:\n',df_array1)


# - 1차원 형태의 리스트와 넘파이 ndarray부터 DataFrame 변환
# - 1차원 데이터이므로 칼럼은 1개만 필요하며, 칼럼명은 'col1'으로 지정
# - 1차원 형태의 데이터를 기반으로 DataFrame을 생성하므로 칼럼명이 한 개만 필요하다는 사실에 주의

# In[15]:


#3개의 칼럼명이 필요함.
col_name2 = ['col1','col2','col3']

#2행X3열 형태의 리스트와 ndarray 생성한 뒤 이를 DataFrame으로 변환
list2 = [[1,2,3],[11,12,13]]
array2 = np.array(list2)
print('array2 shape: ', array2.shape)
df_list2 = pd.DataFrame(list2, columns=col_name2)
print('2차원 리스트로 만든 DataFrame:\n',df_list2)
df_array2 = pd.DataFrame(array2,columns=col_name2)
print('2차원 ndarray로 만든 DataFrame:\n',df_array2)


# - 2행 3열 형태의 리스트와 ndarray를 기반으로 DataFrame을 생성하므로 칼럼명 3개 필요

# In[16]:


#key는 문자열 칼럼명으로 매핑, Values는 리스트 형(또는 ndarray) 칼럼 데이터로 매핑
dict = {'col1':[1,11],'col2':[2,12],'col3':[3,13]}
df_dict = pd.DataFrame(dict)
print('딕셔너리로 만든 DataFrame:\n',df_dict)


# - 일반적으로 딕셔너리를 DataFrame으로 변환 시에는 딕셔너리의 키(Key)는 칼럼명으로, 딕셔너리의 값(Value)은 해당하는 칼럼 데이터로 변환된다.
# - 키의 경우는 문자열, 값의 경우는 리스트(또는 ndarray) 형태로 딕셔너리를 구성해야 한다.

# #### DataFrame을 넘파이 ndarray, 리스트, 딕셔너리로 변환하기

# - 많은 머신러닝 패키지가 기본 데이터 형으로 넘파이 ndarray를 사용한다.
# - DataFrame을 넘파이 ndarray로 변환하는 것은 DataFrame 객체의 values를 이용해 쉽게 할 수 있다.

# In[17]:


#DataFrame을 ndarray로 변환
array3 = df_dict.values
print('df_dict.values 타입: ', type(array3), 'df_dict.values shape : ',array3.shape)
print(array3)


# - DataFrame -> ndarray로 변환

# In[19]:


#DataFrame을 리스트로 변환
list3 = df_dict.values.tolist()
print('df_dict.values.tolist() 타입 : ', type(list3))
print(list3)

#DataFrame을 딕셔너리로 변환
dict3 = df_dict.to_dict('list')
print('\n df_dict.to_dict() 타입:',type(dict3))
print(dict3)


# - DataFrame -> 리스트와 딕셔너리로 변환
# - 리스트로의 변환은 values로 얻은 ndarray에 tolist()를 호출
# - 딕셔너리로의 변환은 DataFrame 객체의 to_dict() 메서드를 호출하는데, 인자로 'list'를 입력하면 딕셔너리의 값이 리스트형으로 반환된다.

# ### DataFrame의 칼럼 데이터 세트 생성과 수정

# - DataFrame의 칼럼 데이터 세트 생성과 수정 역시 []연산자를 이용해 쉽게 할 수 있다.
# - Titanic DataFrame의 새로운 칼럼 Age_0을 추가하고 일괄적으로 0값을 할당
# - DataFrame[]내에 새로운 칼럼명을 입ㄹㄱ하고 값을 할당해주기만 하면 된다

# In[20]:


titanic_df['Age_0'] = 0
titanic_df.head(3)


# - 새로운 칼럼명 'Age_0'으로 모든 데이터값이 0으로 할당된 Series가 기존 DataFrame에 추가됨

# In[21]:


titanic_df['Age_by_10'] = titanic_df['Age']*10
titanic_df['Family_No'] = titanic_df['SibSp']+titanic_df['Parch']+1
titanic_df.head(3)


# - 기존 칼럼 Series를 가공해 새로운 칼럼 Series인 Age_by_10과 Family_No가 새롭게 DataFrame에 추가됨
# - DataFrame 내의 기존 칼럼 값도 쉽게 일괄적으로 업데이트 할 수 있다.
# - 업데이트를 원하는 칼럼 Series를 DataFrame[]내에 칼럼명으로 입력한 뒤에 값을 할당해주면 된다.

# In[22]:


titanic_df['Age_by_10']=titanic_df['Age_by_10']+100
titanic_df.head(3)


# - Age_by_10 칼럼의 모든 값이 기존 값+100으로 업데이트 됨

# ### DataFrame 데이터 삭제

# - DataFrame에서 데이터 삭제는 drop()메서드를 이용한다.
# - DataFrame.drop(**labels=None,axis=0**,index=None,columns=None,level=None,**inplace=False**,error='raise')
# - 가장 중요한 파라미터는 labels, axis, inplace이다.
# - axis 값에 따라서 특정 칼럼 또는 특정 행을 드롭한다.
# - axis0은 로우 방향 축, axis1은 칼럼 방향 축 (판다스의 DataFrame은 2차원 데이터만 다루므로 axis0, axis1로만 axis가 구성돼있다.)
# - drop()메서드에 axis=1을 입력하면 칼럼 축 방향으로 드롭을 수행하므로 칼럼을 드롭하겠다는 의미.
# - labels에 원하는 칼럼명을 입력하고 axis=1을 입력하면 지정된 칼럼을 드롭한다.
# - drop()메서드에 axis=0을 입력하면 로우 축 방향으로 드롭을 수행하므로 칼럼을 드롭하겠다는 의미.
# - DataFrame의 특정 로우를 가리키는 것은 인덱스이다. 따라서 axis를 0으로 지정하면 DataFrame은 자동으로 labels에 오는 값을 인덱스로 간주한다.

# In[23]:


titanic_drop_df = titanic_df.drop('Age_0',axis=1)
titanic_drop_df.head(3)


# - titanic_df.drop('Age_0',axis=1)을 수행한 결과가 titanic_drop_df 변수로 반환됐다.
# - titanic_drop_df 변수의 결과를 보면 'Age_0' 칼럼이 삭제됨

# In[24]:


titanic_df.head(3)


# - 삭제됐다고 생각했던 'Age_0' 칼럼이 여전히 존재
# - 앞선 코드에서 inplace=False로 설정했기 때문 (inplace는 디폴트 값이 False이므로 inplace 파라미터를 기재하지 않으면 자동으로 False가 된다).
# - inplace=False이면 자기 자신의 DataFrame의 데이터는 삭제하지 않으며, 삭제된 결과 DataFrame을 반환한다.
# - inplace=True로 설정하면 자신의 DataFrame의 데이터를 삭제한다. 또한 여러개의 칼럼을 삭제하고 싶으면 리스트 형태로 삭제하고자 하는 칼럼명을 입력해 labels 파라미터로 입력하면 된다.

# In[25]:


drop_result = titanic_df.drop(['Age_0','Age_by_10','Family_No'], axis=1, inplace=True)
print('inplace=True로 drop 후 반환된 값 : ',drop_result)
titanic_df.head(3)


# - 'Age_0','Age_by_10','Family_No' drop 결과
# - drop()시 inplace=True로 설정하면 반환값이 None(아무 값도 아님)이 된다.
# - inplace=True로 설정한 채로 반환 값을 다시 자신의 DataFrame 객체로 할당하면 안된다.

# In[26]:


pd.set_option('display.width',1000)
pd.set_option('display.max_colwidth',15)
print('#### before axis 0 drop ####')
print(titanic_df.head(3))

titanic_df.drop([0,1,2],axis=0,inplace=True)

print('#### after axis 0 drop ####')
print(titanic_df.head(3))


# - axis=0으로 설정한 뒤 index 0,1,2(맨 앞 3개 데이터)로우 삭제

# - axis : DataFrame의 로우를 삭제할 때는 axis=0, 칼럼을 삭제할 때는 axis=1으로 설정.
# - 원본 DataFrame은 유지하고 드롭된 DataFrame을 새롭게 객체 변수로 받고 싶다면 inplace=False로 설정(디폴트 값이 False임). (ex: titanic_drop_df = titanic_df.drop('Age_0',axis=1,inplace=False)
# - 원본 DataFrame에 드롭된 결과를 적용할 경우에는 inplace=True를 적용 (ex: titanic_df.drop('Age_0',axis=1,inplace=True)
# - 원본 DataFrame에서 드롭된 DataFrame을 다시 원본 DataFrame 객체 변수로 할당하면 원본 DataFrame에서 드롭된 결과를 적용할 경우와 같음(단, 기존 원본 DataFrame 객체 변수는 메모리에서 추후 제거됨). (ex: titanic_df = titanic_df.drop('Age_0',axis=1,inplace=False)

# ### Index 객체

# - 판다스의 Index 객체는 RDBMS의 PK(Primary Key)와 유사하게 DataFrame, Series의 레코드를 고유하게 식별하는 객체이다.
# - DataFrame, Series에서 Index 객체만 추출하려면 DataFrame.index 또는 Series.index 속성을 통해 가능하다.

# In[28]:


#원본 파일 다시 로딩
titanic_df = pd.read_csv('/Users/hansohyeon/Desktop/Data/train.csv')
#Index 객체 추출
indexes = titanic_df.index
print(indexes)
#Index 객체를 실제 값 array로 변환
print('Index 객체 array값:\n',indexes.values)


# - 반환된 Index 객체의 실제 값은 넘파이 1차원 ndarray
# - Index 객체의 values 속성으로 ndarray 값을 알 수 있다.
# - Index 객체는 식별성 데이터를 1차원 array로 가지고 있다. 또한 ndarray와 유사하게 단일 값 반환 및 슬라이싱도 가능하다.

# In[30]:


print(type(indexes.values))
print(indexes.values.shape)
print(indexes[:5].values)
print(indexes.values[:5])
print(indexes[6])


# In[31]:


indexes[0] = 5


# - 한번 만들어진 DataFrame 및 Series의 Index 객체는 함부로 변경이 불가하다.

# In[32]:


series_fair = titanic_df['Fare']
print('Fair Series max 값: ',series_fair.max())
print('Fair Series sum 값: ',series_fair.sum())
print('sum() Fair Series: ',sum(series_fair))
print('Fair Series +3:\n ',(series_fair+3).head(3))


# - Series 객체는 Index 객체를 포함하지만 Series 객체에 연산 함수를 적용할 때 Index는 연산에서 제외한다. 
# - Index는 오직 식별용으로만 사용

# In[33]:


titanic_reset_df = titanic_df.reset_index(inplace=False)
titanic_reset_df.head(3)


# - DataFrame 및 Series에 reset_index() 메서드를 수행하면 새롭게 인덱스를 연속 숫자 형으로 할당하며 기존 인덱스는 'index'라는 새로운 칼럼명으로 추가한다.
# - reset_index()는 인덱스가 연속된 int 숫자형 데이터가 아닐 경우에 다시 이를 연속 int 숫자형 데이터로 만들 때 주로 사용한다.
# - 예를들어 Series와 value_counts()를 소개하는 예제에서 'Pclass'칼럼 Series의 value_counts()를 수행하면 'Pclass' 고유 값이 식별자 인덱스 역할을 했다.
# - Series에 reset_index()를 적용하면 Series가 아닌 DataFrame이 반환되니 유의하자.
# - 기존 인덱스가 칼럼으로 추가돼 칼럼이 2개가 되므로 Series가 아닌 DataFrame이 반환된다.

# In[34]:


print('### before reset_index ###')
value_counts=titanic_df['Pclass'].value_counts()
print(value_counts)
print('value_counts 객체 변수 타입: ', type(value_counts))
new_value_counts = value_counts.reset_index(inplace=False)
print('### After reset_index ###')
print(new_value_counts)
print('new_value_counts 객체 변수 타입: ',type(new_value_counts))


# - Series에 reset_index()를 적용하면 새롭게 연속 숫자형 인덱스가 만들어지고 기존 인덱스는 'index'칼럼명으로 추가되면서 DataFrame으로 변환됨을 알 수 있다.
# - reset_index()의 parameter 중 drop=True로 설정하면 기존 인덱스는 새로운 칼럼으로 추가되지 않고 삭제(drop)된다. -> 새로운 칼럼으로 추가되지 않으므로 그대로 Series로 유지된다.

# ### 데이터 셀렉션 및 필터링

# - 넘파이의 경우 '[ ]'연산자 내 단일 값 추출, 슬라이싱, 팬시 인덱싱, 불린 인덱싱을 통해 데이터를 추출
# - 판다스의 경우 iloc[ ],loc[ ]연산자를 통해 동일한 작업을 수행

# #### DataFrame의 [ ]연산자

# - 넘파이 [ ]연산자 : 행의 위치, 열의 위치, 슬라이싱 범위 등을 지정해 데이터를 가져올 수 있음
# - DataFrame [ ]연산자 : 칼럼명 문자(또는 칼럼명의 리스트 객체), 또는 인덱스로 변환 가능한 표현식 = DataFrame 뒤에 있는 [ ]는 칼럼만 지정할 수 있는 '칼럼 지정 연산자'

# In[35]:


print('단일 칼럼 데이터 추출:\n', titanic_df['Pclass'].head(3))
print('\n여러 칼럼의 데이터 추출:\n', titanic_df[['Survived','Pclass']].head(3))
print('[]안에 숫자 index는 KeyError 오류 발생:\n',titanic_df[0])


# - DataFrame에 ['칼럼명']으로 '칼럼명'에 해당하는 칼럼 데이터의 일부만 추출
# - 여러 개의 칼럼에서 데이터를 추출하려면 ['칼럼1','칼럼2']와 같이 리스트 객체 이용
# - titanic_df[0] 같은 표현식은 오류 발생 -> DataFrame 뒤의 [ ]에는 칼럼명을 지정해야 하는데, 0은 칼럼명이 아님.

# In[36]:


titanic_df[0:2]


# - 판다스의 인덱스 형태로 변환 가능한 표현식은 [ ]내에 입력할 수 있다.

# In[37]:


titanic_df[titanic_df['Pclass']==3].head(3)


# - 불린 인덱싱 표현도 가능
# - [ ]내의 불린 인덱싱 기능은 원하는 데이터를 편리하게 추출해준다.

# - DataFrame 바로 뒤의 [ ] 연산자는 넘파이의 [ ]나 Series[ ]와 다르다.
# - DataFrame 바로 뒤의 [ ] 내 입력값은 칼럼명(또는 칼럼의 리스트)을 지정해 칼럼 지정 연산에 사용하거나 불린 인덱스 용도로만 사용해야 한다.
# - DataFrame[0:2]와 같은 슬라이싱 연산으로 데이터를 추출하는 방법은 사용하지 않는게 좋다

# #### DataFrame iloc[ ] 연산자

# - 판다스는 DataFrame의 로우나 칼럼을 지정하여 데이터를 선택할 수 있는 인덱싱 방식으로 iloc[ ]와 loc[ ]를 제공한다.
# - iloc[ ]는 위치(Location) 기반 인덱싱 방식으로 동작하며 loc[ ]는 명칭(Label)기반 인덱싱 방식으로 동작한다.
# - 위치 기반 인덱싱: 행과 열의 위치를, 0을 출발점으로 하는 세로축, 가로축 좌표 정숫값으로 지정하는 방식.
# - 명칭 기반 인덱싱: 데이터 프레임의 인덱스 값으로 행 위치를, 칼럼의 명칭으로 열 위치를 지정하는 방식
# 
# - iloc[ ]는 위치 기반 인덱싱만 허용하기 때문에 행과 열의 좌표 위치에 해당하는 값으로 정숫값 또는 정수형의 슬라이싱, 팬시 리스트 값을 입력해줘야 한다.

# In[38]:


data = {'Name':['Chulimin','Eunkyung','Jinwoong','Soobeom'],
        'Year':[2011,2016,2015,2015],
        'Gender':['Male','Female','Male','Male']
}
data_df = pd.DataFrame(data,index=['one','two','three','four'])
data_df


# In[39]:


# data_df = DataFrame
# data_df의 첫 번째 행, 첫 번째 열의 데이터 iloc[]을 이용해 추출
data_df.iloc[0,0]


# - iloc[행 위치 정숫값, 열 위치 정ㅇ숫값]과 같이 명확하게 DataFrame의 행 위치와 열 위치를 좌표 정숫값 형태로 입력하여 해당 위치에 있는 데이터를 가져올 수 있다.

# In[40]:


#열 위치에 위치 정숫값이 아닌 칼럼 명칭 입력
data_df.iloc[0,'Name']


# In[41]:


# 행 위치에 DataFrame의 인덱스 값 입력
data_df.iloc['one',0]


# - 반면에 iloc[ ]에 DataFrame의 인덱스 값이나 칼럼명을 입력하면 오류 발생

# In[42]:


data_df.iloc[1,0]


# - 두번째 행의 첫번째 열 위치에 있는 단일 값 반환

# In[43]:


data_df.iloc[2,1]


# - 세번째 행의 두번째 열 위치에 있는 단일 값 반환

# In[44]:


data_df.iloc[0:2,[0,1]]


# - 0:2 슬라이싱 범위의 첫번째에서 두번째 행과 첫번째, 두번째 열에 해당하는 DataFrame 반환

# In[45]:


data_df.iloc[0:2,0:3]


# - 0:2 슬라이싱 범위의 첫번째에서 두번째 행의 0:3 슬라이싱 범위의 첫번째부터 세번째 열 범위에 해당하는 DataFrame 반환

# In[46]:


data_df.iloc[:]


# - 전체 DataFrame 반환

# In[47]:


data_df.iloc[:,:]


# - 전체 DataFrame 반환

# In[48]:


print('\n 맨 마지막 칼럼 데이터 [:,-1]\n',data_df.iloc[:,-1])
print('\n 맨 마지막 칼럼을 제외한 모든 데이터 [:,:-1]\n',data_df.iloc[:,:-1])


# - iloc[ ]는 열 위치에 -1을 입력하여 DataFrame의 가장 마지막 열 데이터를 가져오는 데 자주 사용한다.
# - 넘파이와 마찬가지로 판다스의 인덱싱에서도 -1은 맨 마지막 데잍 값을 의미한다.
# - 특히 머신러닝 학습 데이터의 맨 마지막 칼럼이 타깃 값인 경우가 많은데, 이 경우 iloc[:-1]을 하면 맨마지막 칼럼의 값, 즉 타깃 값을 가져오고, iloc[:,:-1]을 하게 되면 처음부터 맨 마지막 칼럼을 제외한 칼럼의 값, 즉 피처값들을 가져 오게 된다.
# - iloc[ ]는 슬라이싱과 팬시 인덱싱은 제공하나 명확한 위치 기반 인덱싱이 사용되어야 하는 제약으로 인해 불린 인덱싱은 제공하지 않는다.

# #### DataFrame loc[ ] 연산자

# - loc[ ]는 명칭(Label) 기반으로 데이터를 추출한다.
# - 행 위치에는 DataFrame 인덱스 값을, 열 위치에는 칼럼명 입력 = loc[인덱스값, 칼럼명]

# In[49]:


data_df.loc['one','Name']


# - 인덱스 값을 DataFrame의 행 위치를 나타내는 고유한 '명칭'으로 생각하면 의미적으로 한결 이해하기 쉽다.
# - '명칭기반'이라는 문맥적 의미 때문에 loc[ ]를 사용 시 행 위치에 DataFrame의 인덱슥 ㅏㅂㅅ이 들어가는 부분을 간과할 수 있다.
# - 일반적으로 인덱스는 0부터 시작하는 정숫값인 경우가 많기 때문에 실제 인덱스 값을 확인하지 않고 loc[ ]의 행 위치에 무턱대고 정숫값을 입력해서 오류가 나는 경우들이 종종 발생한다.

# In[50]:


#다음 코드는 오류를 발생시킨다.
data_df.loc[0,'Name']


# - loc[ ]에 슬라이싱 기호':'를 적용할 때 한가지 유의할 점이 있다.
# - 일반적으로 슬라이싱을 '시작값':'종료값'과 같이 지정하면 시작 값 ~ 종료 값-1까지의 범위를 의미한다.
# - 그러나 loc[ ]에 슬라이싱 기호를 적용하면 종료값-1이 아니라 종료 값까지 포함하는 것을 의미.
# - 명칭은 숫자형이 아닐 수 있기 때문에 -1을 할 수가 없다.

# In[51]:


print('위치 기반 iloc slicing\n',data_df.iloc[0:1,0],'\n')
print('명칭 기반 loc slicing\n',data_df.loc['one':'two','Name'])


# - 위치기반 iloc[0:1]은 0번째 행 위치에 해당하는 1개의 행을 반환
# - 명칭기반 loc['one':'two']는 2개의 행 반환

# In[52]:


data_df.loc['three','Name']


# - 인덱스 값 three인 행의 Name 칼럼의 단일값 반환

# In[53]:


data_df.loc['one':'two',['Name','Year']]


# - 인덱스 값 one부터 two까지 행의 Name과 Year 칼럼에 해당하는 DataFrame 반환

# In[54]:


data_df.loc['one':'three','Name':'Gender']


# - 인덱스 값 One부터 three까지 행의 Name부터 Gender 칼럼까지의 DataFrame 반환

# In[55]:


data_df.loc[:]


# - 모든 데이터 값 반환

# In[56]:


data_df.loc[data_df.Year >= 2014]


# - iloc[ ]와 다르게 loc[ ]는 불린 인덱싱이 가능. Year 칼럼의 값이 2014 이상인 모든 데이터를 불린 인덱싱으로 추출

# 1. 개별 또는 여러 칼럼 값 전체를 추출하고자 한다면 iloc[ ]나 loc[ ]을 사용하지 않고 DataFrame['칼럼명']만으로 충분. 하지만 행과 열을 함께 사용하여 데이터를 추출해야한다면 iloc[ ]나 loc[ ]를 사용해야 한다.
# 2. iloc[ ]와 loc[ ]을 이해하기 위해서는 명칭 기반 인덱싱과 위치 기반 인덱싱의 차이를 먼저 이해해야한다. DataFrame의 인덱스나 칼럼명으로 데이터에 접근하는 것은 명칭 기반 인덱싱. 0부터 시작하는 행, 열의 위치 좌표에만 의존하는 것은 위치 기반 인덱싱
# 3. iloc[ ]는 위치 기반 인덱싱만 가능하다. 따라서 행과 열 위치 값으로 정수형 값을 지정해 원하는 데이터를 반환한다.
# 4. loc[ ]는 명칭 기반 인덱싱만 가능하다. 따라서 행 위치에 DataFrame 인덱스가 오며, 열 위치에는 칼럼명을 지정해 원하는 데이터를 반환한다.
# 5. 명칭 기반 인덱싱에서 슬라이싱을 '시작점:종료점'으로 지정할 때 시작점에서 종료점을 모두 포함한 위치레 있는 데이터를 반환한다.

# #### 불린 인덱싱

# - 처음부터 가져올 값을 조건으로 [ ]내에 입력하면 자동으로 원하는 값을 필터링한다.
# - [ ], loc[ ]에서 공통으로 지원한다. 단지 iloc[ ]는 정수형 값이 아닌 불린 값에 대해서는 지원하지 않기 대문에 불린 인덱싱이 지원되지 않는다.

# In[57]:


titanic_df = pd.read_csv('/Users/hansohyeon/Desktop/Data/train.csv')
titanic_boolean = titanic_df[titanic_df['Age']>60]
print(type(titanic_boolean))
titanic_boolean


# - 승객 중 나이(Age)가 60세 이상인 데이터 추출
# - 반환된 titanic_boolean 객체의 타입은 DataFrame
# - [ ]내에 불린 인덱싱을 적용하면 반환되는 객체가 DataFrame이므로 원하는 칼럼명만 별도로 추출할 수 있다.

# In[58]:


titanic_df[titanic_df['Age']>60][['Name','Age']].head(3)


# - 60세 이상인 승객의 나이와 이름만 추출

# In[59]:


titanic_df.loc[titanic_df['Age']>60,['Name','Age']].head(3)


# - loc[ ] 이용

# - 여러개의 복잡한 조건들 이용 가능
# 1. and 조건일 때는 &
# 2. or 조건일 때는 |
# 3. Not 조건일 때는 ~

# In[60]:


titanic_df[(titanic_df['Age']>60)&(titanic_df['Pclass']==1)&(titanic_df['Sex']=='female')]


# - 나이 60세 이상 & Pclass = 1 & 여성 데이터 추출

# In[63]:


cond1 = titanic_df['Age']>60
cond2 = titanic_df['Pclass']==1
cond3 = titanic_df['Sex']=='female'
titanic_df[cond1&cond2&cond3]


# ### 정렬, Aggregation 함수, GroupBy 적용

# #### DataFrame, Series 정렬 - sort_values( )

# - DataFrame과 Series의 정렬을 위해서는 sort_values()메서드 사용
# - sort_values()는 RDBMS SQL의 order by 키워드와 매우 유사.
# - sort_values()의 주요 입력 파라미터는 by,ascending,inplace이다.
# - by로 특정 칼럼을 입력하면 해당 칼럼으로 정렬을 수행
# - ascneding=True로 설정하면 오름차순으로 정렬
# - inplace=False로 설정하면 sort_values()를 호출한 DataFrame은 그대로 유지하며 DataFrame을 결과로 반환한다.
# - inplace=True로 설정하면 호출한 DataFrame의 정렬 결과를 그대로 적용한다. (기본은 inplace=False)

# In[64]:


titanic_sorted=titanic_df.sort_values(by=['Name'])
titanic_sorted.head(3)


# - titanic_df를 Name 컬럼으로 오름차순 정렬

# In[65]:


titanic_sorted=titanic_df.sort_values(by=['Pclass','Name'],ascending=False)
titanic_sorted.head(3)


# #### Aggregation 함수 적용

# - DataFrame에서 min( ), max( ), sum( ), count( )와 같은 aggregation 함수의 적용은 RDBMS SQL의 aggregation 함수 적용과 유사하다.
# - DataFrame의 경우 DataFrame에서 바로 aggregation을 호출할 경우 모든 칼럼에 해당 aggregation을 적용한다는 차이가 있다.

# In[66]:


titanic_df.count()


# - titanic_df에 count( )을 적용하면 모든 칼럼에 count( )결과를 반환한다.
# - 단, count( )는 Null 값을 반영하지 않은 결과를 반환한다.
# - 때문에 Null값이 있는 Age, Cabin, Embarked 칼럼은 count( )의 결과값이 다르다.

# In[67]:


titanic_df[['Age','Fare']].mean()


# - 특정 칼럼에 aggregation 함수를 적용하기 위해서는 DataFrame에 대상 칼럼들만 추출해 aggregation을 적용하면 된다.

# #### groupby( ) 적용

# - DataFrame의 groupby( )는 RDBMS SQL의 groupby 키워드와 유사하면서도 다르다.
# - DataFrame의 groupby( ) 사용 시 입력 파라미터 by에 칼럼을 입력하면 대상 칼럼으로 groupby된다.
# - DataFrame에 groupby( )를 포출하면 DataFrameGroupBy라는 또 다른 형태의 DataFrame을 반환한다.

# In[68]:


titanic_groupby = titanic_df.groupby(by='Pclass')
print(type(titanic_groupby))


# - DataFrame에 groupby( )를 호출해 반환된 결과에 aggregation 함수를 호출하면 groupby( )대상 칼럼을 제외한 모든 칼럼에 해당 aggregation함수를 적용한다.

# In[69]:


titanic_groupby = titanic_df.groupby('Pclass').count()
titanic_groupby


# - DataFrame의 groupby( )에 특정 칼럼만 aggregation 함수를 적용하려면 groupby( )로 반환된 DataFrameGroupBy 객체에 해당 칼럼을 필터링한 뒤 aggregation 함수를 적용한다.

# In[71]:


titanic_groupby = titanic_df.groupby('Pclass')[['PassengerId','Survived']].count()
titanic_groupby


# - titanic_df.groupby('Pclass')로 반환된 DataFrameGroupBy 객체에 [['PassengerId','Survived']]로 필터링 해 PassengerId와 Survived 칼럼에만 count를 수행

# - DataFrame groupby( )의 경우 적용하려는 여러 개의 aggregation 함수명을 DataFrameGroupBy 객체의 agg( )내에 인자로 입력해서 사용하면 된다.

# In[72]:


titanic_df.groupby('Pclass')['Age'].agg([max,min])


# - DataFrame의 groupby( )를 이용해 API 기반으로 처리하다 보니 SQL의 group by보다 유연성이 떨어질 수 밖에 없다.
# - DataFrame groupby( )는 좀 더 복잡한 처리가 필요하다.
# - groupby( )는 agg( )를 이용해 이 같은 처리가 가능한데, agg( )내에 입력값으로 딕셔너리 형태로 aggregation이 적용될 칼럼들과 aggregation 함수를 입력한다.

# In[73]:


agg_format={'Age':'max','SibSp':'sum','Fare':'mean'}
titanic_df.groupby('Pclass').agg(agg_format)


# ### 결손 데이터 처리하기

# - 판다스는 결손 데이터(Missing Data)를 처리하는 편리한 API를 제공한다.
# - 결손 데이터는 칼럼에 값이 없는, 즉 NULL인 경우를 의미하며, 이를 넘파이의 NaN으로 표시한다.
# - 기본적으로 머신러닝 알고리즘은 이 NaN 값을 처리하지 ㅇ낳으므로 이 값을 다른 값으로 대체해야 한다.
# - 또한 NaN 값은 평균, 총합 등의 함수 연산 시 제외가 된다.
# - 특정 칼럼의 100개 데이터 중 10개가 NaN 값일 경우, 이 칼럼의 평균 값은 90개 데이터에 대한 평균이다.
# - NaN 여부를 확인하는 API는 isna( )이며, NaN 값을 다른 값으로 대체하는 API는 fillna( )이다.

# #### isna( )로 결손 데이터 여부 확인

# - isna( )는 데이터가 NaN인지 아닌지 알려준다.
# - DataFrame에 isna( )를 수행하면 모든 칼럼의 값이 NaN인지 아닌지를 True나 False로 알려준다.

# In[74]:


titanic_df.isna().head(3)


# In[75]:


titanic_df.isna().sum()


# - 결손 데이터의 개수는 isna( ) 결과에 sum( )함수를 추가해 구할 수 있다.
# - sum( )을 호출 시 True는 내부적으로 숫자 1로 False는 숫자 0으로 변환되므로 결손 데이터의 개수를 구할 수 있다.

# #### fillna( )로 결손 데이터 대체하기

# - fillna( )를 이용하면 결손 데이터를 편리하게 다른 값으로 대체할 수 있다.

# In[76]:


titanic_df['Cabin'] = titanic_df['Cabin'].fillna('C000')
titanic_df.head(3)


# - 타이타닉 데이터 세트의 'Cabin' 칼럼의 NaN 값을 'C000'으로 대체

# - 주의해야 할 점은 fillna( )를 이용해 반환 값을 다시 받거나 inplace=True 파라미터를 fillna( )에 추가해야 실제 데이터 세트 값이 변경된다는 점이다.

# In[77]:


titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')
titanic_df.isna().sum()


# - 'Age'칼럼의 NaN 값을 평균 나이로, 'Embarked'칼럼의 NaN값을 'S'로 대체해 모든 결손 데이터 처리

# ### apply lambda 식으로 데이터 가공

# - 판다스의 경우 칼럼에 일괄적으로 데이터 가공을 하는 것이 속도 면에서 더 빠르거나 복잡한 데이터 가공이 필요할 경우 어쩔 수 없이 apply lambda를 이용한다.
# - lambda 식은 파이썬에서 함수형 프로그래밍(functional programming)을 지원하기 위해 만들었다.
# - lambda는 함수의 선언과 함수 내의 처리를 한 줄의 식으로 쉽게 변환하는 식이다.
# - ex) lambda x : x^2에서 ':'로 입력 인자와 반환될 입력 인자의 계산식을 분리한다. ':'의 왼쪽에 있는 x는 입력인자를 가리키며, 오른쪽은 입력 인자의 계산식이다. 오른쪽의 계산식은 결국 반환값을 의미한다.
# - lambda식을 이용할 때 여러 개의 데이터 값을 입력 인자로 사용해야 할 경우에는 보통 map( )함수를 결합해서 사용한다.

# In[78]:


titanic_df['Name_len'] = titanic_df['Name'].apply(lambda x : len(x))
titanic_df[['Name','Name_len']].head(3)


# - DataFrame의 apply에 lambda식 적용
# - 'Name' 칼럼의 문자 열 개수를 별도의 칼럼인 'Name_len'에 생성

# In[79]:


titanic_df['Child_Adult']=titanic_df['Age'].apply(lambda x:'Child' if x<=15 else 'Adult')
titanic_df[['Age','Child_Adult']].head(8)


# - lambda 식에 if else 절을 사용
# - 나이가 15세 미만이면 'Child', 그렇지 않다면 'Adult'로 구분하는 새로운 칼럼 'Child_Adult' 생성

# - lambda 식은 if else를 지원하는데, 주의할 점이 있다.
# - if 절의 경우 if식보다 반환 값을 먼저 기술해야 한다. 이는 lambda 식 ':'기호의 오른편에 반환 값이 있어야 하기 때문이다. 따라서 lambda x : if x<=15 'Child' else 'Adult'가 아니라 lambda x : 'Child' if x<=15 else 'Adult'이다.
# - else의 경우는 else식이 먼저 나오고 반환 값이 나중에 오면 된다.
# - 또한 if,else만 지원하고 else if는 지원하지 않는다. else if를 이용하기 위해서는 else 절을 ( )로 내포해 ( )내에서 다시 if else 적용해 사용한다.

# In[81]:


titanic_df['Age_Cat']=titanic_df['Age'].apply(lambda x : 'Child' if x<=15 else('Adult' if x<=60 else 'Elderly'))
titanic_df['Age_Cat'].value_counts()


# - 나이가 15세 이하이면 Child, 15~60세 사이는 Adult, 61세 이상은 Elderly로 분류하는 'Age_Cat'칼럼 생성

# - 첫번재 else절에서 ( )로 내포했는데, ( ) 안에서도 'Adult' if x<=60과 같이 if절의 경우 반환 값이 식보다 먼저 나왔음에 유의하기.
# - else if가 많이 나와야 하는 경우나 switch case문의 경우는 else를 계속 내포해서 쓰기에는 부담이 된다. 그래서 이 경우에는 아예 별도의 함수를 만드는게 더 나을 수 있다.

# In[82]:


def get_category(age):
    cat = ''
    if age<=5 : cat = 'Baby'
    elif age<=12 : cat = 'Child'
    elif age<=18 : cat = 'Teenager'
    elif age<=25 : cat = 'Student'
    elif age<=35 : cat = 'Young Adult'
    elif age<=60 : cat = 'Adult'
    else : cat = 'Elderly'
        
    return cat

# lambda식에 위에서 생성한 get_category() 함수를 반환값으로 지정.
# get_category(X)는 입력값으로 'Age' 칼럼 값을 받아서 해당하는 cat 반환
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))
titanic_df[['Age','Age_cat']].head()


# ## [05] 정리

# - 넘파이와 판다스는 머신러닝 학습을 위해 정복해야 할 매우 중요한 요소이다.
# - 실제로 ML 모델을 생성하고 예측을 수행하는데 있어서 ML알고리즘이 차지하는 비중보다 데이터를 전처리하고 적절한 피처를 가공/추출하는 부분이 훨씬 많은 비중을 차지하게 된다.

# In[ ]:




