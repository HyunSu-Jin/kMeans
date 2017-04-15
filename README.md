# kMeans
implemented by jupyter notebook

## Clustering
Clustering은 머신러닝의 unsupervised Learning과 동일한 의미로써 쓰인다. 즉, dataset에 속한 각 data들에 class label이 주어 지지 않았을 때 각 데이터간의 유사도(similarity)를 측정하여 유사도가 높은 데이터들끼리 군집화(clustering)하는 기법이다.

## Partitioning method
각 cluster들이 계층적 구조를 가지지 않고 모두 같은 level을 갖는 경우를 의미한다.
k-means algorithm은 partitioning method이며 그중에서도 각 cluster가 mutually exclusive하게 군집화하는 클러스터링 기법이다.

## 유사도(Similarity)
각 데이터간의 유사도를 평가하는 지표는 데이터의 feature의 특성에 의하여 결정된다. 위 예제에서는 데이터형식에서는 유사도 측정지표로써 Euclidean distance를 사용하도록 한다.
그 외의 측정지표로써 consine,Jaccard 등이 있다.

## Best Cluster
좋은 Partition 분할은 다음과 같이 정의할 수 있다.
1. 하나의 Partition에 속한 데이터끼리 유사성이 높은지 (closed 한지)
2. 각각의 다른 Partition끼리 얼마나 다른지 (far apart 한지)
모든 가능한 경우의 수 중에서 위 두가지 지표가 최상인 경우를 Global optimum 이다.

## Greedy Local Best Search (k-Means)
Partition을 바탕으로 클러스터링할 때 전역 최적을 찾아내기란 가능한 모든 분할 방법을 전부 고려해야 할 것이기 떄문에 보통 그 계산량을 감당할 수 없다. 따라서 대개는 Heuristic 기법을 활용하는 Greedy Local Best Search를 사용한다.
이에대한 예가 바로 k-Means algorithm이다.
Local search는 매 trial이 global optimum을 보장하지 않는다.(local optimum 일 수 있다.)

## k-Means
k-Means 클러스터링의 알고리즘은 다음과 같다.

(1) 임의로 k개의 data를 선택해서 초기 클러스터의 centroid를 만든다.
(2) 모든 data를 각 클러스터의 centroid와 유사도를 Euclidean distance로 비교해서 가장 유사도가 높은 클러스터로 배정한다.
(3) 각기 구해낸 클러스터에서 새로운 centroid를 계산한다.
(4) 클러스터의 data 구성이 변화하지 않을 때 까지 (2),(3)을 반복한다.

## k-Means의 한계
1. Scalability
k-Means는 데이터의 규모가 커졌을때 성능이 사용이 불가능할 정도로 저하된다.
알고리즘 단계중 DB에 존재하는 모든 튜플과 모든 centroid간의 거리를 계산하여야 하므로 이에 대한 cost 비용이 크다. 또한, data의 feature의 수가 많아질수록 Euclidean distance의 계산비용도 증가한다.

2. Parametic Clustering
실제 App에서는 군집(cluster)의 개수를 무엇으로 할지 정할 수 없는 상황도 있는데 k-Means는 반드시 정해진 k를 입력받아야 한다는 문제가 있다.

3. Robustness
k-Means는 centroid로써 vector간의 평균을 취하므로 통계의 평균의 속성이 Noise(=Outlier)에 취약한 점 때문에 k-Means 또한 Noise에 취약하다.

## 주요 코드
<pre><code>
class KMeans:
    def __init__(self,k):
        self.k = k
        self.means = None # 각 cluster의 centroid를 나타내는 값. k개의 centroid를 갖는 list
    
    def classify(self,input):
        distances = [euclidean_distance(input,centroid)for centroid in self.means]
        distances = np.array(distances)
        return np.argmin(distances)
    
    def train(self,inputs):
        self.means = random.sample(inputs,self.k) # 초기 중심점을 dataset에서 부터 임의로 지정한다. training의 의미는 이 controid값을 배정하는 것
        assignments = None
        
        while True:
            # 소속되는 군집을 다시 찾기
            new_assignments = []
            for group in map(self.classify,inputs):
                new_assignments.append(group)
            #print(new_assignments)
            # 만약 clustering의 결괏값이 변하지 않았다면,
            if assignments == new_assignments:
                return
            else:
                assignments = new_assignments
                #print('assignments :',assignments)
                # conetorid 재조정
                for group_idx in range(self.k):
                    vectors =[]
                    for data,label in zip(inputs,assignments):
                        #print('label : ',label,'\ndata:',data)
                        if label == group_idx:
                             vectors.append(data)
                    vectors = np.array(vectors)
                    new_centroid = np.sum(vectors,axis=0) / vectors.shape[0]
                    self.means[group_idx] = new_centroid
</code></pre>

## 예제
![image3](/img/image3.png)

## 적절한 k 선택
각 cluster의 centroid와 해당 cluster에 속한 각 data간의 거리의 총합을 Error라 하고 k의 값을 점차 증가시키면 다음과 같은 graph를 얻어 적절한 k값을 선택할 수 있다.

![image2](/img/image2.png)
