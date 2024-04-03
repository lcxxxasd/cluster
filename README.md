

使用前需要指定embeding model，默认为bge-small-zh-v1.5（可更改huggingface支持任意的embeding model）

### 样例代码
```python
import pandas as pd
from cluster_kmeas import ClusterAnalyzer

sentences = []
file = "./data/nio_query_fallback_pv_5.csv"

df = pd.read_csv(file)


for idx, row in df.iterrows():
    index = idx
    query = row['query']
    sentences.append(query)


clusterAnalyzer = ClusterAnalyzer()

clusterAnalyzer.load_embedding_model("BAAI/bge-small-zh-v1.5")

query_embeddings = clusterAnalyzer.get_query_embeddings(sentences)

# myCentroids为簇质心. clustAssing为簇分配结果，shape为[len(sentence), 2], 第一列为簇编号，第二列为到所属簇的距离
myCentroids, clustAssing = clusterAnalyzer.bi_kmeans(query_embeddings, limit_eval = 2)  
```