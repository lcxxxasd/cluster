import time
from typing import List
import numpy
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


class ClusterAnalyzer:
    def __init__(self):
        pass

    def load_embedding_model(self, embedding_model = "BAAI/bge-small-zh-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        self.model.eval()

    def get_query_embeddings(self, query: List[str]):
        """
        获取query的embedding

        """

        start_time = time.time()

        encoded_input = self.tokenizer(query, padding=True, truncation=True, return_tensors='pt')

        encoded_input = self.tokenizer(query, padding=True, truncation=True, return_tensors='pt')

        # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
        # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        end_time = time.time()

        print("embedding耗时: {:.4f}秒".format(end_time - start_time))
        return embeddings.numpy()

    def dist_eclud(self, vecA,vecB):
        """
        计算两个向量的欧式距离
        """
        return np.sqrt(np.sum(np.power(vecA-vecB,2)))

    def rand_cent(self, data, k):
        """
        随机生成k个点作为质心，其中质心均在整个数据数据的边界之内
        """
        n = data.shape[1] # 获取数据的维度
        centroids = np.mat(np.zeros((k,n)))
        for j in range(n):
            minJ = np.min(data[:,j])
            rangeJ = np.float64(np.max(data[:,j])-minJ)
            centroids[:,j] = minJ+rangeJ*np.random.rand(k,1)
        return centroids

    def k_means(self, data: numpy.ndarray, k, max_iter = 30):
        """
        k-Means聚类算法,返回最终的k各质心和点的分配结果
        :param data: query embeddings，通过queryEmbedding()获取\
        :param k: 聚类的簇数
        :param max_iter: 最大迭代次数
        :return: 最终的k各质心和点的分配结果
        """
        m = data.shape[0]  #获取样本数量
        # 构建一个簇分配结果矩阵，共两列，第一列为样本所属的簇类值，第二列为样本到簇质心的误差
        clusterAssment = np.mat(np.zeros((m,2)))
        # 1. 初始化k个质心
        centroids = self.rand_cent(data,k)
        clusterChanged = True
        iter = 0
        while clusterChanged and iter <= max_iter:
            clusterChanged = False
            for i in range(m):
                minDist = np.inf
                minIndex = -1
                # 2. 找出最近的质心
                for j in range(k):
                    distJI = self.dist_eclud(centroids[j,:],data[i,:])
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j
                # 3. 更新每一行样本所属的簇
                if clusterAssment[i,0] != minIndex:
                    clusterChanged = True
                clusterAssment[i,:]=minIndex,minDist**2
            # print(centroids) # 打印质心
            # 4. 更新质心
            for cent in range(k):
                ptsClust = data[np.nonzero(clusterAssment[:,0].A==cent)[0]] # 获取给定簇的所有点
                if ptsClust.shape[0] != 0:
                    centroids[cent,:] = np.mean(ptsClust,axis=0) # 沿矩阵列的方向求均值
            iter += 1
        return centroids,clusterAssment

    def bi_kmeans(self, data : numpy.ndarray, limit_eval = 2):
        """
        二分K-Means聚类算法
        :param data: query embeddings，通过queryEmbedding()获取
        :param limit_eval: 判断对二分kmeans是否停止的阈值，当一个簇的二分增幅小于limit_eval时不再对该簇进行划分，即这个簇已经达到最终状态，不可再分。
        :return: 最终的k各质心和点的分配结果
        """

        start_time = time.time()
        m = data.shape[0]
        clusterAssment = np.mat(np.zeros((m,2)))
        # 创建初始簇质心
        centroid0 = np.mean(data,axis=0).tolist()[0]
        centList = [centroid0]
        # 计算每个点到质心的误差值
        for j in range(m):
            clusterAssment[j,1] = self.dist_eclud(np.mat(centroid0),data[j,:])**2
        while (True):
            lowestSSE = np.inf
            sseAll = sum(clusterAssment[:, 1])  # 获取所有数据集的sse
            for i in range(len(centList)):
                # 获取当前簇的所有数据
                ptsInCurrCluster = data[np.nonzero(clusterAssment[:,0].A == i)[0],:]
                # 对该簇的数据进行K-Means聚类
                centroidMat, splitClustAss = self.k_means(ptsInCurrCluster,2)
                sseSplit = sum(splitClustAss[:,1]) # 该簇聚类后的sse
                sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:,0].A != i)[0],1]) # 获取剩余收据集的sse

                if (sseSplit + sseNotSplit) < lowestSSE:
                    bestCentToSplit = i
                    bestNewCents = centroidMat
                    bestClustAss = splitClustAss.copy()
                    lowestSSE = sseSplit + sseNotSplit
            if sseAll - lowestSSE < limit_eval:
                break
            # 将簇编号0,1更新为划分簇和新加入簇的编号
            bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0]= len(centList)
            bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0]= bestCentToSplit

            print("the bestCentToSplit is: ",bestCentToSplit)
            print("the len of bestClustAss is: ",len(bestClustAss))
            # 增加质心
            centList[bestCentToSplit] = bestNewCents[0,:]
            centList.append(bestNewCents[1,:])

            # 更新簇的分配结果
            clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
        end_time = time.time()
        print("聚类耗时: {:.4f}秒".format(end_time - start_time))
        return centList, clusterAssment



