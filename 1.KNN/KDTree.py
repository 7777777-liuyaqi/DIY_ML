# %load KDTree.py
#!/usr/bin/env python

import numpy as np
import os
import time
from collections import Counter
from sklearn import neighbors, datasets



class Preprocessing(object):
    """
    预处理过程
    导入训练数据：Preprocessing.file2matrix(filename)
    归一化数据集:Preprocessing.autoNorm(dataSet)
    """
    def file2matrix(self,filename):
        """
        导入训练数据
        :param filename: 数据文件路径
        :return: 数据矩阵returnMat和对应的类别label
        """
        mat=[]
        label=[]
        with open(filename, 'r') as f:
            for line in f:
                lst=line.strip().split("\t")
                data=map(eval, lst[:-1])
                mat.append(list(data))
                label.append(int(lst[-1]))
        returnMat=np.matrix(mat)
        return returnMat,label

    def autoNorm(self,dataSet):
        """
        Desc：
            归一化特征值，消除属性之间量级不同导致的影响
        Args：
            dataSet -- 需要进行归一化处理的数据集
        Returns：
            normDataSet -- 归一化处理后得到的数据集
            ranges -- 归一化处理的范围
            minVals -- 最小值

        归一化公式：
            Y = (X-Xmin)/(Xmax-Xmin)
            其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
        """
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        ranges = maxVals - minVals
        norm_dataset = (dataSet - minVals) / ranges
        return norm_dataset, ranges, minVals



class Node(object):
    def __init__(self):
        self.father = None
        self.left = None
        self.right = None
        self.feature = None  #记录当前的分割特征
        self.split = None    #记录当前的分割点
     
    def __str__(self):
        return "feature: %s, split: %s" % (str(self.feature), str(self.split))
    
    @property
    def brother(self):
        """Find the node's brother.
        Returns:node -- Brother node.
        """
        if self.father :
            if self.father.left is self:
                ret=self.father.right
            else:
                ret = self.father.left
        else:
            ret = None
        return ret    


class Distance(object):
    def get_euclidean_distance(self,arr1, arr2) -> float:
        """
        Calculate the Euclidean distance of two vectors.
        Arguments:
            arr1 {ndarray}
            arr2 {ndarray}
        Returns:
            float
        """
        return ((arr1 - arr2) ** 2).sum() ** 0.5

    def get_eu_dist(self,arr1, arr2) -> float:
        """
        Calculate the Euclidean distance of two vectors.
        Arguments:
            arr1 {list} -- 1d list object with int or float
            arr2 {list} -- 1d list object with int or float
        Returns:
            float -- Euclidean distance
        """

        return sum((x1 - x2) ** 2 for x1, x2 in zip(arr1, arr2)) ** 0.5


    def get_cosine_distance(self,arr1, arr2):
        """Calculate the cosine distance of two vectors.
        Arguments:
            arr1 {list} -- 1d list object with int or float
            arr2 {list} -- 1d list object with int or float
        Returns:
            float -- cosine distance
        """
        numerator = sum(x1 * x2 for x1, x2 in zip(arr1, arr2))
        denominator = (sum(x1 ** 2 for x1 in arr1) *
                       sum(x2 ** 2 for x2 in arr2)) ** 0.5
        return numerator / denominator


class KDTree(object):
    def __init__(self):
        """KD Tree class to improve search efficiency in KNN.
        Attributes:
            root: the root node of KDTree.
        """
        self.root = Node()
     
    def __str__(self):
        """Show the relationship of each node in the KD Tree.
        Returns:
            str -- KDTree Nodes information.
        """
        i=0
        ret=[]
        que=[(self.root,-1)]
        while que:
            nd,idx=que.pop(0)
            ret.append("%d -> %d: %s" % (idx, i, str(nd)))
            if nd.left:
                que.append((nd.left,i))
            if nd.right:
                que.append((nd.right, i))
            i+=1
        return "\n".join(ret)
    
    def _get_median_idx(self, X, idxs, feature):
        """Calculate the median of a column of data.
        Arguments:
            X {matrix} 
            idxs {list} -- 1D list with int.
            feature {int} -- Feature number.
        Returns:
            list -- The row index corresponding to the median of this column.
        """
        n=len(idxs)
        k=n//2
        col=list(map(lambda i: (i, X[i,feature]),idxs))
        median_idx=sorted(col,key=lambda x: x[1])[k][0]
        return median_idx
    
    def _get_variance(self, X, idxs, feature):
        """Calculate the variance of a column of data.
        Arguments:
            X {matrix} 
            idxs {list} -- 1D list with int.
            feature {int} -- Feature number.
        Returns:
            float -- variance
        """
        col=list(map(lambda i:X[i,feature],idxs))
#         col_sqr=list(map(lambda i:i**2,col))
#         #         D(X) = E{[X-E(X)]^2} = E(X^2)-[E(X)]^2
#         return np.mean(col_sqr)-np.mean(col)**2
        return np.var(col)

    def _choose_feature(self, X, idxs):
        """Choose the feature which has maximum variance.
        Arguments:
            X {matrix} 
            idxs {list} -- 1D list with int.
        Returns:
            feature number {int}
        """
        feature=range(X.shape[1])
        variances =list(map(lambda i:(i,self._get_variance(X, idxs, i)),feature))
        return max(variances,key=lambda x: x[1])[0]
    
    def _split_feature(self, X, idxs, feature, median_idx):
        """Split indexes into two arrays according to split point.
        Arguments:
            median_idx {float} -- Median index of the feature.
        Returns:
            list -- [left idx, right idx]
        """
        idxs_split = [[], []]
        split_val = X[median_idx,feature]
        for idx in idxs:
            # Keep the split point in current node.
            if idx != median_idx:
                if X[idx,feature]<split_val:
                    idxs_split[0].append(idx)
                else:
                    idxs_split[1].append(idx)
        return idxs_split
    
    def build_tree(self, X, y):
        """Build a KD Tree. The data should be scaled so as to calculate variances.
        Arguments:
            X {matrix} 
            y {list} -- 1d list object with int or float.
            
        """
        nd = self.root #1. 建立根节点；
        idxs=list(range(X.shape[0]))
        que = [(nd, idxs)]
        while que:
            nd, idxs = que.pop(0)
            # Stop split if there is only one element in this node
            if len(idxs) ==1:
                nd.split = (tuple(X[idxs[0]].tolist()[0]), y[idxs[0]])  
                continue
            
            feature= self._choose_feature(X,idxs) #2. 选取方差最大的特征作为分割特征；
            median_idx=self._get_median_idx(X, idxs, feature)  #3. 选择该特征的中位数作为分割点；
            idxs_split = self._split_feature(X, idxs, feature, median_idx)   #4. 将数据集中该特征小于中位数的传递给根节点的左儿子，大于中位数的传递给根节点的右儿子；         
            nd.feature = feature
            nd.split = (tuple(X[median_idx].tolist()[0]), y[median_idx])
            idxs_left, idxs_right = idxs_split[0],idxs_split[1]
            if idxs_left:
                nd.left = Node()
                nd.left.father = nd
                que.append((nd.left,idxs_left))
            if idxs_right :
                nd.right = Node()
                nd.right.father = nd
                que.append((nd.right, idxs_right))  

    def _search(self, Xi, nd):
        """Search Xi from the KDTree until Xi is at an leafnode.
        Arguments:
            Xi {array} -- 1d list with int or float.
        Returns:
            node -- Leafnode.
        """
        while nd.left or nd.right:
            if not nd.left:
                nd=nd.right
            elif not nd.right:
                nd = nd.left
            else:
                if Xi[nd.feature] < nd.split[0][nd.feature]:
                    nd = nd.left
                else:
                    nd = nd.right
        return nd       
    
    def _get_eu_dist(self, Xi, nd):
        """Calculate euclidean distance between Xi and node.
        Arguments:
            Xi {array} -- 1d list with int or float.
            nd {node}
        Returns:
            float -- Euclidean distance.
        """
        X0=np.array(list(nd.split[0]))
        return np.sum((X0-Xi)**2)**0.5
    
    def _get_hyper_plane_dist(self, Xi, nd):
        """Calculate euclidean distance between Xi and hyper plane.
        Arguments:
            Xi {array} -- 1d list with int or float.
            nd {node}
        Returns:
            float -- Euclidean distance.
        """
        feature=nd.feature
        X0=np.array(list(nd.split[0]))
        return abs(Xi[feature] - X0[feature])
    
    def nearest_neighbour_search(self, Xi):
        """
        Nearest neighbour search and backtracking.
        Arguments:
            Xi {array} -- The normalized data.
        Returns:
            node -- The nearest node to Xi.
        搜索过程：
        1. 从根节点开始，根据目标在分割特征中是否小于或大于当前节点，向左或向右移动。
        2. 一旦算法到达叶节点，它就将节点点保存为“当前最佳”。    
        3. 回溯，即从叶节点返回到根节点
        4.
        5. 如果目标与当前节点的父节点所在的将数据集分割为两份的超平面相交，说明当前节点的兄弟节点所在的子树有可能包含更近的点。因此需要对这个兄弟节点递归执行1-4步。
        """
        best_dist=float("inf")
        nd_best = self._search(Xi, self.root) 
        que=[(self.root,nd_best)]
        while que:
            nd_root,cur_node=que.pop(0)
            # Calculate distance between Xi and root node
            dist = self._get_eu_dist(Xi, nd_root)
            # Update best node and distance.
            if dist < best_dist:
                best_dist, nd_best = dist, nd_root
            while cur_node is not nd_root:  
                cur_dist=self._get_eu_dist(Xi,cur_node)
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    nd_best = cur_node
                if cur_node.brother and  (
                    best_dist > self._get_hyper_plane_dist(Xi,cur_node.father))  :
                    _nd_best = self._search(Xi,cur_node.brother)
                    que.append((cur_node.brother,_nd_best))
                cur_node=cur_node.father
            return nd_best

    def k_nearest_neighbor_search(self,Xi,k):
        """
        Nearest neighbour search and backtracking.
        Arguments:
            k {int} --The number of nearest neighbor.
            Xi {array} -- The normalized data.
        Returns:
            pointlist -- The list of k nearest neighbor points.
        """
        pointlist=[]  #存储排序后的k近邻点和对应距离
        knears={}     #用于存储k近邻的点以及与目标点的距离
        nd_best = self._search(Xi, self.root) 
        que=[(self.root,nd_best)]
        while que:
            nd_root,cur_node=que.pop(0)
            dist = self._get_eu_dist(Xi, nd_root)
            if len(knears)<k:
                knears.setdefault(cur_node.split,dist)
                pointlist=sorted(knears.items(),key=lambda item: item[1],reverse=True)
            elif dist<pointlist[0][1]:
                knears.pop(pointlist[0][0])
                knears.setdefault(cur_node.split,dist)
                pointlist=sorted(knears.items(),key=lambda item: item[1],reverse=True)
            while cur_node is not nd_root:  
                cur_dist=self._get_eu_dist(Xi,cur_node)
                if len(knears)<k:
                    knears.setdefault(cur_node.split,cur_dist)
                    pointlist=sorted(knears.items(),key=lambda item: item[1],reverse=True)
                elif cur_dist<pointlist[0][1]:
                    knears.pop(pointlist[0][0])
                    knears.setdefault(cur_node.split,cur_dist)
                    pointlist=sorted(knears.items(),key=lambda item: item[1],reverse=True)
                if cur_node.brother and  (
                    self._get_hyper_plane_dist(Xi,cur_node.father) < pointlist[0][1])  :
                    _nd_best = self._search(Xi,cur_node.brother)
                    que.append((cur_node.brother,_nd_best))
                cur_node=cur_node.father 
        return pointlist


class KNNBase(object):
    def line_classify(self,inX, dataSet, labels, k):
        """
        Desc:
            kNN 的分类函数
        Args:
            inX -- 用于分类的输入向量/测试数据
            dataSet -- 训练数据集的 features
            labels -- 训练数据集的 labels
            k -- 选择最近邻的数目
        Returns:
            输入向量的预测分类 labels

        程序使用欧式距离公式.
        """
        x=inX-dataSet
        x_norm=np.linalg.norm(x, ord=2, axis=1)
        k_labels = [labels[index] for index in x_norm.argsort()[0 : k]]
        label = Counter(k_labels).most_common(1)[0][0]
        return label



def main(k):
    pro = Preprocessing()
    data_mat , labels = pro.file2matrix("datingTestSet2.txt")
    ratio = 0.1  
    norm_mat, ranges, min_vals = pro.autoNorm(data_mat)
    m = norm_mat.shape[0]
    num_test = int(m * ratio) 
#     print('num_test = ', num_test)
    error_count = 0 
    X = norm_mat[num_test : m]
    Y = labels[num_test : m]
    x = norm_mat[:num_test]
    
    classify0 = KNNBase()
    classify1 = KDTree()
    # result0=[]
    # result1=[]
    
    start = time.time()
    for i in range(num_test):
        result=classify0.line_classify(x[i], X, Y, k)
        error_count +=( result != labels[i])
#         print("the classifier came back with: %d, the real answer is: %d" % (result, labels[i]))
    run_time_1 = time.time() - start    
    print("the line_classify total error rate is: %.2f" % (error_count / num_test))
    print("Exhausted search %.4f s" % run_time_1)
    
    error_count = 0
    classify1.build_tree(X, Y)
    start = time.time()
   
    for i in range(num_test):
        pointlist = classify1.k_nearest_neighbor_search(norm_mat[i].getA()[0],k)
        result=Counter(list(map(lambda x: x[0][1],pointlist))).most_common()[0][0]
#         print(pointlist,result)
        error_count += result != labels[i]
      
    run_time_2 = time.time() - start     
    print("the kd tree  total error rate is: %.2f" % (error_count / num_test))
    print("KD Tree Search %.4f s"% run_time_2)
    
    clf = neighbors.KNeighborsClassifier(n_neighbors=k,algorithm="kd_tree", weights="distance")
    clf.fit(X, Y)
    error_count = 0
    start = time.time()
    for i in range(num_test):
        result = clf.predict(x[i])
        error_count += result != labels[i]
    run_time_3 = time.time() - start     
    print("the sklearn kd_tree  total error rate is: %.2f" % (error_count / num_test))
    print("the sklearn kd_tree Search %.4f s"% run_time_3)

if __name__ == "__mian__":
    main(k=3)


