# **3. Emsemble methods**
常见的集成学习框架有：Bagging和Boosting。两种集成学习框架在基学习器的产生和综合结果的方式上会有些区别，我们先做些简单的介绍。
![ML Types](doc/doc_ensembleMethods/ensemble1.jpeg)
## **1. Bagging**
![ML Types](doc/doc_ensembleMethods/ensemble2.jpeg)
**Bagging** 全称叫 Bootstrap aggregating, 每个基学习器都会对训练集进行有放回抽样得到子训练集，比较著名的采样法为 0.632 自助法。每个基学习器基于不同子训练集进行训练，并综合所有基学习器的预测值得到最终的预测结果。Bagging 常用的综合方法是投票法，票数最多的类别为预测类别。
## **1. Boosting**
![ML Types](doc/doc_ensembleMethods/ensemble3.jpeg)
**Boosting** Boosting 训练过程为阶梯状，基模型的训练是有顺序的，每个基模型都会在前一个基模型学习的基础上进行学习，最终综合所有基模型的预测值产生最终的预测结果，用的比较多的综合方式为加权法。


## **References：**
<br/>[1] 知乎[@阿泽](https://www.zhihu.com/search?type=content&q=random%20forest)
<br/>[2]towards data science[@Fernando López](https://towardsdatascience.com/ensemble-learning-bagging-boosting-3098079e5422)





