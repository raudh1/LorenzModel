import numpy as np
from sklearn.neighbors import KDTree
rng = np.random.RandomState(0)
X = [[-1,2],[1,2],[-2,1],[-1.5,0],[0,0],[1,1],[1.5,2]]  # 10 points in 3 dimensions
tree = KDTree(X, leaf_size=2)
dist, ind = tree.query(X[:1], k=3)
print(dis)
