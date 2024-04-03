import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# PCA 결과를 반환하는 함수
def get_pca_data(ss_data, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(ss_data)
    
    return pca.transform(ss_data), pca


# PCA 결과를 Pandas.DataFrame으로 정리하는 함수
# def get_pd_from_pca(pca_data, cols=['pca_component_1','pca_component_2']):
def get_pd_from_pca(pca_data, cols):
    if isinstance(cols, list):
        return pd.DataFrame(pca_data, columns=cols)
    elif isinstance(cols, int):
        cols = ['pca_' + str(n+1) for n in range(cols)]
        return pd.DataFrame(pca_data, columns=cols)


# PCA로 얻은 주성분의 전체 설명력을 출력하는 함수
def print_variances_ratio(pca, with_sum=True):
    print('variance_ratio:', pca.explained_variance_ratio_)
    if with_sum:
        print('Sum of variance_ratio:', sum(pca.explained_variance_ratio_))


# Random Forest Model을 K=5로 교차검정한 후 Accuracy Score를 반환
def rf_scores(X, y, cv=5):
    rf = RandomForestClassifier(n_estimators=100, random_state=13)
    scores_rf = cross_val_score(rf, X, y, scoring='accuracy', cv=cv)
    
    print('Score :', np.mean(scores_rf))



#=============================================================================    



import matplotlib.pyplot as plt

# v0에서 부터 v1까지 화살표를 그리는 함수
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrow_props = dict(
		arrowstyle = '->',
		linewidth = 2,
		color = 'black',
		shrinkA = 0, shrinkB = 0
	)
    ax.annotate("", v1, v0, arrowprops=arrow_props)