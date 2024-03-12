# Function 1. 분류기의 성능을 반환하는 함수

from sklearn.metrics import (
	accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

def get_clf_scores(y_test, pred):
    acc = accuracy_score(y_test, pred)
    precis = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, pred)
    
    return acc, precis, recall, f1, auc



# Function 2. 모델의 성능을 출력하는 함수

from sklearn.metrics import confusion_matrix

def print_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    acc, pre, re, f1, auc = get_clf_scores(y_test, pred)
    
    print('《confusion matrix》')
    print(confusion)
    print('='*25)
    
    print('Accuracy: {0:.4f}, Precision: {1:.4f}'.format(acc, pre))
    print('Recall: {0:.4f}, F1: {1:.4f}, AUC: {2:.4f}'.format(re, f1, auc))
    

    
# Function 3. 모델과 데이터를 주면 성능을 출력하는 함수

def get_result(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    return get_clf_scores(y_test, pred)



# Function 4. 여러 가지 모델의 성능을 정리해서 pandas.DataFrame으로 반환하는 함수

def get_results_df(models, model_names, X_train, y_train, X_test, y_test):
    col_names = ['accuracy', 'precision', 'recall', 'F1', 'roc_auc']
    
    results = []
    for model in models:
        results.append(get_result(model, X_train, y_train, X_test, y_test))
    
    return pd.DataFrame(results, columns=col_names, index=model_names)



# Function 5. 모델별 ROC 커브를 그려주는 함수

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def draw_roc_curve(models, model_names, X_test, y_test):
    plt.figure(figsize=(10,10))
    
    for model in range(len(models)):
        pred = models[model].predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, pred)
        plt.plot(fpr, tpr, label=model_names[model])
    
    plt.plot([0,1], [0,1], 'k--', label='random guess')
    plt.title('ROC curve')
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()



# Function 6. Outlier에 해당하는 Index(ListType)을 반환하는 함수

import numpy as np

def get_outlier(df=None, column=None, weight=1.5):
    fraud = df[df['Class']==1][column]
    quantile_25 = np.percentile(fraud.values, 25)
    quantile_75 = np.percentile(fraud.values, 75)
    
    iqr = quantile_75 - quantile_25
    iqr_weight = iqr * weight
    lowest_val = quantile_25 - iqr_weight
    highest_val = quantile_75 + iqr_weight
    
    outlier_index = fraud[(fraud < lowest_val) | (fraud > highest_val)].index
    
    return outlier_index

