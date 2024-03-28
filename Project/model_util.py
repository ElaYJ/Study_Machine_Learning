# 베스트 모델
from sklearn.model_selection import GridSearchCV

def grid_search_cv_ressult(gs, cols=['params','rank_test_score','mean_train_score','mean_test_score']):
	score_df = pd.DataFrame(gs.cv_results_).sort_values(by='rank_test_score').reset_index(drop=True)
	return score_df[cols]


def get_best_model(X, y, clf=None, params=None, scoring='f1_macro', report=True, cols=None):
    skfold = StratifiedKFold(n_splits=5, shuffle=True)
    grid_cv = GridSearchCV(
		estimator=clf, param_grid=params, cv=skfold, scoring=scoring, 
  		return_train_score=True, n_jobs=-1
	)
    grid_cv.fit(X, y)
    print('Best Params:', grid_cv.best_params_)
    
    if report:
        if cols == None:
            cols=['params','rank_test_score','mean_train_score','mean_test_score']
        display(grid_search_cv_ressult(grid_cv, cols).head(7))
    
    return grid_cv.best_estimator_




# 성능 지표
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


def get_clf_scores(y_test, pred):
    acc = accuracy_score(y_test, pred)
    precis = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, pred)
    
    return acc, precis, recall, f1, auc


def print_clf_scores(y_test, pred_test):
	# print(' Accuracy :', accuracy_score(y_test, pred_test))
	# print(' Percision:', precision_score(y_test, pred_test))
	# print(' Recall:', recall_score(y_test, pred_test))
	# print(' F1 Score:', f1_score(y_test, pred_test))
	# print(' AUC Score:', roc_auc_score(y_test, pred_test))
	
	print('Accuracy: {0:.4f}, Precision: {1:.4f}'.format(
		accuracy_score(y_test, pred_test), precision_score(y_test, pred_test)
	))
	print('Recall: {0:.4f}, F1-score: {1:.4f}, AUC: {2:.4f}'.format(
		recall_score(y_test, pred_test), f1_score(y_test, pred_test), roc_auc_score(y_test, pred_test)
	))
 
 

def show_classification_report(y_test=None, pred_test=None, confusion=False):
    print(
		classification_report(y_test, pred_test, target_names=['Once Order(0)', 'Re Order(1)']),
        end='\n\n'
	)
    
    if confusion:
        cfs_val = confusion_matrix(y_test, pred_test)
        print('《confusion matrix》')
        print(cfs_val)
        print('='*25)
        get_clf_score(y_test, pred_test)



def show_roc_curve(model_clf, X_test, y_test):
	pred_proba = model_clf.predict_proba(X_test)

	fpr, tpr, thresholds = roc_curve(y_test, pred_proba[:,1])

	df = pd.DataFrame(
		data = [thresholds, fpr, tpr],
		index=['THs', 'FPR', 'TPR']
	)
	display(df)

	plt.figure(figsize=(7,6))
	plt.plot([0,1], [0,1], 'c', ls='dashed') #--> y=x 직선
	plt.plot(fpr, tpr, 'r') #--> x: fpr, y: tpr
	plt.grid()
	plt.show()


def show_models_roc(models, X_test, y_test, score=False):
	plt.figure(figsize=(10,8))
	plt.plot([0,1], [0,1], label='random_guess')

	for model_name, model in models.items():
		pred = model.predict_proba(X_test)[:, 1]

		fpr, tpr, thresholds = roc_curve(y_test, pred)
		if score:
			print("==", model_name)
			df = pd.DataFrame(
				data = [thresholds, fpr, tpr],
				index = ['THs', 'FPR', 'TPR']
			)
			display(df)

		plt.plot(fpr, tpr, label=model_name)

	plt.legend(fontsize=12)
	plt.grid()
	plt.show()


def show_feature_importance(clf, labels=None):
    import_val = clf.feature_importances_
    df = pd.DataFrame({'Features':labels,'importance':import_val})
    df = df.sort_values(by='importance', ascending=False)
    display(df)
    
    df.set_index('Features', inplace=True)
    
    sns.barplot(x=df['importance'], y=df.index, hue=df.index, legend=False, palette='Spectral')
    plt.xlabel("Importance", fontsize=12)
    plt.title("Features Coef. Importance")
    plt.show()


