# FILE ONE: grid search and classification model (accuracy, precision, recall, f-score; AUC and ROC curve in FILE TWO)
    # logistic regression
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

data = pd.read_csv('English (normalized).csv')
y = data['y']
x = data.drop(labels=['y'],axis=1)
cv = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=1)

for train_index,test_index in cv.split(x,y):
    X_train, X_test, y_train, y_test = x.iloc[train_index],x.iloc[test_index],y.iloc[train_index],y.iloc[test_index]

    clf = LogisticRegression(max_iter=1000)
    param_grid = {'C':[0.1,1,5,10,15,20,30,40,50,60,70,80,85,90,95,100], 'penalty': ['l1','l2'], 'solver': ['liblinear']}
    clf_grid = GridSearchCV(clf, param_grid, cv=5, verbose=0, n_jobs=-1)

    clf_grid.fit(X_train,y_train)
    y_pred = clf_grid.predict(X_test)

    print('best parameter:\n',clf_grid.best_params_)
    print('Best score is {}'.format(clf_grid.best_score_))
    print('accuracy:', metrics.accuracy_score(y_test, y_pred), 'precision:', metrics.precision_score(y_test, y_pred),
          'recall:', metrics.recall_score(y_test, y_pred), 'f-score:', metrics.accuracy_score(y_test, y_pred), 'cm:', metrics.confusion_matrix(y_test, y_pred))

    # linear SVM
from sklearn import svm

    clf = svm.SVC(kernel='linear')
    param_grid = {'C':[0.001, 0.10, 1, 10, 25, 50, 100, 200,300,500,1000]}
    clf_grid = GridSearchCV(clf,param_grid,cv=5,verbose=0,n_jobs=-1)

    clf_grid.fit(X_train,y_train)
    y_pred = clf_grid.predict(X_test)

    print('best parameter:\n',clf_grid.best_params_)
    print('Best score is {}'.format(clf_grid.best_score_))
    print('accuracy:', metrics.accuracy_score(y_test, y_pred), 'precision:', metrics.precision_score(y_test, y_pred),
          'recall:', metrics.recall_score(y_test, y_pred),'f-score:', metrics.accuracy_score(y_test, y_pred), 'cm:', metrics.confusion_matrix(y_test, y_pred))

        # linear SVM coefficients (weights) with the best parameters
    clf = svm.SVC(kernel='linear', C=10)
    clf.fit(x, y)
    for i in range(len(clf.coef_[0])):
        print(data.columns[i], clf.coef_[0][i])

    # decision tree
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('English (unnormalized).csv')
y = data['y']
x = data.drop(labels=['y'],axis=1)
cv = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=1)

for train_index,test_index in cv.split(x,y):
    X_train, X_test, y_train, y_test = x.iloc[train_index],x.iloc[test_index],y.iloc[train_index],y.iloc[test_index]

    clf = DecisionTreeClassifier(random_state=10)
    param_grid = {'max_depth': range(1,20,2),'min_samples_leaf': range(1,8),'max_features': [0.2, 0.4,0.6, 0.8]}
    clf_grid = GridSearchCV(clf,param_grid,cv=5,scoring='accuracy',n_jobs=-1)

    clf_grid.fit(X_train,y_train)
    y_pred = clf_grid.predict(X_test)

    print('best parameter:\n', clf_grid.best_params_)
    print('Best score is {}'.format(clf_grid.best_score_))
    print('accuracy:', metrics.accuracy_score(y_test, y_pred), 'precision:', metrics.precision_score(y_test, y_pred),'recall:', metrics.recall_score(y_test, y_pred), 'f-score:', metrics.accuracy_score(y_test, y_pred),
          'cm:',metrics.confusion_matrix(y_test, y_pred))

    # XGBoost
    # grid search and step-by-step tuning parameters: 1. fix learning rate and number of estimators for tuning tree-based parameters; 2. tune max_depth and min_child_weight; 3. tune gamma;
                                                  # 4. tune subsample and colsample_bytree; 5. tune Regularization Parameters; 6. tune learning rate and number of estimators;
                                                  # reference guide at https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    # tuning learning rate and number of estimators as an example
from xgboost.sklearn import XGBClassifier

    param_test5 = {'n_estimators': range(50,500,50),'learning_rate':[i/100.0 for i in range(1,10)]}
    gsearch5 = GridSearchCV(estimator=XGBClassifier(max_depth=5,min_child_weight=1, gamma=0.0, subsample=0.9, colsample_bytree=0.7,reg_alpha=0.1,
                                                    objective= 'binary:logistic',nthread=4, scale_pos_weight=1, seed=2),param_grid = param_test5, scoring='accuracy',n_jobs=-1,iid=False, cv=5)

    gsearch5.fit(X_train, y_train)
    y_pred = gsearch5.predict(X_test)

    print('best parameter:\n', gsearch5.best_params_)
    print('Best score is {}'.format(gsearch5.best_score_))
    print('accuracy:', metrics.accuracy_score(y_test, y_pred), 'precision:', metrics.precision_score(y_test, y_pred),
          'recall:', metrics.recall_score(y_test, y_pred), 'f-score:', metrics.accuracy_score(y_test, y_pred),
          'cm:', metrics.confusion_matrix(y_test, y_pred))

        # XGBoost feature importance score with the best parameters
    clf = XGBClassifier(max_depth=5, min_child_weight=1, gamma=0.0, subsample=0.9, colsample_bytree=0.7,
                        reg_alpha=0.1, n_estimators=350, learning_rate=0.05)
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    data = clf.feature_importances_
    print(data)

# FILE TWO: AUC and ROC curves for logistic regression, linear SVM, decision tree and XGBoost
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.style.use('ggplot')
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics

data1 = pd.read_csv('English (normalized).csv')
y1 = data1['y']
x1 = data1.drop(labels=['y'],axis=1)
cv1 = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=1)

for train_index,test_index in cv1.split(x1,y1):
    X_train1, X_test1, y_train1, y_test1 = x1.iloc[train_index],x1.iloc[test_index],y1.iloc[train_index],y1.iloc[test_index]

    lr = LogisticRegression(max_iter=1000, C=100, penalty='l2', solver='liblinear')
    lr.fit(X_train1, y_train1)
    y_pred1 = lr.predict(X_test1)

    print('roc_auc_score:', metrics.roc_auc_score(y_test1, y_pred1))

    y_predict_probabilities1 = lr.predict_proba(X_test1)[:, 1]
    fpr1, tpr1, _ = roc_curve(y_test1, y_predict_probabilities1)
    roc_auc1 = auc(fpr1, tpr1)

    svm = svm.SVC(kernel='linear',C=10, probability=True)
    svm.fit(X_train1, y_train1)
    y_pred2 = svm.predict(X_test1)

    print('roc_auc_score:', metrics.roc_auc_score(y_test1, y_pred2))

    y_predict_probabilities2 = svm.predict_proba(X_test1)[:,1]
    fpr2, tpr2, _ = roc_curve(y_test1, y_predict_probabilities2)
    roc_auc2 = auc(fpr2, tpr2)

data2 = pd.read_csv('English (unnormalized).csv')
y2 = data2['y']
x2 = data2.drop(labels=['y'], axis=1)
cv2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)

for train_index, test_index in cv2.split(x2, y2):
    X_train2, X_test2, y_train2, y_test2 = x2.iloc[train_index], x2.iloc[test_index], y2.iloc[train_index], y2.iloc[test_index]

    dt = DecisionTreeClassifier(random_state=10,max_depth=5, min_samples_leaf=1,max_features=0.8)
    dt.fit(X_train2, y_train2)
    y_pred3 = dt.predict(X_test2)

    print('roc_auc_score:', metrics.roc_auc_score(y_test2, y_pred3))

    y_predict_probabilities3 = dt.predict_proba(X_test2)[:, 1]
    fpr3, tpr3, _ = roc_curve(y_test2, y_predict_probabilities3)
    roc_auc3 = auc(fpr3, tpr3)

    xgb = XGBClassifier(max_depth=5,learning_rate =0.05, min_child_weight=1, n_estimators=350,gamma=0, subsample=0.9, colsample_bytree=0.7,reg_alpha=0.1,
                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=2,n_jobs=-1,probability=True)
    xgb.fit(X_train2, y_train2)
    y_pred4 = xgb.predict(X_test2)

    print('roc_auc_score:', metrics.roc_auc_score(y_test2, y_pred4))

    y_predict_probabilities4 = xgb.predict_proba(X_test2)[:, 1]
    fpr4, tpr4, _ = roc_curve(y_test2, y_predict_probabilities4)
    roc_auc4 = auc(fpr4, tpr4)

    plt.figure()
    plt.plot(fpr1, tpr1, linestyle='--', color='blue', lw=2, label='Logistic Regression (area = %0.2f)' % roc_auc1)
    plt.plot(fpr2, tpr2, linestyle='-.', color='orange', lw=2, label='SVM (area = %0.2f)' % roc_auc2)
    plt.plot(fpr3, tpr3, linestyle=':', color='green', lw=2, label='Decision Tree (area = %0.2f)' % roc_auc3)
    plt.plot(fpr4, tpr4, linestyle='-', color='red', lw=2, label='XGBoost (area = %0.2f)' % roc_auc4)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('English-speaking education systems')
    plt.legend(loc='lower right')
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Times New Roman']
    plt.show()

# FILE THREE: linear SVM and XGBoost classification models of top 10, 20, 30, 40, 50, 60, full-set features
    # linear SVM
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

data = pd.read_csv('English (normalized).csv')
y = data['y']
x = data.drop(labels=['y'],
                      #'ASBR07E', 'ASBR07D', 'ASBR07F', 'ASBR07B', 'ASBR07A', 'ASBR06E', 'ASBR01H', 'ASBR06F', 'ASBR06H', 'ATBR09AB'(top 10)
                      #'ASBR01D', 'ATBR11D', 'ASBR06G', 'ASBR06B', 'ASBR06C', 'ATBR12G', 'ATBR19B', 'ATBR10D', 'ATBR13B', 'ATBR12C',(top 20)
                      #'ASBR07C', 'ATBR08E', 'ASBR01A', 'ATBR12F', 'ATBR10G', 'ASBR01E', 'ATBR10B', 'ATBR08B', 'ASBR01C', 'ATBR11H',(top 30)
                      #'ATBR11C', 'ATBR22A', 'ATBR19A', 'ATBR13D', 'ASBR06A', 'ATBR13A', 'ATBR11F', 'ATBR11A','ATBR17', 'ASBR01B',(top 40)
                      #'ASBR06D', 'ATBR19C', 'ATBR11G', 'ATBR11B','ASBR01G', 'ATBR22C', 'ATBR08D', 'ATBR09BC', 'ATBR12E', 'ATBR11E',(top 50)
                      #'ASBR01I', 'ATBR10A', 'ATBR18', 'ATBR12D', 'ATBR09BB', 'ATBR10C', 'ATBR09BA', 'ATBR13C', 'ATBR10E', 'ATBR09AC',(top 60)
                      #'ATBR12A', 'ATBR09AA', 'ATBR22B', 'ATBR08C', 'ATBR08A', 'ATBR10F', 'ATBR12H', 'ATBR12I', 'ATBR12B', 'ASBR01F', 'ATBR11I'],(full-set)
                      axis=1)
cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)

for train_index, test_index in cv.split(x, y):
    X_train, X_test, y_train, y_test = x.iloc[train_index], x.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

    svm = svm.SVC(kernel='linear', C=10)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    print('accuracy:', metrics.accuracy_score(y_test, y_pred), 'precision:', metrics.precision_score(y_test, y_pred),
          'recall:', metrics.recall_score(y_test, y_pred), 'cm:', metrics.confusion_matrix(y_test, y_pred))

    # XGBoost
from xgboost.sklearn import XGBClassifier

data = pd.read_csv( 'English (unnormalized).csv')
y = data['y']
x = data.drop(labels=['y',
                      #'ASBR07E', 'ASBR07D', 'ASBR07F', 'ASBR07B', 'ASBR07A', 'ASBR06E', 'ASBR07C', 'ASBR06C', 'ASBR06D', 'ASBR06F',(top 10)
                      #'ASBR06G', 'ASBR06H', 'ASBR01H', 'ATBR10D', 'ASBR01B', 'ASBR06A', 'ASBR06B', 'ASBR01A', 'ATBR09AB', 'ATBR12G',(top 20)
                      #'ATBR08C', 'ATBR19B', 'ATBR12F', 'ATBR12C', 'ASBR01D', 'ATBR11B', 'ATBR10F', 'ATBR13A', 'ATBR09AA', 'ATBR09AC',(top 30)
                      #'ATBR19A', 'ATBR09BC', 'ATBR18', 'ATBR09BB', 'ATBR12A', 'ATBR12D', 'ATBR08D', 'ATBR12B', 'ATBR13B', 'ATBR08B',(top 40)
                      #'ASBR01F', 'ATBR22B', 'ATBR13D', 'ASBR01C', 'ATBR10G', 'ATBR08E', 'ATBR17', 'ATBR12E', 'ATBR11H', 'ATBR10B',(top 50)
                      #'ASBR01E', 'ATBR08A', 'ASBR01G', 'ATBR22C', 'ATBR10E', 'ATBR11G', 'ATBR11F', 'ATBR11A', 'ASBR01I', 'ATBR22A',(top 60)
                      #'ATBR13C', 'ATBR09BA', 'ATBR11C', 'ATBR19C', 'ATBR10A', 'ATBR10C', 'ATBR12I', 'ATBR11D', 'ATBR12H', 'ATBR11I', 'ATBR11E'],(full-set)
                      axis=1)
cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)

for train_index, test_index in cv.split(x, y):
    X_train, X_test, y_train, y_test = x.iloc[train_index], x.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

    xgb = XGBClassifier(max_depth=5,learning_rate =0.05, n_estimators=350,gamma=0.0, subsample=0.9, colsample_bytree=0.7,reg_alpha=0.1, n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    print('accuracy:',metrics.accuracy_score(y_test,y_pred),'precision:',metrics.precision_score(y_test,y_pred),'recall:',metrics.recall_score(y_test,y_pred),'cm:',metrics.confusion_matrix(y_test,y_pred))