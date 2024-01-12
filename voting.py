# -*- coding: utf-8 -*-
"""

"""

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
import pandas as pd
from functools import reduce
import utils.tools as utils
import scipy.io as sio
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import scale,StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
#from L1_Matine import elasticNet
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold,LeaveOneOut
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier

#def elasticNet(data,label,alpha =np.array([0.001, 0.002, 0.003,0.004, 0.005, 0.006, 0.007, 0.008,0.009, 0.01])):
    #enetCV = ElasticNetCV(alphas=alpha,l1_ratio=0.1).fit(data,label)
    #enet=ElasticNet(alpha=enetCV.alpha_, l1_ratio=0.1)
    #enet.fit(data,label)
    #mask = enet.coef_ != 0
    #new_data = data[:,mask]
    #return new_data,mask



def SelectModel(modelname):

    if modelname == "SVM":
        
        model = SVC(kernel='rbf', C=6, gamma=0.05, probability=True)
    

    elif modelname == "GBDT":
        
        model = GradientBoostingClassifier(n_estimators=1200,max_depth=6,subsample=0.7,learning_rate=0.05)

    elif modelname == "RF":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=500)

    elif modelname == "XGBOOST":
        from xgboost.sklearn import XGBClassifier
        #import xgboost as xgb
        #model = xgb()
        print('+++++++++++++++++++++++++')
        model = xgb.XGBClassifier(max_depth=15, learning_rate=0.01,
                 n_estimators=500, silent=True,
                 objective="binary:logistic", booster='gbtree',
                 n_jobs=3, nthread=3, gamma=1, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=1, reg_lambda=2, scale_pos_weight=1,
                 base_score=0.5)

    elif modelname == "KNN":
        from sklearn.neighbors import KNeighborsClassifier as knn
        model = knn()
		
    elif modelname == "ERT":
        from sklearn.ensemble import ExtraTreesClassifier as ERT
        model = ExtraTreesClassifier()
    elif modelname == "lgb":
        model = lgb.LGBMClassifier(n_estimators=500,max_depth=5,learning_rate=0.01)
    else:
        pass
    return model

def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return acc, precision, sensitivity, specificity, MCC

def get_oof(clf,n_folds,X_train,y_train,X_test):
    ntrain = X_train.shape[0]
    ntest =  X_test.shape[0]
    classnum = len(np.unique(y_train))
    kf = KFold(n_splits=n_folds,random_state=1)
    oof_train = np.zeros((ntrain,classnum))
    oof_test = np.zeros((ntest,classnum))


    for i,(train_index, test_index) in enumerate(kf.split(X_train)):
        kf_X_train = X_train[train_index] # 
        kf_y_train = y_train[train_index] # 

        kf_X_test = X_train[test_index]  # k-fold

        clf.fit(kf_X_train, kf_y_train)
        oof_train[test_index] = clf.predict_proba(kf_X_test)

        oof_test += clf.predict_proba(X_test)
    oof_test = oof_test/float(n_folds)
    return oof_train, oof_test



#data1 = sio.loadmat(r'E:/code/all_3_optemDim_rand50.mat')

#data2=data1.get('label_mappedX')


#row=data2.shape[0]
#column=data2.shape[1]
#index = [i for i in range(row)]
#np.random.shuffle(index)
#index=np.array(index)
#data_1=data2[index,:]
#shu=data_1[:,np.array(range(1,column))]
#label=data_1[:,0]
#data_train = sio.loadmat('ACP740_optemDim.mat')
#data = data_train.get('ACP740_optemDim')  # Remove the data in the dictionary
#label1 = np.ones((376, 1))  # Value can be changed
#label2 = np.zeros((364, 1))
label = np.append(label1, label2)
X = data
y = label


#X=shu
#label[label==-1]=0
#y=label
sepscores = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
ypred=np.ones((1,2))*0.5
skf= StratifiedKFold(n_splits=5)
loo = LeaveOneOut()

for train, test in skf.split(X,y):

   print((train.shape, test.shape))
   #modelist = ['lgb','XGBOOST','SVM','RF','KNN']
   modelist = ['lgb','XGBOOST','SVM','GBDT','RF','ERT']
   #modelist = ['lgb','XGBOOST','SVM','GBDT','RF','ERT']
   #modelist = [ 'SVM', 'GBDT', 'RF', 'ERT']
   #modelist = ['SVM', 'GBDT',  'ERT']
   #modelist = ['lgb','XGBOOST','SVM','GBDT','RF']
   #modelist = ['lgb','XGBOOST','SVM']
clf1_ET = ExtraTreesClassifier()
clf2_SVM = SVC()
clf3_LR = LogisticRegression()
clf4_NB       = GaussianNB()
clf5_neigh = KNeighborsClassifier(n_neighbors=3)
clf6_MLP = MLPClassifier(random_state=1, max_iter=300)
clf7_RF = RandomForestClassifier()
clf8_tree = DecisionTreeRegressor(max_depth=3, random_state=0)
clf9_AdaBoost = AdaBoostClassifier()
clf10_lgbm = LGBMClassifier(random_state=5)
eclf1 = VotingClassifier(estimators=[('ExTrees', clf1_ET), ('svm', clf2_SVM), ('LR', clf3_LR), 
                         ('NB', clf4_NB ),('neigh', clf5_neigh ),('MLP', clf6_MLP ),('RF', clf7_RF ),
                         ('DecisionTree', clf8_tree ),('AdaBoost', clf9_AdaBoost ),('LGBM', clf10_lgbm )],
                         voting='hard')
eclf1.fit(X_train, y_train)
scores = eclf1.predict(X_test)
scores=np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))
result1=np.mean(scores,axis=0)

H1=result1.tolist()
sepscores.append(H1)

result=sepscores

data_csv = pd.DataFrame(data=result)
data_csv.to_csv('all_result.csv')


row=yscore.shape[0]
yscore=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore)
yscore_sum.to_csv('all_Fscore.csv')

ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('alltest.csv')

ypred=ypred[np.array(range(1,row)),:]
ypred_sum = pd.DataFrame(data=ypred)
ypred_sum.to_csv('allypred.csv')

fpr, tpr, _ = roc_curve(ytest[:,0], yscore[:,0])
auc_score=np.mean(scores, axis=0)[7]
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='STACK ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

data_csv = pd.DataFrame(data=result)
data_csv.to_csv('STACK.csv')
colum = ['ACC', 'precision', 'npv', 'Sn', 'Sp','MCC','F1','AUC']
ind=['1', '2', '3','4','5','6']
data_csv = pd.DataFrame(columns=colum, data=result,index=ind)
data_csv.to_csv(r'Stack-Optimalfeatures-ACP740.csv')

