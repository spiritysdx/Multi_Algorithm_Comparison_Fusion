"""
模型：逻辑回归（LR）、随机森林（RF）、XGBoost（XGB）、LightGBM（LGBM）、支持向量机（SVC）、多层感知器（MLP）
采样：SMOTE
训练：K折交叉验证、网格搜索
结果：使用MLP使用SMOTE采样后数据进行概率预测
F1分数是一个结合了模型的精确度（Precision）和召回率（Recall）的指标。
在机器学习中，这两个指标通常是矛盾的，即提高一个可能会降低另一个。
F1分数的出现就是为了找到一个平衡点。
F1分数可以看作是精确度和召回率的一种综合度量，对于那些希望在正例和负例之间取得平衡的任务来说是一个有用的评估指标。
"""
import re
import math
import warnings
import tqdm
import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.pyplot import MultipleLocator
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, Binarizer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")


%matplotlib inline
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# conda install tqdm numpy pandas scipy matplotlib xgboost 
# conda install -c conda-forge sklearn-contrib-lightning
# conda install -c conda-forge tpot-imblearn
# conda install lightgbm
# conda install openpyxl

train = data5[:100]
label = data_kz['是否扩张'].values[:100]
predict = data5[100:]
# 定义参数网格
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

param_grid_lgbm = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20],
    'learning_rate': [0.01, 0.1, 0.2],
}

param_grid_xgb = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'max_depth': [5, 10, 20, 30, 40],
    'learning_rate': [0.01, 0.1, 0.2],
}

param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
}

param_grid_mlp = {
    'hidden_layer_sizes': [(50, 50), (100, 100), (50, 100)],
    'activation': ['relu', 'tanh'],
}

param_grid_lr = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'], 
}
sm = SMOTE(random_state=0)
xres, yres = sm.fit_resample(train.values, label)
xtrain, xvalid, ytrain, yvalid = train_test_split(train.values,label,random_state=50,test_size=0.2)
xtrain_res, xvalid_res, ytrain_res, yvalid_res = train_test_split(xres,yres,random_state=50,test_size=0.2)

# 采样前
def function_before(model, param_grid, cv_values=[5], n_jobs=16):
    best_auc = 0
    best_cv = None
    best_model = None
    best_params = None

    for cv in cv_values:
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=n_jobs)
        grid_search.fit(train.values, label)

        if grid_search.best_score_ > best_auc:
            best_model = grid_search.best_estimator_
            best_auc = grid_search.best_score_
            best_params = grid_search.best_params_
            best_cv = cv

    if best_model is None:
        raise ValueError("No model was trained successfully.")

    acc = round(cross_val_score(best_model, train.values, label, cv=best_cv, scoring='accuracy').mean(), 2)
    recall = round(cross_val_score(best_model, train.values, label, cv=best_cv, scoring='recall').mean(), 2)
    precision = round(cross_val_score(best_model, train.values, label, cv=best_cv, scoring='precision').mean(), 2)
    f1 = round(cross_val_score(best_model, train.values, label, cv=best_cv, scoring='f1').mean(), 2)

    result_dict = {
        'best_model': best_model,
        'best_auc': round(best_auc, 4),
        'best_params': best_params,
        'acc': acc,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'best_cv': best_cv
    }

    return result_dict

cv_values = [5, 10]
best_results_rf = function_before(RandomForestClassifier(random_state=100), param_grid_rf, cv_values)
best_results_lgbm = function_before(LGBMClassifier(random_state=100), param_grid_lgbm, cv_values)
best_results_xgb = function_before(XGBClassifier(random_state=100), param_grid_xgb, cv_values)
best_results_svc = function_before(SVC(probability=True, random_state=100), param_grid_svc, cv_values)
best_results_mlp = function_before(MLPClassifier(random_state=100), param_grid_mlp, cv_values)
best_results_lr = function_before(LogisticRegression(random_state=100), param_grid_lr, cv_values)

model0 = best_results_rf['best_model']
model1 = best_results_lgbm['best_model']
model2 = best_results_xgb['best_model']
model3 = best_results_svc['best_model']
model4 = best_results_mlp['best_model']
model5 = best_results_lr['best_model']

model0.fit(xtrain,ytrain)
model1.fit(xtrain,ytrain)
model2.fit(xtrain,ytrain)
model3.fit(xtrain,ytrain)
model4.fit(xtrain,ytrain)
model5.fit(xtrain,ytrain)

def f(model_list,name_list,types='train'):
    plt.figure(figsize=(8, 7), dpi=100, facecolor='w')    # dpi:每英寸长度的像素点数；facecolor 背景颜色
    plt.xlim((-0.01, 1.02))  # x,y 轴刻度的范围
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))  #绘制刻度
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    if types == 'test':
        for model,name in zip(model_list,name_list):
            ytest_prob = model.predict_proba(xvalid)[:,1]
            fpr, tpr, _ = metrics.roc_curve(yvalid, ytest_prob)
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, '-', lw=2.5, label=f'{name} AUC:%.5f' % auc)  # 绘制AUC 曲线
    else:
        for model,name in zip(model_list,name_list):
            ytest_prob = model.predict_proba(xtrain)[:,1]
            fpr, tpr, _ = metrics.roc_curve(ytrain, ytest_prob)
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, '-', lw=2.5, label=f'{name} AUC:%.5f' % auc)  # 绘制AUC 曲线
    plt.legend(loc='best',fontsize=15)    # 设置显示标签的位置
    plt.xlabel('假阳率', fontsize=14)   #绘制x,y 坐标轴对应的标签
    plt.ylabel('真阳率', fontsize=14)
    plt.tick_params(labelsize=23)

    plt.grid(b=True, ls=':')  # 绘制网格作为底板;b是否显示网格线；ls表示line style
    # plt.savefig(f'q1_roc_auc({types}(采样前)).png',dpi=1000)
    plt.savefig(f'q1_roc_auc({types}(采样前)).eps')
    plt.show()

f([model0, model1, model2, model3, model4, model5], ['RF', 'LGBM', 'XGB', 'SVC', 'MLP', 'LR'], 'test')
f([model0, model1, model2, model3, model4, model5], ['RF', 'LGBM', 'XGB', 'SVC', 'MLP', 'LR'], 'train')

# 采样后
def function_after(model, param_grid, cv_values=[5], n_jobs=16):
    best_auc = 0
    best_cv = None
    best_model = None
    best_params = None

    for cv in cv_values:
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=n_jobs)
        grid_search.fit(xres, yres)
        
        if grid_search.best_score_ > best_auc:
            best_model = grid_search.best_estimator_
            best_auc = grid_search.best_score_
            best_params = grid_search.best_params_
            best_cv = cv

    if best_model is None:
        raise ValueError("No model was trained successfully.")

    acc = round(cross_val_score(best_model, xres, yres, cv=best_cv, scoring='accuracy').mean(), 2)
    recall = round(cross_val_score(best_model, xres, yres, cv=best_cv, scoring='recall').mean(), 2)
    precision = round(cross_val_score(best_model, xres, yres, cv=best_cv, scoring='precision').mean(), 2)
    f1 = round(cross_val_score(best_model, xres, yres, cv=best_cv, scoring='f1').mean(), 2)
    
    result_dict = {
        'best_model': best_model,
        'best_auc': round(best_auc, 4),
        'best_params': best_params,
        'acc': acc,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'best_cv': best_cv
    }
    
    return result_dict

cv_values = [5, 10]
best_results_rf = function_after(RandomForestClassifier(random_state=100), param_grid_rf, cv_values)
best_results_lgbm = function_after(LGBMClassifier(random_state=100), param_grid_lgbm, cv_values)
best_results_xgb = function_after(XGBClassifier(random_state=100), param_grid_xgb, cv_values)
best_results_svc = function_after(SVC(probability=True, random_state=100), param_grid_svc, cv_values)
best_results_mlp = function_after(MLPClassifier(random_state=100), param_grid_mlp, cv_values)
best_results_lr = function_after(LogisticRegression(random_state=100), param_grid_lr, cv_values)

model0 = best_results_rf['best_model']
model1 = best_results_lgbm['best_model']
model2 = best_results_xgb['best_model']
model3 = best_results_svc['best_model']
model4 = best_results_mlp['best_model']
model5 = best_results_lr['best_model']

model0.fit(xtrain_res,ytrain_res)
model1.fit(xtrain_res,ytrain_res)
model2.fit(xtrain_res,ytrain_res)
model3.fit(xtrain_res,ytrain_res)
model4.fit(xtrain_res,ytrain_res)
model5.fit(xtrain_res,ytrain_res)

def f(model_list,name_list,types='train'):
    plt.figure(figsize=(8, 7), dpi=100, facecolor='w')    # dpi:每英寸长度的像素点数；facecolor 背景颜色
    plt.xlim((-0.01, 1.02))  # x,y 轴刻度的范围
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))  #绘制刻度
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    if types == 'test':
        for model,name in zip(model_list,name_list):
            ytest_prob = model.predict_proba(xvalid_res)[:,1]
            fpr, tpr, _ = metrics.roc_curve(yvalid_res, ytest_prob)
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, '-', lw=2.5, label=f'{name} AUC:%.5f' % auc)  # 绘制AUC 曲线
    else:
        for model,name in zip(model_list,name_list):
            ytest_prob = model.predict_proba(xtrain_res)[:,1]
            fpr, tpr, _ = metrics.roc_curve(ytrain_res, ytest_prob)
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, '-', lw=2.5, label=f'{name} AUC:%.5f' % auc)  # 绘制AUC 曲线
    plt.legend(loc='best',fontsize=15)    # 设置显示标签的位置
    plt.xlabel('假阳率', fontsize=14)   #绘制x,y 坐标轴对应的标签
    plt.ylabel('真阳率', fontsize=14)
    plt.tick_params(labelsize=23)

    plt.grid(b=True, ls=':')  # 绘制网格作为底板;b是否显示网格线；ls表示line style
    # plt.savefig(f'q1_roc_roc_auc({types}（采样后）).png',dpi=1000)
    plt.savefig(f'q1_roc_roc_auc({types}（采样后）).eps')
    plt.show()

f([model0, model1, model2, model3, model4, model5], ['RF', 'LGBM', 'XGB', 'SVC', 'MLP', 'LR'], 'test')
f([model0, model1, model2, model3, model4, model5], ['RF', 'LGBM', 'XGB', 'SVC', 'MLP', 'LR'], 'train')

# 创建mlp和svc的DataFrame
df1 = pd.DataFrame([best_results_mlp])
df1 = df1.transpose()
df1.columns = ['mlp']

df2 = pd.DataFrame([best_results_svc])
df2 = df2.transpose()
df2.columns = ['svc']

# 创建RF的DataFrame
df_rf = pd.DataFrame([best_results_rf])
df_rf = df_rf.transpose()
df_rf.columns = ['rf']

# 创建LGBM的DataFrame
df_lgbm = pd.DataFrame([best_results_lgbm])
df_lgbm = df_lgbm.transpose()
df_lgbm.columns = ['lgbm']

# 创建XGB的DataFrame
df_xgb = pd.DataFrame([best_results_xgb])
df_xgb = df_xgb.transpose()
df_xgb.columns = ['xgb']

# 创建LR的DataFrame
df_lr = pd.DataFrame([best_results_lr])
df_lr = df_lr.transpose()
df_lr.columns = ['lr']

# 合并所有的DataFrame
result_df = pd.concat([df1, df2, df_rf, df_lgbm, df_xgb, df_lr], axis=1)
result_df = result_df.T.drop(['best_model', 'best_params'], axis = 1)
result_df

mlp_model.fit(xres,yres)
mlp_model_result = [round(i,4) for i in mlp_model.predict_proba(data5.values)[:,1]]
mlp_model_result = pd.DataFrame({'扩张概率':mlp_model_result})
mlp_model_result.to_excel('q12扩张概率.xlsx',index=False)
# 'name':data1['ID'].values,
mlp_model_result