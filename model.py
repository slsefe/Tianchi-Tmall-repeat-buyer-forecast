import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import roc_auc_score

def make_train_set(trainFeatures, trainData, userFeatures, sellerFeatures, userSellerFeatures):
    if os.path.exists(trainFeatures):
        trainSet = pickle.load(open(trainFeatures,'rb'))
    else:     
        trainSet = pd.read_csv(trainData)
        trainSet.rename(columns={'merchant_id':'seller_id'},inplace=True)
        userInfo = pickle.load(open(userFeatures,'rb'))
        trainSet = pd.merge(trainSet,userInfo,how='left',on=['user_id'])
        sellerInfo = pickle.load(open(sellerFeatures,'rb'))
        trainSet = pd.merge(trainSet,sellerInfo,how='left',on=['seller_id'])
        userSellers = pickle.load(open(userSellerFeatures,'rb'))
        trainSet = pd.merge(trainSet,userSellers,how='left',on=['user_id','seller_id'])
        del userInfo,sellerInfo,userSellers
        pickle.dump(trainSet,open(trainFeatures,'wb'))
    return trainSet
    
def make_test_set(testFeatures, testData, userFeatures, sellerFeatures, userSellerFeatures):
    if os.path.exists(testFeatures):
        testSet = pickle.load(open(testFeatures,'rb'))
    else:     
        testSet = pd.read_csv(testData)
        testSet.rename(columns={'merchant_id':'seller_id'},inplace=True)
        userInfo = pickle.load(open(userFeatures,'rb'))
        testSet = pd.merge(testSet,userInfo,how='left',on=['user_id'])
        sellerInfo = pickle.load(open(sellerFeatures,'rb'))
        testSet = pd.merge(testSet,sellerInfo,how='left',on=['seller_id'])
        userSellers = pickle.load(open(userSellerFeatures,'rb'))
        testSet = pd.merge(testSet,userSellers,how='left',on=['user_id','seller_id'])
        del userInfo,sellerInfo,userSellers
        pickle.dump(testSet,open(testFeatures,'wb'))
    return testSet
    
trainFeatures = 'd:/JulyCompetition/features/trainSetWithFeatures.pkl'
testFeatures = 'd:/JulyCompetition/features/testSetWithFeatures.pkl'
trainData = 'd:/JulyCompetition/input/train_format1.csv'
testData = 'd:/JulyCompetition/input/test_format1.csv'
userFeatures = 'd:/JulyCompetition/features/userInfo_Features.pkl'
sellerFeatures = 'd:/JulyCompetition/features/sellerInfo_Features.pkl'
userSellerFeatures = 'd:/JulyCompetition/features/userSellerActions.pkl'

trainSet = make_train_set(trainFeatures, trainData, userFeatures, sellerFeatures, userSellerFeatures)
testSet = make_test_set(testFeatures, testData, userFeatures, sellerFeatures, userSellerFeatures)

from sklearn.model_selection import train_test_split
x = trainSet.loc[:,trainSet.columns != 'label']
y = trainSet.loc[:,trainSet.columns == 'label']
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 2018)

del X_train['user_id']
del X_train['seller_id']
del X_test['user_id']
del X_test['seller_id']
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

# LR
params=[
    {'penalty':['l1'],
    'C':[100,1000],
    'solver':['liblinear']},
    {'penalty':['l2'],
    'C':[100,1000],
    'solver':['lbfgs']}]
clf = LogisticRegression(random_state=2018, max_iter=1000,  verbose=2)
grid = GridSearchCV(clf, params, scoring='roc_auc',cv=10, verbose=2)
grid.fit(X_train, y_train) 
lr=grid.best_estimator_

# xgboost

x_val = x_test.iloc[:int(x_test.shape[0]/2),:]
y_val = y_test.iloc[:int(y_test.shape[0]/2),:]
 
x_test = x_test.iloc[int(x_test.shape[0]/2):,:] 
y_test = y_test.iloc[int(y_test.shape[0]/2):,:]
 
del x_train['user_id'],x_train['seller_id'],x_val['user_id'],x_val['seller_id']
 
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_val, label=y_val)

param = {'n_estimators': 500,
     'max_depth': 4, 
     'min_child_weight': 3,
     'gamma':0.3,
     'subsample': 0.8,
     'colsample_bytree': 0.8,  
     'eta': 0.125,
     'silent': 1, 
     'objective': 'binary:logistic',
     'eval_metric':'auc',
     'nthread':16
    }
plst = param.items()
evallist = [(dtrain, 'train'),(dtest,'eval')]
bst = xgb.train(plst, dtrain, 500, evallist, early_stopping_rounds=10)

def create_feature_map(features):
    outfile = open(r'd:/JulyCompetition/output/featureMap/firstXGB.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
    
def feature_importance(bst_xgb):
    importance = bst_xgb.get_fscore(fmap=r'd:/JulyCompetition/output/featureMap/firstXGB.fmap')
    importance = sorted(importance.items(), reverse=True)
 
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    return df
 
create_feature_map(list(x_train.columns[:]))
feature_importance = feature_importance(bst)
feature_importance.sort_values("fscore", inplace=True, ascending=False)
 
users = x_test[['user_id', 'seller_id']].copy()
del x_test['user_id']
del x_test['seller_id']
x_test_DMatrix = xgb.DMatrix(x_test)
y_pred = bst.predict(x_test_DMatrix)
roc_auc_score(y_test,y_pred)

# multimodel
def get_models(SEED=2018):
    """
    :parameters: None: None
    :return: models: Dict
    :Purpose: 
    """
    lgb = lgb.LGBMClassifier(num_leaves=50,learning_rate=0.05,n_estimators=250,class_weight='balanced',random_state=SEED)
    xgb = xgb.XGBClassifier(max_depth=4,min_child_weight=2,learning_rate=0.15,n_estimators=150,nthread=4,gamma=0.2,subsample=0.9,colsample_bytree=0.7, random_state=SEED)
    knn = KNeighborsClassifier(n_neighbors=1250,weights='distance',n_jobs=-1)
    lr = LogisticRegression(C=150,class_weight='balanced',solver='liblinear', random_state=SEED)
    nn = MLPClassifier(solver='lbfgs', activation = 'logistic',early_stopping=False,alpha=1e-3,hidden_layer_sizes=(100,5), random_state=SEED)
    gb = GradientBoostingClassifier(learning_rate=0.01,n_estimators=600,min_samples_split=1000,min_samples_leaf=60,max_depth=10,subsample=0.85,max_features='sqrt',random_state=SEED)
    rf = RandomForestClassifier(min_samples_leaf=30,min_samples_split=120,max_depth=16,n_estimators=400,n_jobs=2,max_features='sqrt',class_weight='balanced',random_state=SEED)

    models = {
              'knn': knn, 
              'xgb':xgb,
              'lgm':lgb,
              'mlp-nn': nn,
              'random forest': rf,
              'gbm': gb,
              'logistic': lr
              }

    return models
def train_predict(model_list):
    """
    :parameters: model_list: Dict
    :return: P: pd.DataFrame
    :Purpose: 
    """
    Preds_stacker = np.zeros((y_test.shape[0], len(model_list)))
    Preds_stacker = pd.DataFrame(Preds_stacker)

    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        if name == 'xgb' or name == 'lgm':
            m.fit(x_train,y_train.values.ravel(),eval_metric='auc')
        else:
            m.fit(x_train, y_train.values.ravel())
        Preds_stacker.iloc[:, i] = m.predict_proba(x_test)[:, 1]
        cols.append(name)
        print("done")

    Preds_stacker.columns = cols
    print("Done.\n")
    return Preds_stacker

def score_models(Preds_stacker, true_preds):
    """
    :parameters: Preds_stacker: pd.DataFrame   true_preds: pd.Series
    :return: None
    :Purpose: cal AUC for every model
    """
    print("Scoring models.")
    for m in Preds_stacker.columns:
        score = roc_auc_score(true_preds, Preds_stacker.loc[:, m])
        print("%-26s: %.3f" % (m, score))
    print("Done.\n")
    
models = get_models()
Preds = train_predict(models)
score_models(Preds, y_test)

# stacking
def train_base_learners(base_learners, xTrain, yTrain, verbose=True):
    """
    :parameters: model_list: Dict， xTrain：pd.DataFrame， yTrain：pd.DataFrame
    :return: None
    :Purpose: train base model
    """
    if verbose: print("Fitting models.")
    for i, (name, m) in enumerate(base_learners.items()):
        if verbose: print("%s..." % name, end=" ", flush=False)
        if name == 'xgb' or name == 'lgm':
            m.fit(xTrain,yTrain.values.ravel(),eval_metric='auc')
        else:
            m.fit(xTrain, yTrain.values.ravel())
        if verbose: print("done")
 
def predict_base_learners(pred_base_learners, inp, verbose=True):
    """
    :parameters: model_list: Dict， inp
    :return: P：pd.DataFrame
    :Purpose: base model prediction
    """
    P = np.zeros((inp.shape[0], len(pred_base_learners)))
    if verbose: print("Generating base learner predictions.")
    for i, (name, m) in enumerate(pred_base_learners.items()):
        if verbose: print("%s..." % name, end=" ", flush=False)
        p = m.predict_proba(inp)
        # With two classes, need only predictions for one class
        P[:, i] = p[:, 1]
        if verbose: print("done")
    return P
  
def ensemble_predict(base_learners, meta_learner, inp, verbose=True):
    """
    :parameters: model_list: Dict， meta_learner， inp
    :return: P_pred， P
    :Purpose: meta model prediction
    """
    P_pred = predict_base_learners(base_learners, inp, verbose=verbose)
    return P_pred, meta_learner.predict_proba(P_pred)[:, 1]

## 1.base model
base_learners = get_models()
## 2.meta model
meta_learner = GradientBoostingClassifier(
    n_estimators=5000,
    loss="exponential",
    max_features=3,
    max_depth=4,
    subsample=0.8,
    learning_rate=0.0025, 
    random_state=SEED
)
 
## splitting data as 0.5:0.5 to train base model and meta model respectively
xtrain_base, xpred_base, ytrain_base, ypred_base = train_test_split(
    x_train, y_train, test_size=0.5, random_state=SEED)
## 3.train base model
train_base_learners(base_learners, xtrain_base, ytrain_base)
## 4.base model prediction
P_base = predict_base_learners(base_learners, xpred_base)
## 5.train meta model
meta_learner.fit(P_base, ypred_base.values.ravel())
## 6.meta model prediction
P_pred, p = ensemble_predict(base_learners, meta_learner, x_test)
print("\nEnsemble ROC-AUC score: %.3f" % roc_auc_score(y_test, p))
