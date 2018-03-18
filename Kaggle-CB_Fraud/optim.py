# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from time import time
import datetime
from math import sqrt
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb
from xgboost import plot_importance
import lightgbm as lgb
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.optimizers import RMSprop

t0 = time()
obj_save_path = './obj_save/'


#######################
#                     #
#     CREDIT CARD     #
#   FRAUD DETECTION   #
#                     #
#######################


# ----------
# Load data
# ----------

data = pd.read_csv('./data/creditcard.csv', sep=',', decimal='.')
data.info()
data.isnull().sum()


# ----------
# Preprocess data
# ----------

numericFeatures = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
scale_num = StandardScaler()
scale_num.fit(data[numericFeatures])
data[numericFeatures] = scale_num.transform(data[numericFeatures])
data.head()

plt.bar([0,1], height = data.Class.value_counts(), tick_label = ['No fraud','Fraud'])


# ----------
# Create train & validation subsets
# ----------

y_data = data['Class']
X_data = data.drop('Class', 1)

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.33, random_state=29)

# pickle.dump((X_train, X_val, y_train, y_val), open(obj_save_path+'train_val_df.p', 'wb'))
X_train, X_val, y_train, y_val = pickle.load(open(obj_save_path+'train_val_df.p', 'rb'))
print('Ready to start ML part !')


# ----------
# I. Random Forest
# ----------

print('ML part. I : starting Random Forest !')

evol_AUC=[]
evol = np.arange(30,210,5)
nb = len(evol)
with tqdm(total=nb) as pbar:
    for param in evol:
        for i in np.arange(1,6):
            auc = []
            model_rf = RandomForestClassifier(n_estimators = param, #50
                                              max_depth=20,
                                              min_samples_split=5,
                                              min_samples_leaf=20,
                                              bootstrap=True, oob_score=True, criterion='gini',
                                              #random_state=321,
                                              n_jobs=4, verbose=1)
            model_rf.fit(X_train, y_train)
            pred = model_rf.predict(X_val)
            print(confusion_matrix(y_val, pred))
            res = roc_auc_score(y_val, pred)
            print(res)
            auc.append(res)
        cv_auc = np.mean(auc)
        print('AUC : ', cv_auc)
        evol_AUC.append(cv_auc)
        pbar.update()

plt.plot(evol, evol_AUC)
plt.title("Random Forest - evol param")
plt.show()

#evol_AUC = [0.9072688515853984, 0.9072795097255529, 0.8949231725730526, 0.8887609912070343, 0.9011066702193802, 0.9041771027622346, 0.9134363620214939, 0.9072688515853984, 0.9072741806554756, 0.9011066702193802, 0.9041984190425438, 0.8949338307132072, 0.9072741806554756, 0.9103712585487167, 0.9041984190425438, 0.9103659294786394, 0.9072741806554756, 0.8949338307132072, 0.8980149213962164, 0.9072688515853984, 0.9072795097255529, 0.9041877609023893, 0.9041930899724665, 0.9072795097255529, 0.9072795097255529, 0.9072795097255529, 0.9072795097255529, 0.9072795097255529, 0.907263522515321, 0.9041930899724665, 0.9072795097255529, 0.9041930899724665, 0.9072795097255529, 0.9041930899724665, 0.9041877609023893, 0.9103606004085621]


# pickle.dump(model_rf, open(obj_save_path+'model_rf.p', 'wb'))
#model_rf = pickle.load(open(obj_save_path+'model_rf.p', 'rb'))

def plot_imp_rf(model_rf, X):
    importances = model_rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model_rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    names = X.columns[indices]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print(str(f+1)+'. feature '+str(names[f])+' ('+str(importances[indices[f]])+')')
    # Plot the feature importances of the forest
    plt.figure(figsize=(10, 5))
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), names, rotation=80)
    plt.xlim([-1, X.shape[1]])
    plt.show()

plot_imp_rf(model_rf, X_train)
# oob_error = 1 - model_rf.oob_score_

def verif_valid(model, X_val, y_val):
    if type(model) == Sequential:
        X_val = np.array(X_val)
    reality = y_val
    predictions = model.predict(X_val)
    if type(model) == lgb.basic.Booster:
        for i in range(len(predictions)):
            if predictions[i] >= 0.5:  # threshold = 0.5
               predictions[i] = 1
            else:
               predictions[i] = 0
    if len(predictions.shape) == 2:
        predictions = predictions[:, 0]
    print('Matrice de confusion :')
    print(confusion_matrix(reality, predictions))
    print('Métriques de précision associées :')
    print(classification_report(reality, predictions))
    print('Score AUC :')
    print(roc_auc_score(reality, predictions))

verif_valid(model_rf, X_val, y_val)

print('ML part. I : Random Forest, done !')


# ----------
# II. GRADIENT BOOSTING
# ----------

print('ML part. II : starting Gradient Boosting !')

model_gradb = GradientBoostingClassifier(loss='deviance',
                                        learning_rate=0.2,
                                        n_estimators=100,
                                        subsample=0.9,
                                        #min_samples_leaf=10,
                                        max_depth=6,
                                        random_state=321, verbose=0)

model_gradb.fit(X_train, y_train)

# pickle.dump(model_gradb, open(obj_save_path+'model_gradb.p', 'wb'))
#model_gradb = pickle.load(open(obj_save_path+'model_gradb.p', 'rb'))

verif_valid(model_gradb, X_val, y_val)

print('ML part. II : Gradient Boosting, done !')



# ----------
# III. XGBoost
# ----------

print('ML part. III : starting XGBoost !')

model_xgb = xgb.XGBClassifier(base_score=0.5,
                              subsample=0.8,
                              max_delta_step=2,
                              max_depth=7,
                              min_child_weight=3,
                              learning_rate=0.1,
                              n_estimators=580,
                              objective='binary:logistic',
                              #booster='gbtree',
                              colsample_bytree=0.85,
                              gamma=0,
                              reg_alpha=0,
                              reg_lambda=1,
                              scale_pos_weight=1,
                              seed=321, silent=0)

model_xgb.fit(X_train, y_train)

print(model_xgb)

# pickle.dump(model_xgb, open(obj_save_path+'model_xgb.p', 'wb'))
#model_xgb = pickle.load(open(obj_save_path+'model_xgb.p', 'rb'))

plot_importance(model_xgb)
plt.show()

verif_valid(model_xgb, X_val, y_val)

print('ML part. III : XGBoost, done !')



# ----------
# IV. LightGBM
# ----------

print('ML part. IV : starting LightGBM !')

d_train = lgb.Dataset(X_train, label=y_train)
params = {}
params['learning_rate'] = 0.1
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10
model_lgbm = lgb.train(params, d_train, 500)

# pickle.dump(model_lgbm, open(obj_save_path+'model_lgbm.p', 'wb'))
#model_lgbm = pickle.load(open(obj_save_path+'model_lgbm.p', 'rb'))

verif_valid(model_lgbm, X_val, y_val)

print('ML part. IV : LightGBM, done !')




# ----------
# VI. ADABOOST
# ----------

print('ML part. III : starting Adaboost !')

model_adab = AdaBoostClassifier(#base_estimator=RandomForestClassifier(),
                               n_estimators=300,
                               learning_rate=0.28,
                               #loss='linear',
                               random_state=321)

model_adab.fit(X_train, y_train)

# pickle.dump(model_adab, open(obj_save_path+'model_adab.p', 'wb'))
#model_adab = pickle.load(open(obj_save_path+'model_adab.p', 'rb'))

verif_valid(model_adab, X_val, y_val)

print('ML part. III : Adaboost, done !')


# ----------
# XIII. Neural Network
# ----------

print('ML part. V : starting Neural Network !')

y_train_nn = pd.get_dummies(y_train)
y_val_nn = pd.get_dummies(y_val)

def nn_model():
    seed = 321
    np.random.seed(seed)
    rmsprop = RMSprop(lr=0.0001)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # for train, test in kfold.split(X, y):
    model_nn = Sequential()
    model_nn.add(Dense(50, input_shape=(30,), activation='relu'))
    model_nn.add(Dropout(0.2))
    model_nn.add(Dense(100, activation='relu', kernel_initializer='normal'))
    model_nn.add(Dropout(0.3))
    model_nn.add(Dense(2, activation='softmax'))
    model_nn.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=rmsprop)

    # Compile model
    model_nn.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
    return model_nn

model_nn = nn_model()
model_nn.summary()
hist = model_nn.fit(np.array(X_train), np.array(y_train_nn),
                    epochs=15, validation_split=0.33,
                    shuffle=True, verbose=1)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()

# model_nn.save(obj_save_path+'model_nn.p')
# model_nn = load_model(obj_save_path+'model_nn.p')

loss, acc = model_nn.evaluate(np.array(X_val), np.array(y_val_nn))
print('The accuracy on the test set is ',(acc*100),'%')


verif_valid(model_nn, X_val, y_val)

print('ML part. V : Neural Network, done !')


# ----------
# XIV. Stacking model (Neural Network)
# ----------

print('ML part. VI : starting Stacking Model (Neural Network) !')


def pred_ML(X):
    model_rf = pickle.load(open(obj_save_path+'model_rf.p', 'rb'))
    model_xgb = pickle.load(open(obj_save_path+'model_xgb.p', 'rb'))
    model_adab = pickle.load(open(obj_save_path+'model_adab.p', 'rb'))
    model_gradb = pickle.load(open(obj_save_path+'model_gradb.p', 'rb'))
    model_nn = load_model(obj_save_path+'model_nn.p')

    rf = pd.DataFrame(model_rf.predict(X))
    xgb = pd.DataFrame(model_xgb.predict(X))
    adab = pd.DataFrame(model_adab.predict(X))
    gradb = pd.DataFrame(model_gradb.predict(X))
    nn = pd.DataFrame(model_nn.predict(np.array(X))[:, 0])
    X_stack = pd.concat([rf, xgb, adab, gradb, nn], axis=1)
    return X_stack

X_train_stack = pred_ML(X_train)
X_val_stack = pred_ML(X_val)
#X_test_stack = pred_ML(X_test)

def stacking_nn_model():
    seed = 321
    np.random.seed(seed)
    model_nn = Sequential()
    model_nn.add(Dense(5, input_dim=5, activation='relu', kernel_initializer='normal'))
    model_nn.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model_nn.compile(optimizer='adam', loss='mean_squared_error')
    return model_nn

model_stack_nn = stacking_nn_model()
model_stack_nn.summary()
hist = model_stack_nn.fit(np.array(X_train_stack), np.array(y_train),
                          epochs=20, validation_split=0.33,
                          shuffle=True, verbose=1)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model RMSE')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()

# model_stack_nn.save(obj_save_path+'model_stack_nn.p')
model_stack_nn = load_model(obj_save_path+'model_stack_nn.p')

verif_valid(model_stack_nn, X_val_stack, y_val)

print('ML part. VI : Stacking Model (Neural Network), done !')


# ----------
# XV. Stacking Model (mean)
# ----------

print('ML part. VII : starting Stacking Model (mean) !')


def check_predictions(predictions, reality):
    plt.hist(predictions)
    plt.title('Histogram of Predictions')
    plt.show()
    plt.plot(predictions, reality, 'ro')
    plt.xlabel('Predictions')
    plt.ylabel('Reality')
    plt.title('Predictions x Reality')
    plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
    plt.show()
#    print('Explained var (predictions) = '+str(explained_variance_score(reality, predictions)))
#    print('RMSE = '+str(sqrt(mean_squared_error(reality, predictions))))

pred_val_stack_avg = X_val_stack.mean(axis=1)
check_predictions(pred_val_stack_avg, y_val)

print('ML part. VII : Stacking Model (mean), done !')


# ----------
# XVI. Predictions on the test set
# ----------

# Différents models construits :
# [model_rf, model_xgb, model_adab, model_gradb, model_nn, model_stack_nn]
#
# Choix du plus performant :
#model = model_stack_nn
#
#X_test = np.array(X_test)
#X_test_stack = np.array(X_test_stack)
#
#predictions = model.predict(X_test_stack)
#predictions = predictions[:, 0]
#plt.hist(predictions)
#plt.show()
#
#predictions = X_test['tH2'] + predictions
#plt.hist(predictions)
#plt.show()
#
#res = pd.DataFrame({
#        'date': X_test_date,
#        'insee': X_test_insee,
#        'ech': X_test_ech,
#        'tH2_obs': predictions},
#        columns=['date', 'insee', 'ech', 'tH2_obs'])
#
#res.to_csv('new_sub.csv', sep=";", index=False)
#
#print('THE END : submission ready !')


tf = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
print('time end : ', tf)
print('time enlapsed : ', int((time()-t0)/60), 'min')



