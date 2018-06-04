# -*- coding: utf-8 -*-

#######################
#                     #
#     INCOME DATA     #
#                     #
#######################


# ----------
# Load data
# ----------

def create_y(age):
    """create the binomial age variable"""
    if(age < 45):
        return 0 # YOUNG
    else:
        return 1 # OLD

def load_data():
    """load data"""
    url = 'C:/tmp/census_data/adult'
    data_url = url + '.data'
    test_url = url + '.test'
    head = ['age','workclass','fnlwgt','education','education_num',
            'marital_status','occupation','relationship','race',
            'gender','capital_gain','capital_loss','hours_per_week',
            'native_country','income_bracket']
    data = pd.read_csv(data_url, sep=',', decimal='.', header=None, names=head)
    data.info()
    print(data.isnull().sum())
    data['test'] = 0
    test = pd.read_csv(test_url, sep=',', decimal='.', header=None, names=head)
    test.info()
    print(test.isnull().sum())
    test['test'] = 1    
    df = pd.concat([data, test])
    
    y_old = list(map(create_y, df['age']))
    df['y_old'] = y_old
    df = df.drop('age', 1)
    df['native_continent'] = df['native_country'].map({'?': '?', 'Cambodia': 'Asia', 'Canada': 'North-America', 'China': 'Asia', 'Columbia': 'South-America', 'Cuba': 'North-America', 'Dominican-Republic': 'North-America', 'Ecuador': 'South-America', 'El-Salvador': 'North-America', 'England': 'Europe', 'France': 'Europe', 'Germany': 'Europe', 'Greece': 'Europe', 'Guatemala': 'North-America', 'Haiti': 'North-America', 'Holand-Netherlands': 'Europe', 'Honduras': 'North-America', 'Hong': 'Asia', 'Hungary': 'Europe', 'India': 'Asia', 'Iran': 'Asia', 'Ireland': 'Europe', 'Italy': 'Europe', 'Jamaica': 'North-America', 'Japan': 'Asia', 'Laos': 'Asia', 'Mexico': 'North-America', 'Nicaragua': 'North-America', 'Outlying-US(Guam-USVI-etc)': 'North-America', 'Peru': 'South-America', 'Philippines': 'Asia', 'Poland': 'Europe', 'Portugal': 'Europe', 'Puerto-Rico': 'North-America', 'Scotland': 'Europe', 'South': 'Asia', 'Taiwan': 'Asia', 'Thailand': 'Asia', 'Trinadad&Tobago': 'South-America', 'United-States': 'North-America', 'Vietnam': 'Asia', 'Yugoslavia': 'Europe'})
    return df



# ----------
# Preprocess data
# ----------

def preprocess(df):
    """preprocess data : scale & create indicators"""
    #df['workclass_num'] = df.workclass.map({'Private':0, 'State-gov':1, 'Federal-gov':2, 'Self-emp-not-inc':3, 'Self-emp-inc':4, 'Local-gov':5, 'Without-pay':6})
    df['over50K'] = np.where(df.income_bracket == '<=50K', 0, 1)
    df['marital_status_num'] = df.marital_status.map({'Widowed':0, 'Divorced':1, 'Separated':2, 'Never-married':3, 'Married-civ-spouse':4, 'Married-AF-spouse':4, 'Married-spouse-absent':5})
    df['race_num'] = df.race.map({'White':0, 'Black':1, 'Asian-Pac-Islander':2, 'Amer-Indian-Eskimo':3, 'Other':4})
    df['gender_num'] = np.where(df.gender == 'Female', 0, 1)
    df['relationship_num'] = df.relationship.map({'Not-in-family':0, 'Unmarried':0, 'Own-child':0, 'Other-relative':0, 'Husband':1, 'Wife':1})
    df.info()
    # GET DUMMIES for categorical var
    df = pd.concat([df, pd.get_dummies(df[['workclass','education','marital_status','occupation','relationship','race','gender','native_country','native_continent']])], axis=1)
    df = df.drop(['workclass','education','marital_status','occupation','relationship','race','gender','native_country','native_continent','income_bracket'], 1)
    
    numericFeatures = ['fnlwgt', 'education_num', 'capital_gain',
                       'capital_loss', 'hours_per_week', #'workclass_num',
                       'marital_status_num', 'race_num',
                       'gender_num', 'relationship_num']
    scale_num = StandardScaler()
    scale_num.fit(df[numericFeatures])
    df[numericFeatures] = scale_num.transform(df[numericFeatures])
    #df = df.convert_objects(convert_numeric=True)
    
    plt.bar([0,1], height = df.y_old.value_counts(), tick_label = ['less than 45 years old','45 years old or more'])
    plt.show()
    return df



# ----------
# Split train & test
# ----------

def split_train_test(df, write=False):
    """split the test set"""
    test = df[df['test'] == 1].drop('test', 1)
    data = df[df['test'] == 0].drop('test', 1)
    
    if write:
        test.to_csv('cleaned_test.csv', sep=';', index=False)
        data.to_csv('cleaned_data.csv', sep=';', index=False)
    
    return test, data



# ----------
# Create train & validation subsets
# ----------

def create_sets(test, data, test_size=0.2, write=False):
    """Create train & valid set"""
    y_test = test['y_old']
    X_test = test.drop('y_old', 1)
    y_data = data['y_old']
    X_data = data.drop('y_old', 1)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=test_size, random_state=123)
    if write:
        pickle.dump((X_train, X_val, y_train, y_val), open(obj_save_path+'train_val_df.p', 'wb'))
        #X_train, X_val, y_train, y_val = pickle.load(open(obj_save_path+'train_val_df.p', 'rb'))
    return X_train, y_train, X_val, y_val, X_test, y_test 



# ----------
# Define baseline
# ----------

def get_baseline(y_test):
    """get the baseline"""
    base = pd.DataFrame(y_test)
    base['baseline'] = 0
    return metrics.accuracy_score(y_test, base['baseline'])



# ----------
# Checking models quality
# ----------

def verif_valid(model, X_val, y_val, cut=0.5):
    """Check if the tested model have a good quality"""
    if type(model) == Sequential:
        X_val = np.array(X_val)
    reality = y_val
    
    if ((type(model) == svm.classes.OneClassSVM) | (type(model) == lgb.basic.Booster) | (type(model) == svm.classes.LinearSVC) | (type(model) == Sequential)):
        pred_score = model.predict(X_val)
        if (type(model) == svm.classes.OneClassSVM):
            pred_score = np.where(pred_score == -1, 1, 0)
    else:
        pred_score = model.predict_proba(X_val)[:,1]
    
    plt.hist(pred_score)
    plt.title('Distribution of the prediction score')
    plt.show()
    #if (type(model) == Sequential):
    #    predictions = np.where(pred_score > 0.5, 1, 0)
    #else:
    predictions = np.where(pred_score > cut, 1, 0)
    
    print('Matrice de confusion :')
    conf_mat = confusion_matrix(reality, predictions)
    print(pd.DataFrame(conf_mat))
    print('Associated metrics :')
    print(classification_report(reality, predictions))
    fpr, tpr, _ = roc_curve(y_val, pred_score)
    
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'r-', lw=4)
    actual_fpr = conf_mat[1, 0] / (conf_mat[1, 0] + conf_mat[0, 0])
    actual_tpr = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])
    plt.plot(actual_fpr, actual_tpr, 'bo', lw=10)
    plt.xlabel('False-Positive Rate')
    plt.ylabel('True-Positive Rate')
    plt.title('ROC curve (with AUC = ' + str(round(roc_auc, 6)) + ')')
    plt.plot([0, 1], [0, 1], 'k-')
    plt.show()
    print('Score AUC : ' + str(roc_auc))
    print('Accuracy : ' + str(metrics.accuracy_score(y_val, predictions)))



# ----------
# I. Random Forest
# ----------

def plot_imp_rf(model_rf, X):
    """Plot a graph with the importances of the features from a random forest model"""
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
    plt.figure(figsize=(15, 10))
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), names, rotation=80)
    plt.xlim([-1, X.shape[1]])
    plt.show()

def rf_train(X_train, y_train, write=False):
    """Train the random forest model"""
    model_rf = RandomForestClassifier(n_estimators=1800,
                                      max_depth=20,
                                      min_samples_split=5,
                                      min_samples_leaf=20,
                                      bootstrap=True, oob_score=True, criterion='gini',
                                      random_state=321, n_jobs=4, verbose=0)
    model_rf.fit(X_train, y_train)
    if write:
        pickle.dump(model_rf, open(obj_save_path+'model_rf.p', 'wb'))
        #model_rf = pickle.load(open(obj_save_path+'model_rf.p', 'rb'))
    plot_imp_rf(model_rf, X_train)
    # oob_error = 1 - model_rf.oob_score_
    return model_rf
    


# ----------
# II. GRADIENT BOOSTING
# ----------

def gradb_train(X_train, y_train, write=False):
    """Train the gradient boosting model"""
    model_gradb = GradientBoostingClassifier(loss='deviance',
                                            learning_rate=0.2,
                                            n_estimators=100,
                                            subsample=0.9,
                                            #min_samples_leaf=10,
                                            max_depth=6,
                                            random_state=321, verbose=0)
    model_gradb.fit(X_train, y_train)
    if write:
        pickle.dump(model_gradb, open(obj_save_path+'model_gradb.p', 'wb'))
        #model_gradb = pickle.load(open(obj_save_path+'model_gradb.p', 'rb'))
    return model_gradb



# ----------
# III. XGBoost
# ----------

def xgb_train(X_train, y_train, write=False):
    """Train the XGBoost model"""
    model_xgb = xgb.XGBClassifier(max_depth=7,
                                  min_child_weight=1,
                                  learning_rate=0.01,
                                  n_estimators=5000,
                                  gamma=0.8,
                                  subsample=0.95,
                                  colsample_bytree=0.6,
                                  reg_alpha=0.0025,
                                  objective='binary:logistic',
                                  nthread=4,
                                  scale_pos_weight=1,
                                  seed=123)
    model_xgb.fit(X_train, y_train)
    if write:
        pickle.dump(model_xgb, open(obj_save_path+'model_xgb.p', 'wb'))
        #model_xgb = pickle.load(open(obj_save_path+'model_xgb.p', 'rb'))
    plot_importance(model_xgb)
    plt.show()
    return model_xgb



# ----------
# IV. LightGBM
# ----------

def lgbm_train(X_train, y_train, write=False):
    """Train the LightGBM"""
    d_train = lgb.Dataset(X_train, label=y_train)
    params = {'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.1,
            'num_iterations': 5000,
            'max_bin': 1000,
            'sub_feature': 0.3,
            'num_leaves': 13,
            'min_data': 150,
            'max_depth': 6,
            'n_jobs': 4}
    model_lgbm = lgb.train(params, d_train, 1500)
    if write:
        pickle.dump(model_lgbm, open(obj_save_path+'model_lgbm.p', 'wb'))
        #model_lgbm = pickle.load(open(obj_save_path+'model_lgbm.p', 'rb'))
    return model_lgbm



# ----------
# V. Bagging
# ----------

def bagxgb_train(X_train, y_train, size=10, write=False):
    """Train the bag of XGB models"""
    list_models = []
    #d_train = lgb.Dataset(X_train, label=y_train)
    with tqdm(total=size) as pbar:
        for nb in range(size):
            model = xgb.XGBClassifier(max_depth=7, min_child_weight=1, learning_rate=0.01, n_estimators=5000, gamma=0.8, subsample=0.95, colsample_bytree=0.6, reg_alpha=0.0025, objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                      seed=nb+1)
            model.fit(X_train, y_train)
            list_models.append(model)
            pbar.update()
    return list_models

def verif_valid_from_bag(list_models, X_val, reality, cut=0.5):
    """Check if the tested bagging model have a good quality"""
    bag_scores = []
    for model in list_models:
        pred_score = model.predict_proba(X_val)[:,1]
        bag_scores.append(pred_score.tolist())
    bag_scores = pd.DataFrame(bag_scores).transpose()
    #bag_scores = np.asmatrix(bag_scores)[:,0:5]
    pred_score = bag_scores.mean(axis=1)
    predictions = np.where(pred_score > cut, 1, 0)
    print('Confusion Matrix :')
    conf_mat = confusion_matrix(reality, predictions)
    print(pd.DataFrame(conf_mat))
    print('Associated metrics :')
    print(classification_report(reality, predictions))
    fpr, tpr, _ = roc_curve(reality, pred_score)
    roc_auc = auc(fpr, tpr)
    print('Avg False-Pos rate = ' + str(round(np.mean(fpr), 4)) +
        ' | Avg True-Pos rate : ' + str(round(np.mean(tpr), 4)))
    plt.plot(fpr, tpr, 'r-', lw=4)
    actual_fpr = conf_mat[1, 0] / (conf_mat[1, 0] + conf_mat[0, 0])
    actual_tpr = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])
    plt.plot(actual_fpr, actual_tpr, 'bo', lw=10)
    plt.xlabel('False-Positive Rate')
    plt.ylabel('True-Positive Rate')
    plt.title('ROC curve (with AUC = ' + str(round(roc_auc, 6)) + ')')
    plt.plot([0, 1], [0, 1], 'k-')
    plt.show()
    print('Score AUC : ' + str(roc_auc))
    print('Accuracy : ' + str(metrics.accuracy_score(reality, predictions)))



# ----------
# VII. Neural Network
# ----------

def nn_model():
    """Set the architecture of the neural network"""
    seed = 321
    np.random.seed(seed)
    rmsprop = RMSprop(lr=0.0001)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # for train, test in kfold.split(X, y):
    model_nn = Sequential()
    model_nn.add(Dense(100, input_shape=(117,), activation='relu'))
    model_nn.add(Dropout(0.5))
    model_nn.add(Dense(125, activation='relu', kernel_initializer='normal'))
    model_nn.add(Dropout(0.5))
    model_nn.add(Dense(30, activation='relu', kernel_initializer='normal'))
    model_nn.add(Dropout(0.5))
    model_nn.add(Dense(1, activation='sigmoid'))#softmax
    model_nn.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=rmsprop)
    #model_nn.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=rmsprop)
    # Compile model
    model_nn.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
    return model_nn

def nn_train(df_nn, write=False):
    """Train the neural network model"""
    #features = ['fnlwgt', 'education_num', 'capital_gain', 'capital_loss',
    # 'hours_per_week', 'workclass_num', 'over50K', 'marital_status_num',
    # 'race_num', 'gender_num', 'relationship_num']#, 'y_old']
    scale_num = StandardScaler()
    scale_num.fit(df.loc[:,df.columns != 'y_old'])
    df.loc[:,df.columns != 'y_old'] = scale_num.transform(df.loc[:,df.columns != 'y_old'])
    test_nn = df_nn[df_nn['test'] > 0.5].drop('test', 1)
    data_nn = df_nn[df_nn['test'] <= 0.5].drop('test', 1)
    y_test_nn = test_nn['y_old']
    X_test_nn = test_nn.drop('y_old', 1)
    y_data_nn = data_nn['y_old']
    X_data_nn = data_nn.drop('y_old', 1)
    X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
            X_data_nn, y_data_nn, test_size=0.2, random_state=321)
    model_nn = nn_model()
    model_nn.summary()
    hist = model_nn.fit(np.array(X_train_nn), np.array(y_train_nn),
                        epochs=150, validation_split=0.25,
                        shuffle=True, verbose=0)
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper right')
    plt.show()
    if write:
        model_nn.save(obj_save_path+'model_nn.p')
        # model_nn = load_model(obj_save_path+'model_nn.p')
    return model_nn, X_train_nn, y_train_nn, X_val_nn, y_val_nn, X_test_nn, y_test_nn



# ----------
# MAIN !
# ----------

if __name__ == '__main__':
    
    import numpy as np
    import pandas as pd
    from time import time
    from tqdm import tqdm
    import datetime
    import seaborn as sns
    import pickle
    import matplotlib.pyplot as plt
    from sklearn import svm, metrics
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc#, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier#, AdaBoostClassifier
    import xgboost as xgb
    from xgboost import plot_importance
    import lightgbm as lgb
    from keras.models import Sequential#, load_model
    from keras.layers.core import Dense, Dropout
    from keras.optimizers import RMSprop
    
    t0 = time()
    obj_save_path = './obj_save/'
    
    df = load_data()
    
    df.head()
    df.info()
    
    fig, ((a,b),(c,d),(e,f)) = plt.subplots(3,2,figsize=(10,10))
    plt.xticks(rotation=45)
    sns.countplot(df['workclass'],hue=df['y_old'],ax=f)
    sns.countplot(df['relationship'],hue=df['y_old'],ax=b)
    sns.countplot(df['marital_status'],hue=df['y_old'],ax=c)
    sns.countplot(df['race'],hue=df['y_old'],ax=d)
    #sns.countplot(df['gender'],hue=df['y_old'],ax=d)
    sns.countplot(df['gender'],hue=df['y_old'],ax=e)
    sns.countplot(df['native_continent'],hue=df['y_old'],ax=a)
    
    df = preprocess(df)
    test, data = split_train_test(df)
    X_train, y_train, X_val, y_val, X_test, y_test  = create_sets(test, data)
    baseline = get_baseline(y_test)
    print("Accuracy baseline = " + str(round(baseline, 4)))
    print('Ready to start ML part !')
    
    print('ML part. I : starting Random Forest !')
    model_rf = rf_train(X_train, y_train)
    #verif_valid(model_rf, X_val, y_val, cut=0.475)
    verif_valid(model_rf, X_test, y_test, cut=0.475)
    # AUC : 0.8030 | Accuracy : 0.7455
    print('ML part. I : Random Forest, done !')
    
    print('ML part. II : starting Gradient Boosting !')
    model_gradb = gradb_train(X_train, y_train)
    #verif_valid(model_gradb, X_val, y_val, cut=0.475)
    verif_valid(model_gradb, X_test, y_test, cut=0.475)
    # AUC : 0.8088 | Accuracy : 0.7496
    print('ML part. II : Gradient Boosting, done !')
    
    print('ML part. III : starting XGBoost !')
    model_xgb = xgb_train(X_train, y_train)
    #verif_valid(model_xgb, X_val, y_val, cut=0.475)
    verif_valid(model_xgb, X_test, y_test, cut=0.475)
    # AUC : 0.8251 | Accuracy : 0.7657
    print('ML part. III : XGBoost, done !')
    
    print('ML part. IV : starting LightGBM !')
    model_lgbm = lgbm_train(X_train, y_train)
    #verif_valid(model_lgbm, X_val, y_val, cut=0.475)
    verif_valid(model_lgbm, X_test, y_test, cut=0.475)
    # AUC : 0.8086 | Accuracy : 0.7482
    print('ML part. IV : LightGBM, done !')
    
    print('ML part. V : starting Bagging (from XGB) !')
    models_bagxgb = bagxgb_train(X_train, y_train, size=5)
    #verif_valid_from_bag(models_bagxgb, X_val, y_val, cut=0.475)
    verif_valid_from_bag(models_bagxgb, X_test, y_test, cut=0.475)
    # AUC : 0.8247 | Accuracy : 0.7644
    print('ML part. V : starting Bagging (from XGB) !')
    
    print('ML part. VII : starting Neural Network !')
    model_nn,X_train_nn,y_train_nn,X_val_nn,y_val_nn,X_test_nn,y_test_nn=nn_train(df)
    #verif_valid(model_nn, X_val_nn, y_val_nn, cut=0.5)
    verif_valid(model_nn, X_test_nn, y_test_nn, cut=0.5)
    # AUC : 0.7991 | Accuracy : 0.7493
    print('ML part. VII : Neural Network, done !')


    tf = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
    print('time end : ', tf)
    print('time enlapsed : ', int((time()-t0)/60), 'min')




