#!/usr/bin/env python3

#############################
#   TEXT MINING 
#############################
import csv,sys
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import cross_validation
#from sklearn import datasets
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import neighbors
#from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
#from Ranking import Ranking as rk
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier

#######################################################################
# Partie 1 : Apprentissage par le biais de méthodes de classification #
#######################################################################

#Ouverture du fichier training généré
f = open('dupl_email.csv', 'r', encoding = 'utf8')
#Stockage des variables
reader = csv.reader(f, delimiter=',', quotechar='"')
#X est utilisé pour representer les données
X=[]
#Y est utilisé pour representer les etiquettes
y=[]
#A noter qu'une partie des mails a un content nul
for row in reader:
  X.append(row[1]+row[2]+row[3]) #Content (senders, date, content of the email)
  y.append(row[4]) #Receivers

#Suppressions des headers
y.pop(0)
X.pop(0)
#A tfidf representation for text documents is used
tfidf_v = TfidfVectorizer(max_df=0.95, min_df=2, max_features=100000, stop_words='english')
tfidf = tfidf_v.fit_transform(X)

#Selection du classifieur à implementer pour s'entrainer
#clf = AdaBoostClassifier(n_estimators=100)
#clf = RandomForestClassifier(n_estimators=100)
#clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0) # 0.66
#clf = SGDClassifier(loss='modified_huber', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
clf = KNeighborsClassifier(n_neighbors=100)
clf.fit(tfidf, y) #On entraine le classifieur
#scores = cross_validation.cross_val_score(clf, tfidf, y, cv=10)
#scores.mean()

############################################
# Partie 2 : Prediction des destinataires  #
############################################

#Ouverture du fichier de test généré permettant la prediction des etiquettes
f = open('test_email.csv', 'r', encoding = 'utf8')
reader = csv.reader(f, delimiter=',', quotechar='"')
Xtest=[]
for row in reader:
    Xtest.append(row[1]+row[2]+row[3]) #Content
Xtest.pop(0) #Supprime le header

#The same tfidf representation is used
tfidf_test = tfidf_v.transform(Xtest)

#Labels are predicted to the tfidf data
#pred_test=clf.predict(tfidf_test) <---- donne LE destinataire le plus porbable
a=clf.predict_proba(tfidf_test) #Donne le score par rapport aux voisins

labels_receivers = np.unique(y) #Renvoie tous les destinataires possibles
labels_receivers.sort() # On met cette liste dans l'ordre alphabétique car 
                        # le vecteur de score, par rapport au voisins, prends chaque destinataire unique 
                        # dans l'ordre alphabétique

proba_predict = pd.DataFrame(a) #On met les scores obtenus sous forme de dataframe
proba_predict.columns = labels_receivers.tolist() #On met le nom des labels
predList=list() #Liste qui va contenir toutes les adresses mails prédites
for i in range(len(proba_predict)):
    sorted_proba = proba_predict.ix[i].T.sort_values(ascending=False) #On prend les scores pour la ligne i et on met les résultats dans l'ordre décroissant
    credible_receiver = sorted_proba.ix[sorted_proba>0] #On ne garde que les probas supérieures à 0 (au moins 1 des 15 voisins à ce destinataire dans son mail), c'est notre seuil
    credible_receiver = credible_receiver.index.tolist() #On récupère l'addresse mail des élus
    predList.append(credible_receiver) # On ajoute dans a liste contenant les adresses mails prédites
    
#Si plus de 10 élus, on ramène à 10
for i in range(len(predList)):
    if len(predList[i])>10:
        predList[i] = predList[i][0:9]

##########################################################
# Partie 3 : Création du fichier CSV à tester sur Kaggle #
##########################################################

#Ecriture des résultats de prediction dans le csv 
f = open('test_email.csv', 'r', encoding = 'utf8') 
reader = csv.reader(f, delimiter=',', quotechar='"')
liste=list()
cpt = 0
indice=[]
sender=[]

#On crée une liste comprenant les ids dans l'ordre
for row in reader:
    sender.append(row[1])
    indice.append(row[0]) 
indice.pop(0) #Supprime le header
sender.pop(0) #Supprime le header

#On supprime les predictions où les destinataires = émetteurs
for i in range(len(predList)):
    for email in predList[i]:
        if email == sender[i]:
            key = predList[i].index(email)
            del(predList[i][key])
            
#Création d'un tableau, à parcourir par la suite, permettant de créer le fichier avec les adresses predites
for i in range(len(indice)):
    row = indice[i]+","
    #print(row)
    for elem in predList[i]:
        row = row +elem+" "
        #print(row)
    liste.append(row)

#Création du fichier que l'on va tester sur Kaggle 
f = open('pred_test_1.csv', 'w', encoding = 'utf8')
#reader = csv.reader(f, delimiter=',', quotechar='"')
f.write('Id,Recipients\n') # On ajoute les headers
for row in liste:
    f.write(str(row)+"\n") 
    #print(row)
    cpt=cpt+1
