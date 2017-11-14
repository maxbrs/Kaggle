
# SID - PROJET : PREDICTION DE DESTINATAIRES D'EMAILS
* M. BRIENS
* M. CHAOUNI
* A. PLISSONNEAU

### CONTENU DE L'ARCHIVE :


* Code/
	* Data.py
	* StatDesc.py
	* Network.py
	* Knn.py
	* NetworkPredict.py
	* SVM.py

* Data/
	* Raw/
		* training_set_sid.csv
		* training_info_sid.csv
		* test_set_sid.csv
		* test_info_sid.csv
		* test_freq_sid.csv
		* test_random_sid.csv
	* email.csv (créé par 'Data.py')
	* test_email.csv (créé par 'Data.py')
	* dupl_email.csv (créé par 'Network.py')
	* graph_email.csv (créé par 'Network.py')
	* Oriented_Graph.gml (créé par 'Network.py')
	* graph_connectivity.csv (créé par 'Network.py')

* Graph/
	* common_senders.png (créé par 'StatDesc.py')
	* common_receivers.png (créé par 'StatDesc.py')
	* plot_pagerank.png (créé par 'Network.py')
	* plot_pagerank_small.png (créé par 'Network.py')

* Submissions/
	* resu_freq.csv (créé par 'StatDesc.py')
	* resu_nx_connec.csv (créé par 'Network.py')
	* resu_text_knn.csv
	* resu_text1.csv

### COMMENT PROCÉDER :

1. Tout d'abord, exécutez la totalité du code Data.py (Code/Data.py).
	
	- Il permet, à partir des fichiers 'training_set_sid.csv' et 'training_info_sid.csv', de créer un fichier unique contenant les emails des émetteurs, et non plus des identifiants. Le résultat obtenu est stocké dans le fichier 'email.csv' email.csv.

	- Il permet également de faire de même à partir des fichiers 'test_set_sid.csv' et 'test_info_sid.csv', correspondant aux jeux de tests. Le résultat obtenu est stocké dans le fichier 'test_email.csv' (Data/test_email.csv).

2. Ensuite, exécutez le code StatDesc.py (Code/StatDesc.py).

	- Ce code a pour but de donner un premier résultat de prédiction à tester. Le but ici est simplement de chercher quels sont les 10 individus (identifiés par leur adresse mail) qui reçoivent le plus de mails. L'idée étant, quelque soit le mail du jeu de test, de lui prédire comme destinataire, dans l'ordre, les 10 destinataires les plus fréquents.
	- Ce code s'appuye sur les fichiers .csv précedemment créés, à savoir 'email.csv' (Data/email.csv), et 'test_email.csv' (Data/test_email.csv)
	- Ce résultat est illustré par le graphique 'common_receivers.png' (Graph/common_receivers.png).
	- La dernière partie du code a pour but de créer le fichier csv pouvant être soumis sur Kaggle, à savoir, 'resu_freq.csv' (Submissions/resu_freq.csv). Son score issu de Kaggle est 0.79423.

3. A présent, exécutez le code Network.py (Code/Network.py).

	**Creating a network** :

	Ici, l'objectif est d'effectuer des analyses de network mining. Pour y parvenir, il faut transformer notre fichier 'email.csv' (Data/email.csv), en un fichier à partir duquel il sera plus aisé d'effectuer des analyses relevant du network mining : un fichier .gml. C'est le but de la première partie de ce code (intitulée 'creating a network').
	- Avant de créer un fichier .gml, il est plus pratique de créer un fichier dont chaque ligne ne dispose que d'un unique destinataire : il faut dupliquer les mails comprennant plusieurs destinataires. Pour ce faire, on s'appuye sur le fichier 'email.csv' (Data/email.csv). On créé ainsi le fichier 'dupl_email.csv' (Data/dupl_email.csv), sur lequel la suite du code s'appuie.
	- Ce fichier nouvellement créé ne suffit pas tout à fait à la création du fichier .gml. Il est vital de donner plus d'importance lorsqu'un individu a envoyé plusieurs mails à un autre individu, que s'ils ne s'en sont envoyés qu'un. La suite du code permet d'attribuer un poids ('NbMails') à notre graphe, ainsi qu'un autre poids pondéré ('Weight'), calculé en partant du principe qu'un mail recu par un individu, qui a été envoyé à plusieurs autres personnes (en copie), n'ait pas la même importance qu'un mail ayant été envoyé uniquement à un individu. Lorsque l'option se présentera de choisir la variable de poids dans les calculs de network mining, il sera ainsi possible de tester ces deux variables. Une fois cette partie du code exécutée, on créé le fichier 'graph_email.csv' (Data/graph_email.csv).

	**Network mining** :

	A présent nous pouvons utiliser le fichier .gml.
	- Tout d'abord, il faut convertir le dataframe (fichier .csv) en un fichier .gml : 'Oriented_Graph.gml' (Data/Oriented_Graph.gml).
	- On s'intéresse à la connectivité entre les noeuds (individus) du graphe. Pour ce faire, il existe une fonction 'nx.node_connectivity', qui, pour un noeud source et un noeud cible, calcule un entier. Pour commencer, il faut calculer la connectivité entre chacun des noeuds. Ce résultat est stocké dans le fichier 'graph_connectivity.csv' (Data/graph_connectivity.csv). L'exécution de cette portion du code est assez lourde et longue (la progression s'affiche en %).
	- A partir du fichier 'graph_connectivity.csv' (Data/graph_connectivity.csv), il faut, pour chaque emmeteur du fichier test_email.csv (Data/test_email.csv), connaître les destinataires correspondant aux 10 connectivités les plus fortes. L'exécution de ce code est, là encore, particulièrement longue (la progression s'affiche en %). L'exécution de ce code donne un nouveau résultat à tester via Kaggle, à savoir, 'resu_nx_connec.csv' (Submissions/resu_nx_connec.csv). Le score obtenu est de 0.76941.

	- Ensuite, on s'intéresse au voisinnage (neighborhood), en utilisant la fonction 'nx.average_neighbor_degree', qui, pour un noeud source, un noeud cible et une variable de pondération (soit 'NbMails', soit 'Weight'), calcule une valeur relative au voisinnage entre ces noeuds. L'exécution de ce code nous donne un résultat 'resu_nx_neighbor.csv'.
	NB : Ayant un problème vis-à-vis de l'utilisation de la fonction (average_neighbor_degree), et n'ayant pas suffisemment de temps à accorder à cette partie du code, nous ne disposons pas de ce résultat.

	- Finalement, on souaite réaliser une analyse basée sur pagerank, qui est utilisée dans le web pour mesurer quantitativement la popularité d'une page web en analysant des liens. Pour ce faire, il existe la fonction 'nx.pagerank', qui, pour un graphe donné, un paramètre alpha, et une variable de pondération (soit 'NbMails', soit 'Weight'), calcule le pagerank. Cette fonction nous permet également d'obtenir des graphiques illustrants le résultat de l'analyse pagerank, l'un en complet, et l'autre de manière plus simplifiée, avec les éléments les plus importants uniquement. Ces graphiques sont 'plot_pagerank.png' (Graph/plot_pagerank.png), et 'plot_pagerank_small.png' (Graph/plot_pagerank_small.png).
	NB : Ayant un problème vis-à-vis de l'utilisation de la fonction (nx.pagerank), et n'ayant pas suffisemment de temps à accorder à cette partie du code, nous ne disposons pas de ce résultat.


4. Maintenant, exécutez le code 'TextMining.py' (Code/TextMining_projet.py).

    **TextMining** :

    Ici, l'objectif est d'effectuer des analyses de text mining. Pour cela, nous allons nous interesser aux différentes parties du code :
	 - La partie 1 va nous permettre de nous "entrainer" afin de pourvoir par la suite prédire les destinataires des emails. Pour cela, nous avons décidé d'implementer plusieurs fonctions afin de pouvoir comparer les scores obtenus par la suite. Nous avons focalisé nos démarches sur les méthodes SGDClassifier, AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, KNeighborsClassifier. D'autres fonctions ont été implementées (telle que OneVsRest ou encore LinearSVC), mais n'ont pas été retenues. On obtient le meilleur score sur kaggle suivant , avec la méthode KNeighbors avec comme parametre n_neighbors = 150: 0.86513.

    - La partie 2 nous permet de prédire les destinataires avec la méthode de classification choisie. On ne garde au plus que 10 destinataires et on les stocke dans une liste.
	
    - La partie 3 nous permet de retirer les emetteurs s'envoyant des mails et ensuite de créer le fichier CSV que l'on soumet par la suite sur Kaggle





