import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
#from collections import Counter, defaultdict



###########################
#   CREATING A NETWORK :  #
###########################


#----------------------------------------------------------------------------------------------------
# Here we want to create a file, in order to create a GML file from it
#----------------------------------------------------------------------------------------------------

email = pd.read_csv('email.csv') # Reading the csv file

# WEIGHT :
Num_Rece = [] # It will contain how many receivers there are for each email
for i in range(len(email)): # For each row from email ...
    item = email['rece_mails'][i] # ... we will look at cell that contains the receivers ...
    AsList = item.split() # ... and we will convert it from string (separated with spaces) to a proper list ...
    Num_Rece.append(len(AsList)) # ... and we are looking at the size of this list to know the number of receivers for this email.

weight = '' # We may consider that sending one mail to several persons is not srtonger than sending one mail to one person.
for i in range(len(Num_Rece)): # For each row from email ...
    resu = list(np.repeat(str(1/Num_Rece[i]), Num_Rece[i])) # ... we give a weight, according to the number of receivers of each email ...
    weight = weight + ' ' + ' '.join(resu) # ... and we store it !
weight = weight.split() # We change the string into a proper list ...
for i in range(len(weight)): # ... and for each items of that list ...
    weight[i] = float(weight[i]) # ... we convert the content from string to float (since it is a weight, from 0 to 1).

# RECE_MAIL :
rece_mail = [] # We want to 'unlist' the 'rece_mails' column (due to several receivers for one email)
rece_mails = list(email['rece_mails']) # We convert it first from string to list (of lists) ...
for i in range(len(rece_mails)): # ... and, for each email (items of the list), ...
    rece_mail = rece_mail + rece_mails[i].split( ) # ... we unlist and concatenate everything instead : we get a simple (non-nested) list.

# AND THE OTHERS : ID, SEND_MAIL, DATE, CONTENT
dupl_email = email.drop('rece_mails', 1) # We delete the 'rece_mails' column from the dataframe (already used, and changed into 'rece_mail')
for i in range(len(dupl_email)): # For each row from email ...
    line = dupl_email.iloc[i, :] # ... we look at the whole row ...
    count = Num_Rece[i] # ... and we count the number of receivers of the email.
    while count != 1: # According to the number ...
        dupl_email = dupl_email.append(line, ignore_index = True) # We duplicate the line ...
        count = count-1 # ... the appropriate number of times (we want to have one receiver on each line)
dupl_email = dupl_email.sort(columns = ['id'], ascending = [1])

# We can check that all the 3 things we just created are ready to be joined (same length) :
print('length of \'weight\' : ' + str(len(weight))) # The number of email addresses used to receive, for each (original) email
print('length of \'rece_mail\' : ' + str(len(rece_mail))) # All the email adresses used to receive, in one list (one email per item)
print('length of \'id\', \'send_mail\', \'date\' & \'content\' : ' + str(len(dupl_email))) # All the other columns : 'id', 'send_mail', 'date', 'content'

dupl_email = dupl_email.assign(rece_mail = rece_mail, weight = weight) # Joining the data into one dataframe
dupl_email.to_csv('dupl_email.csv', sep = ',', encoding = 'utf-8', index = False) # Saving it as a CSV file


                 
dupl_email = pd.read_csv('dupl_email.csv') # Reading the CSV file

DFgraph = pd.DataFrame(columns = ['From', 'To', 'NbMails', 'Weight']) # This is the final dataframe we want

From = dupl_email.loc[0, 'send_mail'] # The first sender
To = dupl_email.loc[0, 'rece_mail'] # The first receiver

# Adding the first line from 'dupl_email' to 'DFgraph' (which, of course, do not exists already, since DFgraph is empty)
DFgraph.loc[0, 'NbMails'] = 1
DFgraph.loc[0, 'Weight'] = dupl_email['weight'][0]
DFgraph.loc[0, 'From'] = str(From)
DFgraph.loc[0, 'To'] = str(To)
row_new = len(DFgraph)

for row in range(1, len(dupl_email)): # for each row in email ('dupl_email') ...
    if row%25 == 0: # The loop needs time to run, so ...
        print(str(round(((row/len(dupl_email))*100), 2)) + ' %') # ... we look at how much we've done, printing the progress (in %) !
    From = dupl_email['send_mail'][row] # We look at the sender (From)
    To = dupl_email['rece_mail'][row] # We look at the receiver (To)
    
    existing_row = [] # We reset the 'existing_row' var
    if From in set(DFgraph['From']) and To in set(DFgraph['To']): # if From and To are both already into our new dataframe, ...
        for i in range(len(DFgraph)): # ... for each row of the new dataframe ...
            if ((From == DFgraph.loc[i, 'From']) and (To == DFgraph.loc[i, 'To'])): # ... we will check if they both exists on the same line, ...
                existing_row = i # ... and if they are truly on the same row, then the combination (From, To) alrady exists, ...
                DFgraph.loc[existing_row, 'NbMails'] = DFgraph.loc[existing_row, 'NbMails'] + 1 # ... so we increment the 'NbMails' var, ...
                DFgraph.loc[existing_row, 'Weight'] = DFgraph.loc[existing_row, 'Weight'] + dupl_email['weight'][row] # ... and add the new weight to the previous one.
            
    if existing_row == [] : # If all conditions above has not been satisfied, the 'existing_row' var is still empty, and then ...
    	# ... it means that we can create the new line in the new dataframe :
        DFgraph.loc[row_new, 'NbMails'] = 1
        DFgraph.loc[row_new, 'Weight'] = dupl_email['weight'][row]
        DFgraph.loc[row_new, 'From'] = str(From)
        DFgraph.loc[row_new, 'To'] = str(To)
        row_new = len(DFgraph)

#G1 = nx.from_pandas_dataframe(dupl_email, 'send_mail', 'rece_mail', edge_attr=['id','date','content','weight'], create_using=nx.DiGraph())
#len(DFgraph) == G1.number_of_edges()

# Saving the new (ready to use as network) dataframe ito .CSV file
graph_email = DFgraph
graph_email.to_csv('graph_email.csv', sep = ',', encoding = 'utf-8', index = False) 




#######################
#   NETWORK MINING :  #
#######################


#----------------------------------------------------------------------------------------------------
# Here we want to look at our network
#----------------------------------------------------------------------------------------------------

graph_email = pd.read_csv('graph_email.csv') # Reading the CSV file

# Create the network from the dataframe
OG = nx.from_pandas_dataframe(graph_email, 'From', 'To', edge_attr = ['NbMails','Weight'], create_using = nx.DiGraph())
nx.write_gml(OG, 'Oriented_Graph.gml', stringizer = None)

# Looking at the content of the graph file :
print('Number of nodes = ' + str(OG.number_of_nodes()))
print('Number of edges = ' + str(OG.number_of_edges()))
print('NbMails, FROM : jake.sullivan@... ; TO : hillary.clinton@... = ' + str(OG['jake.sullivan@univ-tlse3.fr']['hillary.clinton@univ-tlse3.fr']['NbMails']))


#----------------------------------------------------------------------------------------------------
# Here we want to realize an analysis based on the connectivity (flow-based connectivity)
#----------------------------------------------------------------------------------------------------
# Cf. : https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.connectivity.connectivity.node_connectivity.html#networkx.algorithms.connectivity.connectivity.node_connectivity

conn = nx.node_connectivity(OG, s = 'jake.sullivan@univ-tlse3.fr', t = 'hillary.clinton@univ-tlse3.fr')

C = pd.DataFrame(columns = ['Source', 'Target', 'Connectivity']) # This is the final dataframe we want
row = len(C)
Size = len(OG.nodes())**2
for source in OG.nodes():
    for target in OG.nodes():
        if row%100 == 0: # The loop needs time to run, so ...
            print(str(round(((row/Size)*100), 2)) + ' %') # ... we look at how much we've done, printing the progress (in %) !
        connec = nx.node_connectivity(OG, s = source, t = target)
        C.loc[row, 'Source'] = source
        C.loc[row, 'Target'] = target
        C.loc[row, 'Connectivity'] = connec
        row = len(C)

graph_connectivity = C
graph_connectivity.to_csv('graph_connectivity.csv', sep = ',', encoding = 'utf-8', index = False) 

#--------------------------------------------------

graph_connectivity = pd.read_csv('graph_connectivity.csv') # Reading the CSV file

# Deleting the lower scores (since the dataframe is too big, and the program is too long to run)
#graph_connectivity = graph_connectivity[graph_connectivity.Connectivity != 0]
#graph_connectivity = graph_connectivity[graph_connectivity.Connectivity != 1]
graph_connectivity = graph_connectivity[graph_connectivity.Connectivity > 1]

graph_connectivity.to_csv('gc.csv', sep = ',', encoding = 'utf-8', index = False)
graph_connectivity = pd.read_csv('gc.csv') # Reading the CSV file

test_email = pd.read_csv('test_email.csv') # Reading the CSV file
test_email.insert(4, 'predict_rece_mail','')

Size = len(test_email['send_mail'])
for email in range(len(test_email['send_mail'])):
    if email%5 == 0: # The loop needs time to run, so ...
            print(str(round(((email/Size)*100), 2)) + ' %') # ... we look at how much we've done, printing the progress (in %) !

    sender = test_email.loc[email, 'send_mail']
    #id_mail = test_email.loc[email, 'id']
	
    C_ForSender = pd.DataFrame(columns = ['Source', 'Target', 'Connectivity']) # This is the final dataframe we want

    for row in range(len(graph_connectivity['Source'])):
        if sender == graph_connectivity.loc[row, 'Source']:
            line = graph_connectivity.iloc[row, :] # ... we look at the whole row ...
            C_ForSender = C_ForSender.append(line, ignore_index = True) # We duplicate the line ...

    C_ForSender = C_ForSender.sort(columns = ['Connectivity'], ascending = [0])
    resu = C_ForSender['Target'][0:10]
    resu = resu.tolist()
    resu = ' '.join(resu)
    test_email['predict_rece_mail'][email] = resu # We put the email addresses in the column !


test_id = test_email['id']
test_rece = test_email['predict_rece_mail']

d = {'Id' : test_id,
     'Recipients' : test_rece}
df = pd.DataFrame(d)

df.to_csv('resu_nx_connec.csv', sep = ',', encoding = 'utf-8', index = False)


#----------------------------------------------------------------------------------------------------
# Here we want to realize an analysis based on the neighborhood (average degree of the neighborhood of each node)
#----------------------------------------------------------------------------------------------------
# Cf. : https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.assortativity.average_neighbor_degree.html#networkx.algorithms.assortativity.average_neighbor_degree

neighborhood = nx.average_neighbor_degree(OG, source = 'jake.sullivan@univ-tlse3.fr', target = 'hillary.clinton@univ-tlse3.fr', weight = 'NbMails')

N = pd.DataFrame(columns = ['Source', 'Target', 'Neighborhood']) # This is the final dataframe we want
row = len(N)
Size = len(OG.nodes())**2
for source in OG.nodes():
    for target in OG.nodes():
        if row%100 == 0: # The loop needs time to run, so ...
            print(str(round(((row/Size)*100), 2)) + ' %') # ... we look at how much we've done, printing the progress (in %) !
        connec = nx.average_neighbor_degree(OG, s = source, t = target,  weight = 'NbMails') # Try with weight = 'Weight' aswell
        N.loc[row, 'Source'] = source
        N.loc[row, 'Target'] = target
        N.loc[row, 'Neighborhood'] = neighborhood
        row = len(N)

graph_neighborhood = N
graph_neighborhood.to_csv('graph_neighborhood.csv', sep = ',', encoding = 'utf-8', index = False) 

#--------------------------------------------------

graph_neighborhood = pd.read_csv('graph_neighborhood.csv') # Reading the CSV file

# Deleting 0s (since the dataframe if too big, and the program is too long to run)
graph_neighborhood = graph_neighborhood[graph_neighborhood.Connectivity > 0.5]

graph_neighborhood.to_csv('gc.csv', sep = ',', encoding = 'utf-8', index = False)
graph_neighborhood = pd.read_csv('gc.csv') # Reading the CSV file

test_email = pd.read_csv('test_email.csv') # Reading the CSV file
test_email.insert(4, 'predict_rece_mail','')

Size = len(test_email['send_mail'])
for email in range(len(test_email['send_mail'])):
    if email%5 == 0: # The loop needs time to run, so ...
            print(str(round(((email/Size)*100), 2)) + ' %') # ... we look at how much we've done, printing the progress (in %) !

    sender = test_email.loc[email, 'send_mail']
    #id_mail = test_email.loc[email, 'id']
	
    C_ForSender = pd.DataFrame(columns = ['Source', 'Target', 'Connectivity']) # This is the final dataframe we want

    for row in range(len(graph_neighborhood['Source'])):
        if sender == graph_neighborhood.loc[row, 'Source']:
            line = graph_neighborhood.iloc[row, :] # ... we look at the whole row ...
            C_ForSender = C_ForSender.append(line, ignore_index = True) # We duplicate the line ...

    C_ForSender = C_ForSender.sort(columns = ['Connectivity'], ascending = [0])
    resu = C_ForSender['Target'][0:10]
    resu = resu.tolist()
    resu = ' '.join(resu)
    test_email['predict_rece_mail'][email] = resu # We put the email addresses in the column !


test_id = test_email['id']
test_rece = test_email['predict_rece_mail']

d = {'Id' : test_id,
     'Recipients' : test_rece}
df = pd.DataFrame(d)

df.to_csv('resu_nx_neighbor.csv', sep = ',', encoding = 'utf-8', index = False)










#----------------------------------------------------------------------------------------------------
# Here we want to realize an analysis based on pagerank
#----------------------------------------------------------------------------------------------------

pagerank = nx.pagerank(OG, alpha=0.85, weight='NBMails') # Try with weight = 'Weight' aswell

pagerank_list = {node: rank for node, rank in pagerank.items()}
nx.set_node_attributes(OG, 'pagerank', pagerank_list)
positions=nx.spring_layout(OG)
nodesize = [x['pagerank']*50000 for v, x in OG.nodes(data = True)]
edgesize = [np.sqrt(e[2]['NbMails']) for e in OG.edges(data = True)]

pylab.rcParams['figure.figsize'] = 16, 16

nx.draw_networkx_nodes(OG, positions, node_size = nodesize, alpha = 0.4)
nx.draw_networkx_edges(OG, positions, edge_size = edgesize, alpha = 0.2)
nx.draw_networkx_labels(OG, positions, font_size = 10)

#plt.savefig("plot_pagerank.png")
plt.title("Graph of all send / receive relationships in the Clinton email database", fontsize = 20)
plt.clf()
#That graph is pretty big. Let's make a smaller one with just the most important people.  
#This will plot only the nodes with pagerank greater than pagerank_cutoff

pagerank_cutoff = 0.0045

small_OG = OG.copy()
for n, p_rank in small_OG.nodes(data=True):
    if p_rank['pagerank'] < pagerank_cutoff: small_OG.remove_node(n)
    
spositions = nx.spring_layout(small_OG, weight = None)
snodesize = [x['pagerank']*50000 for v, x in small_OG.nodes(data = True)]
sedgesize = [np.log(e[2]['NbMails']) for e in small_OG.edges(data = True)]
scolors = np.random.rand(len(small_OG.nodes()))

nx.draw_networkx_nodes(small_OG, spositions, node_size = snodesize, node_color = scolors, alpha = 0.3)
nx.draw_networkx_edges(small_OG, spositions, alpha = 0.3, arrows = False) #, width=sedgesize)
nx.draw_networkx_labels(small_OG, spositions, font_size = 14)
plt.title("Graph of only those people with a pagerank of greater than %s" % pagerank_cutoff, fontsize=20)
#plt.savefig("plot_pagerank_small.png")








