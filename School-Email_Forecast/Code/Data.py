import pandas as pd


######################
#   TRAINING DATA :  #
######################


#----------------------------------------------------------------------------------------------------
# Here we want to add the sender's email address to the file with all email informations.
#----------------------------------------------------------------------------------------------------

# Loading the data from training_set_sid, giving columns names
sender = pd.read_csv('training_set_sid.csv', header = None)
sender.columns = ['send_mail','id']

# Loading the data from training_info_sid, giving columns names
email = pd.read_csv("training_info_sid.csv", header = None)
email.columns = ['id','date','content','rece_mails']

# Adding spaces to match the condition (later)
sender['id'] = (' ' + sender['id'] + ' ')
# Adding a column (empty) for the sender's email address
email.insert(1,'send_mail','')

# Searching for the sender's email, through the id
for i in range(len(email)): # For each mail ...
    id_sender = str(email['id'][i]) # ... we catch the id!
    resu=[]
    for j in range(len(sender)): # For each line in the senders dataset ...
        if(((' ' + id_sender + ' ') in sender['id'][j])): # ... if the id is part of the emails sent by the person of this line ...
            resu = sender['send_mail'][j] # ... we save its email address !
    email['send_mail'][i] = resu # We put the email address in the column !

# Saving the new (all-in-one) file ito .CSV file
email.to_csv('email.csv', sep = ',', encoding = 'utf-8', index = False)




##################
#   TEST DATA :  #
##################


#----------------------------------------------------------------------------------------------------
# Here we want to add the sender's email address on the testing dataset
#----------------------------------------------------------------------------------------------------

# Loading the data from training_set_sid, giving columns names
test_sender = pd.read_csv('test_set_sid.csv', header = None)
test_sender.columns = ['send_mail','id']

# Loading the data from training_info_sid, giving columns names
test_email = pd.read_csv("test_info_sid.csv", header = None)
test_email.columns = ['id','date','content'] # We have to predict the 'rece_mails' column

# Adding spaces to match the condition (later)
test_sender['id'] = (' ' + test_sender['id'] + ' ')
# Adding a column (empty) for the sender's email address
test_email.insert(1,'send_mail','')

# Searching for the sender's email, through the id
for i in range(len(test_email)): # For each mail ...
    id_sender = str(test_email['id'][i]) # ... we catch the id!
    resu=[]
    for j in range(len(test_sender)): # For each line in the senders dataset ...
        if(((' ' + id_sender + ' ') in test_sender['id'][j])): # ... if the id is part of the emails sent by the person of this line ...
            resu = test_sender['send_mail'][j] # ... we save its email address !
    test_email['send_mail'][i] = resu # We put the email address in the column !

# Saving the new (all-in-one) dataframe ito .CSV file
test_email.to_csv('test_email.csv', sep = ',', encoding = 'utf-8', index = False)





