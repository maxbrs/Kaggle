import pandas as pd
import pylab as plt


###############################
#   DESCRIPTIVE STATISTICS :  #
###############################


#----------------------------------------------------------------------------------------------------
# Here we want to count how many how many emails has been sent or received per person.
#----------------------------------------------------------------------------------------------------

# Functions :

def count_times(col, number): # Count how many times each email has been used (to send or to receive)
    times = {}
    k = 0
    for item in col: # For each email address ...
        k = k+1
        if item:
            if item not in times.keys(): # ... if it is a new address
                times[item] = 1 # ... we add a new item (email address), with 1 count!
            else: # ... if it is not a new address ...
                times[item] = times[item]+1 # ... we count +1 for the existing item (email address)!
    email = []
    count = []
    for i in times.keys(): # For each email address in 'times' ...
        if times[i] > number: # ... if it is upper than the step ...
            email.append(i) # ... we store it ...
            count.append(times[i]) # ... and look at the times it has been used
    return (email, count)

def make_hist(count, email, title):
    c = zip(count, email)
    d = sorted(c) # We sort data according to count !
    x = []
    y = []
    for i in d: # For each person ...
        x.append(i[1]) # ... X axis : email address
        y.append(i[0]) # ... Y axis : times it has been used

    plt.bar(range(len(x)),y) # Plotting the bar plot
    plt.xticks(range(len(x)), x,rotation=85) # Adding the legend (email address)
    plt.ylabel('Number of emails')
    plt.title(title)
    plt.show()

#--------------------------------------------------
# Main :

email = pd.read_csv('email.csv') # Reading the csv file

# We want to 'unlist' the 'rece_mails' column (due to several receivers for one email)
rece_mails_list = []
rece_mails = list(email['rece_mails']) # From string to list (of lists)
for i in range(len(rece_mails)): # For each email (items of the list) ...
    rece_mails_list = rece_mails_list + rece_mails[i].split( ) # .. we unlist and concatenate everything instead !

From = email['send_mail'] # All the email adresses used to SEND
To = rece_mails_list # All the email adresses used to RECEIVED


# Plotting barplot of the email addresses used to send
From_mail, From_count = count_times(From, 20)
make_hist(From_count, From_mail, 'Email addresses used to send mails')

# Plotting barplot of the email addresses used to receive
To_mail, To_count = count_times(To, 20)
make_hist(To_count, To_mail, 'Email addresses used to receive mails')


# Used to predict :
To_mail, To_count = count_times(To, 1)
d = sorted(zip(To_count, To_mail), reverse = True)
x = []
for i in d: # For each person ...
        x.append(i[1])
x[0:10]




######################################
#   CREATING THE PREDICTED EMAILS :  #
######################################


#----------------------------------------------------------------------------------------------------
# Here we want to create the csv file for submitting through Kaggle, based on the frequence.
#----------------------------------------------------------------------------------------------------

test_email = pd.read_csv('test_email.csv') # Reading the csv file
test_email.insert(4, 'predict_rece_mail','')

predicted_emails = x[0:10] # Selecting the 10 most frequent emails
predicted_emails = (' '.join(predicted_emails)) # Unlisting the predicted emails


for i in range(len(test_email)):
    test_email['predict_rece_mail'][i] = predicted_emails # We put the email addresses in the column !


test_id = test_email['id']
test_rece = test_email['predict_rece_mail']

d = {'Id' : test_id,
     'Recipients' : test_rece}
df = pd.DataFrame(d)

df.to_csv('resu_freq.csv', sep = ',', encoding = 'utf-8', index = False)





