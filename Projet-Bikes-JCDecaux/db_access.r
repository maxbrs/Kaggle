debut <- Sys.time()

library(RMySQL)

setwd("C:/Users/mbriens/Documents/Kaggle/GIT/Kaggle/Projet-Bikes-JCDecaux")

#----------------------------------------

############################
#                          #
#   JCDECAUX : BIKES APP   #
#                          #
############################

# Cf. https://www.r-bloggers.com/accessing-mysql-through-r/



#----------
# Connecting database :
#----------

user = 'root'
password = 'admin'
database_name = 'bikestations'
host = 'localhost'
port = 5005

mydb = dbConnect(MySQL(), user=user, password=password, dbname=database_name, host=host, port=port)

# Show tables in database
dbListTables(mydb)

# Show attributes in 'bike'
dbListFields(mydb, 'bike')

# Extract data from SQL query
query = dbSendQuery(mydb, "SELECT count(bik_id) as nb_records FROM bike")
data = fetch(query, n=-1)
print(data)










