##### GROUP 3 - Version 1.0 #####
from flask import Flask, request, jsonify
import json
import requests
from flask_restful import Resource, Api
import mysql.connector
from flask_mysqldb import MySQL
import MySQLdb


app = Flask(__name__) # we are using this variable to use the flask microframework
api = Api(app)


@app.route("/filtering/", strict_slashes = False, methods = ['POST'])
def filtering():
	data = request.get_json()
	for article in data :
		date_publication = article['article']['date_publication']
		list_surname_author = article['article']['surname_author']
		name_newspaper = article['article']['name_newspaper']
		id_article = article['id_art']

		try :
			call_pfiltering_article(date_publication, name_newspaper, id_article)
		except : 
			print('Unable to insert article')

		for surname_author in list_surname_author:
			#try : 
			call_pauthor(id_article, surname_author)
			#except : 
			#	print('Author alerady exist.')

		for position_word in article['position_word'] : 
			lemma = position_word['lemma']
			type_entity = position_word['type_entity']
			pos_tag = position_word['pos_tag']
			title = position_word['title']
			position = position_word['position']
			word = position_word['word']
			
			#try : 
			call_pfiltering_position_word(position, word, lemma, title, pos_tag, type_entity, id_article)
			#except: 
			#	print('Unable to insert word')
			
	
	result = json.dumps([[{"message":{"id_article" : id_article}}]])
	query = "COMMIT ;"
	cursor.execute(query)

	return result



def call_pfiltering_article(date_publication, name_newspaper, id_article):
	query = "CALL FILTERING_PARTICLE('" + date_publication + "','" + name_newspaper + "','" + id_article + "');"
	#cursor = db.cursor()
	cursor.execute(query)


def call_pfiltering_position_word(position, word, lemma, title, pos_tag, type_entity, id_article):
	query = "CALL FILTERING_PPOSITION_WORD(" + str(position) + ",'" + word + "','" + lemma + "',"  + str(title) + ",'" + pos_tag + "','" + type_entity  + "'," + str(id_article) + ");"
	#cursor = db.cursor()
	cursor.execute(query)

def call_pauthor(id_article, surname_author):
	query = "CALL FILTERING_PAUTHOR(" + str(id_article) + ",'" + surname_author + "');"
	#cursor = db.cursor()
	cursor.execute(query)


if __name__ == '__main__':

	# MySQL configurations
	#servername = "localhost"
	#username = "DBIndex_user"
	#passwordDB = "password_DBIndex_user"
	#databasename = "DBIndex"
	servername = "localhost"
	username = "root"
	passwordDB = "admin"
	databasename = "DBIndex"

	db = MySQLdb.connect(user = username, passwd = passwordDB, host = servername, db = databasename)
	cursor = db.cursor()
	#app.run(host="localhost", port = 5005, debug = True)
	app.run(host="localhost", port=5005, threaded=True, debug=True)