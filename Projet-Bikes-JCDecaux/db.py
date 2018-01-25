from flask import Flask, jsonify
from flask.ext.mysqldb import MySQL

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'admin'
app.config['MYSQL_PORT'] = 5005
app.config['MYSQL_DB'] = 'bikestations'
mysql = MySQL(app)

@app.route('/getall_bike', methods = ['GET'])
def getall_bike():
	cur = mysql.connection.cursor()
	cur.execute('''SELECT * FROM BIKE''')
	returnval = cur.fetchall()
	results = []
	for i in returnval:
		results.append({'bik_ID': i[0],
						'bik_sta_ID': i[1],
						'bik_timestamp': i[2],
						'bik_status': i[3],
						'bik_stands': i[4],
						'bik_available_stands': i[5],
						'bik_available': i[6]})
	return jsonify(jsonify({'results': results}))

@app.route('/getall_station', methods = ['GET'])
def getall_station():
	cur = mysql.connection.cursor()
	cur.execute('''SELECT * FROM STATION''')
	returnvalues = cur.fetchall()
	printthis = ""
	for i in returnvalues:
		printthis += str(i) + "<br>"
	return printthis


@app.route('/addone/<string:insert>')
def add(insert):
	cur = mysql.connection.cursor()
	cur.execute('''SELECT MAX(id) FROM STATION''')
	maxid = cur.fetchone()
	cur.execute('''INSERT INTO BIKE (bik_sta_ID, bik_timestamp, bik_status, bik_stands, bik_available_stands, bik_available) VALUES (%s, %s, %s, %s, %s, %s)''', (maxid))
	mysql.connection.commit()
	return "Done"

if __name__ == '__main__':
    app.run(debug=True)












