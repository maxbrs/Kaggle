from flask import Flask, jsonify, request
from flask.ext.mysqldb import MySQL

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'admin'
app.config['MYSQL_PORT'] = 5005
app.config['MYSQL_DB'] = 'bikestations'
mysql = MySQL(app)

@app.route('/', methods = ['GET'])
@app.route('/tables/', methods = ['GET'])
def index():
	cur = mysql.connection.cursor()
	cur.execute('''SHOW TABLES''')
	val = cur.fetchall()
	results = []
	for i in val:
		results.append(i)
	return jsonify({'tables': results})

@app.route('/getall_bike/', methods = ['GET'])
def getall_bike():
	cur = mysql.connection.cursor()
	cur.execute('''SELECT * FROM BIKE''')
	val = cur.fetchall()
	results = []
	for i in val:
		results.append({'bik_ID': i[0],
						'bik_sta_ID': i[1],
						'bik_timestamp': i[2],
						'bik_status': i[3],
						'bik_stands': i[4],
						'bik_available_stands': i[5],
						'bik_available': i[6]})
	return jsonify({'bike': results})

@app.route('/getall_station/', methods = ['GET'])
def getall_station():
	cur = mysql.connection.cursor()
	cur.execute('''SELECT * FROM STATION''')
	val = cur.fetchall()
	results = []
	for i in val:
		results.append({'sta_ID': i[0],
						'sta_number': i[1],
						'sta_lat': i[2],
						'sta_lon': i[3],
						'sta_name': i[4],
						'sta_city': i[5],
						'sta_address': i[6],
						'sta_payment': i[7],
						'sta_bonus': i[8],})
	return jsonify({'station': results})


@app.route('/post_station/', methods = ['POST'])
def add_station():
	data = request.get_json()
	for each in data:
		name = each['name']
		lat = each['position']['lat']
		lon = each['position']['lng']
		address = each['address']
		available_bike_stands = each['available_bike_stands']
		available_bikes = each['available_bikes']
		banking = each['banking']
		bonus = each['bonus']
		bike_stands = each['bike_stands']
		city = each['contract_name']
		number = each['number']
		status = each['status']
		time = datetime.fromtimestamp(int(str(each['last_update'])[:-3])).strftime('%Y-%m-%d %H:%M:%S')
		#try : 
		call_bikestation(name, lat, lon, address, available_bike_stands, available_bikes, banking, bonus, bike_stands, city, number, status, time)
		#except: 
		#	print('Unable to insert word')


def call_bikestation(name, lat, lon, address, available_bike_stands, available_bikes, banking, bonus, bike_stands, city, number, status, time):
    	query = "CALL ADD_BIKE_STATION("+str(name)+"," + int(lat)+","+int(lon)+","+str(address)+","+int(available_bike_stands)+","+int(available_bikes)+","+bool(banking)+","+bool(bonus)+","+int(bike_stands)+","+str(city)+","+int(number)+","+str(status)+","+str(time)+");"  
	#cursor = db.cursor()
	cursor.execute(query)




# @app.route('/addone/<string:insert>')
# def add(insert):
# 	cur = mysql.connection.cursor()
# 	cur.execute('''SELECT MAX(id) FROM STATION''')
# 	maxid = cur.fetchone()
# 	cur.execute('''INSERT INTO BIKE (bik_sta_ID, bik_timestamp, bik_status, bik_stands, bik_available_stands, bik_available) VALUES (%s, %s, %s, %s, %s, %s)''', (maxid))
# 	mysql.connection.commit()
# 	return "Done"

if __name__ == '__main__':
    app.run(debug=True)












