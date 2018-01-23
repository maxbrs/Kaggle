
import config
import numpy as np
import requests
import json
from datetime import datetime
import plotly.plotly as py
from plotly.graph_objs import *


key = config.jcdecaux_key
city = 'Toulouse'
# Dublin, Lyon, Nantes, Marseille, Stockholm, Luxembourg
get_url = 'https://api.jcdecaux.com/vls/v1/stations?contract='+city+'&apiKey='+key

r = requests.get(get_url)
print(r.status_code)
#print(r.json())

data = r.json()


name = []
lat = []
lon = []
address = []
available_bike_stands = []
available_bikes = []
banking = []
bike_stands = []
city = []
number = []
status = []
time = []
for station in data:
    name.append(station['name'])
    lat.append(station['position']['lat'])
    lon.append(station['position']['lng'])
    address.append(station['address'])
    available_bike_stands.append(station['available_bike_stands'])
    available_bikes.append(station['available_bikes'])
    banking.append(station['banking'])
    bike_stands.append(station['bike_stands'])
    city.append(station['contract_name'])
    number.append(station['number'])
    status.append(station['status'])
    time.append(datetime.fromtimestamp(int(str(station['last_update'])[:-3])).strftime('%Y-%m-%d %H:%M:%S'))






mapbox_access_token = config.mapbox_key
data = Data([
    Scattermapbox(
        lat=lat,
        lon=lon,
        mode='markers',
        marker=Marker(
            size=9
        ),
        text=name,
    )
])

layout = Layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
 #       accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=np.mean(lat),
            lon=-np.mean(lon)
        ),
        pitch=0,
        zoom=10
    ),
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='Multiple Mapbox')


# Cf. https://plot.ly/~maxbriens/0/








