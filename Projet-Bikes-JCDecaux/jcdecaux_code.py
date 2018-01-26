
import config
import numpy as np
import requests
import json
from datetime import datetime
import time
import plotly.plotly as py
from plotly.graph_objs import *


laps = 10
length = 60
for i in range(int(length/laps)):
    print(str(i*laps) + ' min since START')
    key = config.jcdecaux_key
    response = []
    for city in ('Toulouse', 'Dublin', 'Lyon', 'Nantes', 'Marseille', 'Stockholm', 'Luxembourg'):
        print(city)
        get_url = 'https://api.jcdecaux.com/vls/v1/stations?contract='+city+'&apiKey='+key
        try:
            r = requests.get(get_url)
            print(r.status_code)
            data = r.json()
            data = json.dumps(data, ensure_ascii='False')
            url_POS = 'http://localhost:5000/post_bikedata/'
            headers = {'Content-Type': 'application/json'}
            response.append(requests.post(url_POS, headers=headers, json=data))
        except:
            print('JCDecaux API unreachable')
    time.sleep(60*laps)














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
        accesstoken=mapbox_access_token,
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







