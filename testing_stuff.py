import requests, streamlit as st

def get_current_weather(location="Syracuse"):
    url = f'https://wttr.in/{location}?format=j1'
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise Exception(f'wttr.in error: status {response.status_code}')
    try:
        data = response.json()
    except ValueError:
        raise Exception(f'Could not find a location named {location}')

    current = data['current_condition'][0]
    today = data['weather'][0]
    area = data['nearest_area'][0]

    # hourly entries are every 3 hours: index 3 = 9am, 5 = 3pm, 6 = 6pm
    def snapshot(i):
        h = today['hourly'][i]
        return {'temp_F': float(h['tempF']),
                'feels_like_F': float(h['FeelsLikeF']),
                'description': h['weatherDesc'][0]['value'],
                'chance_of_rain': int(h['chanceofrain'])}

    print({
        'location': f"{area['areaName'][0]['value']}, {area['region'][0]['value']}",
        'temperature_F': float(current['temp_F']),
        'feels_like_F': float(current['FeelsLikeF']),
        'description': current['weatherDesc'][0]['value'],
        'humidity': int(current['humidity']),
        'wind_mph': int(current['windspeedMiles']),
        'uv_index': int(current['uvIndex']),
        'high_F': float(today['maxtempF']),
        'low_F': float(today['mintempF']),
        'max_chance_of_rain': max(int(h['chanceofrain']) for h in today['hourly']),
        'max_chance_of_snow': max(int(h['chanceofsnow']) for h in today['hourly']),
        'morning': snapshot(3),
        'afternoon': snapshot(5),
        'evening': snapshot(6),
    })

get_current_weather("Atlanta")