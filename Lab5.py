import json, requests, streamlit as st
from openai import OpenAI, AuthenticationError
import hydralit_components as hc
from hydralit_components.Loaders.loaders import HyLoader, Loaders

# Retrieve API key and create an OpenAI client.
if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

# Checking if API key is valid.
try:
    st.session_state.client.models.list()
except AuthenticationError:
    st.error("API key needs to be updated.")
    st.stop()

# location can be a city, a zip code, an airport code ('SYR'),
# or a landmark ('Eiffel+Tower')
# note: hard codes units to degrees Fahrenheit
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
    astronomy_data = data["weather"][0]["astronomy"][0]

    # hourly entries are every 3 hours: index 3 = 9am, 5 = 3pm, 6 = 6pm
    def snapshot(i):
        h = today['hourly'][i]
        return {'temp_F': float(h['tempF']),
                'feels_like_F': float(h['FeelsLikeF']),
                'description': h['weatherDesc'][0]['value'],
                'chance_of_rain': int(h['chanceofrain'])}

    return {
        'moon_illumination': int(astronomy_data['moon_illumination']),
        'moon_phase': astronomy_data['moon_phase'],
        "moonrise": astronomy_data['moonrise'],
        "moonset": astronomy_data['moonset'],
        "sunrise": astronomy_data['sunrise'],
        "sunset": astronomy_data['sunset'],
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
    }

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather and today's forecast for a location. "
                           "Use only when the user's request depends on the weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or City and state/country, e.g. 'Syracuse, NY' or 'Lima, Peru'. "
                                       "Use 'Syracuse, NY' if the user gives no location."
                    }
                },
                "required": ["location"]
            }
        }
    }
]

st.markdown(
    """
    <div style="
        background-color: #FFFBE6;
        border: 1px solid #FFE58F;
        border-left: 5px solid #FAAD14;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 24px;
        color: #262730;
    ">
        <div style="font-weight: 600; font-size: 1.05rem; color: #8C5300; margin-bottom: 6px;">
            Bot Details
        </div>
        <ul style="margin: 0; padding-left: 20px; font-size: 0.9rem; line-height: 1.6;">
            <li>Describe your plans, and include a location if you want.</li>
            <li>If you don't give a location, the bot uses <em>Syracuse, NY</em>.</li>
            <li><strong>The bot only checks the weather when your request needs it.</strong></li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# Hiding the "Press Enter to submit" caption in my input field
st.html("""
    <style>
    div[data-testid="InputInstructions"] {
        display: none !important;
    }
    </style>
""")

# Show title and description.
st.title(":material/sunny: Mel's What-to-Wear Bot :material/cloudy:")
st.caption("Tell me your plans and I'll suggest what to wear and what to do outdoors.")

user_input = st.text_input(
    "What are your plans?",
    placeholder="e.g. Going for a walk in Atlanta?"
)

submitted = st.button("Get Advice", type="primary")

system_prompt = (
    "You are a helpful assistant that gives clothing suggestions and "
    "weather-appropriate outdoor activity ideas. If the user's request depends "
    "on the weather, call the get_current_weather tool. If no location is given, "
    "use 'Syracuse, NY'."
)

if submitted and user_input:
    client = st.session_state.client
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    response = client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append(response_message.to_dict())

        for tool_call in tool_calls:
            if tool_call.function.name == "get_current_weather":
                args = json.loads(tool_call.function.arguments)
                location = args.get("location") or "Syracuse, NY"

                try:
                    weather = get_current_weather(location)
                except Exception as e:
                    weather = {"error": str(e)}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(weather),
                })

        messages.append({
            "role": "system",
            "content": (
                "First off: Be fun and funny, but still helpful, with your responses. "
                "Using the weather data above, suggest appropriate clothes to wear today "
                "and outdoor activities that suit the weather. Mention how conditions "
                "change through the day. If the weather data has an error, tell the user "
                "the location couldn't be found. With the astronomy data, moon and sun data, "
                "like mentioned before, be fun and witty. For example, talk about werewolves, "
                "vampires, or other creatures and interesting things related to nighttime, "
                "the moon, sunrise, sunset, or astronomy in general. Also talk about how the lighting "
                "due to the weather, the time, outside lighting (like moon and sun) will affect pictures "
                "and selfies."
            ),
        })

        stream = client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=messages,
            stream=True
        )
        st.write(stream)

    else:
        # No tool needed, so show the answer directly.
        st.write(response_message.content)