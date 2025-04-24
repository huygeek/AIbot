import os
import requests
import datetime
import unicodedata

def normalize_text(text: str) -> str:
    # Bỏ dấu, thường hóa, loại ký tự lạ
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8').lower()
    return ''.join(e for e in text if e.isalnum() or e.isspace()).strip()

def normalize_city(city_name: str) -> str:
    norm = normalize_text(city_name)
    norm = norm.replace(".", "").replace("-", "").replace("_", "").replace("  ", "").replace(" ", "")

    if "hanoi" in norm:
        return "Hanoi"
    elif any(x in norm for x in ["hochiminh", "tphcm", "saigon", "hochiminhcity", "tpchiminh"]):
        return "Ho Chi Minh City"
    else:
        return city_name

def get_forecast(city_name: str = "Hà Nội") -> str:
    city_name = normalize_city(city_name)
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "❌ Thiếu API Key thời tiết."

    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={api_key}&units=metric&lang=vi"
    try:
        response = requests.get(url).json()
        if response.get("cod") != "200":
            return f"❌ Không tìm thấy thành phố '{city_name}'."

        daily = {}
        for item in response["list"]:
            dt = datetime.datetime.fromtimestamp(item["dt"])
            day = dt.strftime("%A")
            if day not in daily:
                daily[day] = {
                    "temps": [item["main"]["temp_min"], item["main"]["temp_max"]],
                    "desc": item["weather"][0]["description"],
                    "icon": weather_icon(item["weather"][0]["main"]),
                    "humidity": item["main"]["humidity"]
                }

        result = f"📅 Dự báo thời tiết tại {city_name}:
"
        for i, (day, data) in enumerate(list(daily.items())[:5]):
            result += f"- {day}: {data['icon']} {round(data['temps'][1])}°C / {round(data['temps'][0])}°C – {data['desc'].capitalize()}, độ ẩm {data['humidity']}%
"

        return result.strip()
    except Exception as e:
        return f"❌ Lỗi khi lấy dự báo: {e}"

def weather_icon(condition: str) -> str:
    icons = {
        "Clear": "☀️", "Clouds": "☁️", "Rain": "🌧️", "Thunderstorm": "⛈️",
        "Snow": "❄️", "Drizzle": "🌦️", "Mist": "🌫️", "Fog": "🌁"
    }
    return icons.get(condition, "🌡️")

def get_weather(city_name: str = "Hà Nội") -> str:
    city_name = normalize_city(city_name)
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "❌ Thiếu API Key thời tiết."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric&lang=vi"

    try:
        res = requests.get(url).json()
        if res.get("cod") != 200:
            return f"❌ Không tìm thấy thành phố '{city_name}'."

        main, weather, wind = res["main"], res["weather"][0], res["wind"]
        return (
            f"{weather_icon(weather['main'])} Thời tiết tại {city_name}:
"
            f"- Nhiệt độ: {main['temp']}°C
"
            f"- Trạng thái: {weather['description'].capitalize()}
"
            f"- Độ ẩm: {main['humidity']}%
"
            f"- Gió: {wind['speed']} km/h"
        )
    except Exception as e:
        return f"❌ Lỗi lấy thời tiết: {e}"
