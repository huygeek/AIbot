import os
import requests

def weather_icon(condition: str) -> str:
    icons = {
        "Clear": "☀️", "Clouds": "☁️", "Rain": "🌧️", "Thunderstorm": "⛈️",
        "Snow": "❄️", "Drizzle": "🌦️", "Mist": "🌫️", "Fog": "🌁"
    }
    return icons.get(condition, "🌡️")

def get_weather(city_name: str = "Hà Nội") -> str:
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "❌ Thiếu API Key thời tiết. Gắn biến OPENWEATHER_API_KEY vào .env nha."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric&lang=vi"
    try:
        res = requests.get(url)
        data = res.json()
        if data.get("cod") != 200:
            return f"❌ Không tìm thấy thành phố '{city_name}'."

        main, weather, wind = data["main"], data["weather"][0], data["wind"]
        return (
            f"{weather_icon(weather['main'])} Thời tiết tại {city_name.title()}:\n"
            f"- Nhiệt độ: {main['temp']}°C\n"
            f"- Trạng thái: {weather['description'].capitalize()}\n"
            f"- Độ ẩm: {main['humidity']}%\n"
            f"- Gió: {wind['speed']} km/h\n\n"
            "Chúc bạn một ngày mát mẻ nha 😌"
        )
    except Exception as e:
        return f"❌ Lỗi lấy thời tiết: {e}"
