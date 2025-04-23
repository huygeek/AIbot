FROM python:3.10-slim

# Cài dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg \
    ffmpeg \
    libnss3 \
    libatk-bridge2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libgtk-3-0 \
    libdrm2 \
    libgbm1 \
    libasound2 \
    libxrandr2 \
    libxss1 \
    libxtst6 \
    libatk1.0-0 \
    libnspr4 \
    libdbus-1-3 \
    fonts-liberation \
    libappindicator3-1 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app
COPY . .

# Cài Python package
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Cài Chromium cho Playwright
RUN python -m playwright install chromium

CMD ["python", "bot/main.py"]
