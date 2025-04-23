# Base image dùng Python 3.9 slim (có bash + apt)
FROM python:3.9-slim

# Thiết lập biến môi trường cơ bản
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Cài các gói system cần cho Playwright + âm thanh
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash \
    curl \
    ffmpeg \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libgbm1 \
    libxkbcommon0 \
    libasound2 \
    libx11-xcb1 \
    libxrandr2 \
    libxshmfence1 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libxau6 \
    libxdmcp6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy code vào container
COPY . .

# Cài pip và các thư viện Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    playwright install --with-deps

# Chạy bot
CMD ["python", "bot/main.py"]
