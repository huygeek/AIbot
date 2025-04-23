# Dùng image chuẩn của Python với Debian (KHÔNG dùng Alpine)
FROM python:3.9-slim

# Thiết lập môi trường Python
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Cài thư viện hệ thống cần thiết cho Playwright Chromium
RUN apt-get update && apt-get install -y \
    wget curl gnupg ffmpeg \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 \
    libxcomposite1 libxdamage1 libxfixes3 \
    libxrandr2 libgbm1 libasound2 libxshmfence1 \
    libx11-xcb1 libxkbcommon0 libgtk-3-0 \
    libdrm2 libx11-6 libxcb1 libxext6 libexpat1 \
    libxau6 libxdmcp6 ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy source vào container
WORKDIR /app
COPY . .

# Cài Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Cài trình duyệt Chromium cho Playwright
RUN playwright install --with-deps

# Chạy bot
CMD ["python", "bot/main.py"]
