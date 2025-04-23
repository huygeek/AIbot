FROM python:3.9-slim  # hoặc python:3.10-slim

# Cài các system dependencies và bash
RUN apt-get update && apt-get install -y \
    bash \
    curl \
    wget \
    ca-certificates \
    ffmpeg \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libasound2 libxshmfence1 libx11-xcb1 \
    libxkbcommon0 libgtk-3-0 libdrm2 libx11-6 libxcb1 libxext6 libexpat1 \
    libxau6 libxdmcp6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Làm việc trong thư mục app
WORKDIR /app

# Copy code
COPY . .

# Cài Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Cài browser cho Playwright
RUN playwright install

CMD ["python", "main.py"]
