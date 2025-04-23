#!/bin/bash

# Cài Chromium cho Playwright (bắt buộc trên Railway)
playwright install chromium

# Chạy file chính của bot
python3 main.py
