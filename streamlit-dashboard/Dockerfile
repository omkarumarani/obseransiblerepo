FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY shared.py .
COPY pages/ ./pages/

# Streamlit config — headless, no telemetry
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_BASE=dark \
    STREAMLIT_THEME_PRIMARY_COLOR=#2196f3 \
    STREAMLIT_THEME_BACKGROUND_COLOR=#0a0e27 \
    STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR=#1a1f3a \
    STREAMLIT_THEME_TEXT_COLOR=#e1e4e8

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
