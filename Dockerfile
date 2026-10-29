FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY bot ./bot
COPY docs ./docs
COPY static ./static
ENV PORT=8000
# Free hosts (Render, Railway, Fly, Hugging Face Spaces) inject $PORT.
CMD uvicorn --factory bot.main:create_app --host 0.0.0.0 --port ${PORT}
