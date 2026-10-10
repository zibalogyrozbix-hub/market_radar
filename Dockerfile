FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
# MetaTrader5 is Windows-only: in Docker use DATA_PROVIDER=csv (or synthetic for a demo)
CMD ["python", "main.py", "--daemon"]
