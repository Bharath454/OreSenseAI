FROM python:3.11-slim

# System deps for GDAL/Rasterio/GeoPandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev libgeos-dev libproj-dev \
    libpq-dev gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV GDAL_CONFIG=/usr/bin/gdal-config

EXPOSE 8000
