# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create models directory
RUN mkdir -p models

# Install package in development mode
RUN pip install -e .

# Run predictions
CMD ["python", "-m", "src.predict"]