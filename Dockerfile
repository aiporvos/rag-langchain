FROM python:3.11

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for ChromaDB persistence
RUN mkdir -p chroma_db

EXPOSE 7860

# Entrypoint for Gradio app
ENTRYPOINT ["python", "app.py"]
