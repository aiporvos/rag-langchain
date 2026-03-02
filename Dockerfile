FROM python:3.11

WORKDIR /app

# The full python image already has git, build-essential, etc.
# We only need to install the python requirements.

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for ChromaDB persistence
RUN mkdir -p vector_db_brain_balance

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
