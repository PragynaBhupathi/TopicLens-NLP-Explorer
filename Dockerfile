FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Copy project
COPY . .

# Create output directories
RUN mkdir -p outputs/saved_models outputs/wordclouds

# Expose API port
EXPOSE 5000

# Default: run the API server
# Run `docker exec <id> python main.py` first to train the model
CMD ["gunicorn", "-b", "0.0.0.0:5000", "--timeout", "120", "api.app:app"]
