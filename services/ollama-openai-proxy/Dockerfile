# Use official Python image
FROM python:3.12-bookworm

# Set working directory inside the container
WORKDIR /app

# Update pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file first for better Docker layer caching
# This allows Docker to cache the pip install step if requirements don't change
COPY requirements-ollama-openai-proxy.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements-ollama-openai-proxy.txt

# Copy application code
COPY app.py .

# Not required for FastAPI, but good practice
EXPOSE 80

# Health check to ensure the service is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

# Command to run the application
CMD ["fastapi", "run", "app.py", "--port", "80", "--host", "0.0.0.0"]