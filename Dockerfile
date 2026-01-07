FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the server directly (keeps the container alive)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
