# Use official Python image
FROM python:3.12-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the app on port 8080
EXPOSE 8080

# Run your FastAPI app using uvicorn
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8080"]
