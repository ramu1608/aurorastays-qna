
# Use a minimal base image with Python 3.13.15-slim installed
FROM python:3.13.15-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory to the container's /app directory
COPY . .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies listed in requirements.txt
RUN pip install -r requirements.txt

# Run the Streamlit application on port 8501
# 0.0.0.0 makes the application accessible outside the container
# XSRF protection is disabled for easier external access
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
