# Use a modern, slim Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /code

# --- THE FIX IS HERE ---
# Install the missing system library required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy just the requirements file first
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your application code
COPY ./main.py /code/main.py

# Expose the port that Render expects for its Docker runtime
EXPOSE 10000

# Command to run the application, hardcoding the port to 10000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
