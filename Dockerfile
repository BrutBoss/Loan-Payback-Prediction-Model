# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the trained model file
# IMPORTANT: Ensure 'xgb_tuned_model.pkl' is in the same directory as the Dockerfile when building
COPY app.py .
COPY xgb_tuned_model.pkl .

# Expose the port where the Flask app runs (defined in app.py)
EXPOSE 9696

# Define the command to run the application using Gunicorn
# FIX: Added '--timeout 120' to prevent workers from timing out during model loading.
# The format is: gunicorn -w <workers> -b <host:port> --timeout <seconds> <module_name>:<app_instance_name>
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "--timeout", "120", "app:app"]