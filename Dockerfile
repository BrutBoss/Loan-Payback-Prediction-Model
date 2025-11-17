# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code, model, AND the templates folder
COPY app.py .
COPY xgb_tuned_model.pkl .
COPY templates templates

# Expose the port where the Flask app runs
EXPOSE 9696

# Define the command to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "--timeout", "360", "app:app"]