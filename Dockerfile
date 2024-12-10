# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory to /app
WORKDIR /app

# Copy only requirements.txt first to leverage Docker caching
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY app.py /app/
COPY saved_models/ /app/saved_models/

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define the default command to run your Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]