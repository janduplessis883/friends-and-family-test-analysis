# Use the official Python image as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the project files to the container
COPY . .

# Install the project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the scheduler.py script when the container starts
CMD ["python", "friendsfamilytest/scheduler.py"]
