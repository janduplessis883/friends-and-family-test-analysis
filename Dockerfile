# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Install any dependencies (if you have a requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copy your scripts into the container
COPY /friendsfamilytest/scheduler.py .
COPY /friendsfamilytest/data.py /usr/src/app/friendsfamilytest/

# Run scheduler.py when the container launches
CMD ["python", "scheduler.py"]
