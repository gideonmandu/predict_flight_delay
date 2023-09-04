# syntax=docker/dockerfile:1.2
FROM python:latest
# put you docker configuration here
# Install make
RUN apt-get update && apt-get install -y make
# Set working directory
WORKDIR /app

# Create a non-root user and switch to it
#RUN useradd --create-home appuser
#USER appuser

# First copy only the requirements files, and install them.
# This layer will be cached if the files don't change.
COPY requirements.txt requirements-dev.txt requirements-test.txt Makefile ./
RUN make install

# Copy the rest of the project files into the container
#COPY --chown=appuser . .
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Set the default command to execute the FastAPI application
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--log-level", "info"]