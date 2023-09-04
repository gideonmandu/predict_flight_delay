# syntax=docker/dockerfile:1.2
FROM python:latest
# put you docker configuration here
# Set working directory
WORKDIR /app

# Create a non-root user and switch to it
RUN useradd --create-home appuser
USER appuser

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .
# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY --chown=appuser . .

# Expose the port the app runs on
EXPOSE 8000

# Set the default command to execute the FastAPI application
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]