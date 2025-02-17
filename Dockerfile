# Use AWS Deep Learning Container as a base image
# FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.2.0-cpu-py310-ubuntu20.04-sagemaker-v1.37
FROM public.ecr.aws/docker/library/python:3.12-slim

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy training script and dependencies
COPY . .

# List all files and directories after copying
RUN echo "Files and directories in /app:" && ls -R /app

# Install additional Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set entry point for the container
CMD ["python", "app/models/model-a/src/code/sg_model.py"]
