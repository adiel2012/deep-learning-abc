FROM python:3.9-slim

WORKDIR /app

# Install git for huggingface transformers if needed (usually not for packaged releases)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# List notebooks
RUN ls -la *.ipynb

# Default command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
