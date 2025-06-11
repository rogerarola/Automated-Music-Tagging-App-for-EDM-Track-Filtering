FROM python:3.10

# Install system packages needed
RUN apt update && apt install -y \
    ffmpeg \
    libfftw3-dev \
    libyaml-dev \
    libtag1-dev \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libsamplerate0-dev \
    libboost-all-dev \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code into the container
COPY . /app
WORKDIR /app

# Streamlit flags to work on Hugging Face
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.enableCORS=false"]