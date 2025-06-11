FROM python:3.10

# Install system dependencies needed
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

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app folder
COPY . /app
WORKDIR /app

# Run the Streamlit app from inside the app folder
CMD ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.enableCORS=false"]