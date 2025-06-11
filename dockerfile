FROM python:3.10

# install system dependencies
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

# install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app folder
COPY . /app
WORKDIR /app

# run Streamlit app
CMD ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.enableCORS=false"]