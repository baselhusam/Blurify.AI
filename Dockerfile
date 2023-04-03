FROM python:3.7.16-bullseye

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt
RUN packages="libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev" && \
    apt-get update && \
    apt-get install -y --no-install-recommends $packages && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run" ]

CMD [ "main.py" ]