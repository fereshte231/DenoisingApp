FROM python:3.9

WORKDIR /app

RUN git clone https://github.com/fereshte231/DenoisingApp.git .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN python -m pip install -r requirements2.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]