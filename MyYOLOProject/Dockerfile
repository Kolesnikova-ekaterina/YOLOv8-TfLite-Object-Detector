FROM ultralytics/ultralytics:latest

WORKDIR /app
COPY . /app/
RUN pip install -r requirements.txt
RUN chmod +x train.py
ENTRYPOINT ["python", "/app/train.py"]
