FROM python:3.11-slim
WORKDIR /app_code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-electra-base')"
COPY ./app ./app
EXPOSE 8000
CMD [ "uvicorn","app.main:app","--host", "0.0.0.0", "--port", "8000" ]
