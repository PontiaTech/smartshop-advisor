FROM python:3.13.7-slim-trixie

# Instalar dependencias y limpiar la cache al final para no agrandar la imagen
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*


# RUN python -m venv /opt/venv
# ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

COPY ./src .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]