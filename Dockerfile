# Cogemos imagen de Python
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# Copiamos archivos necesarios
COPY requirements.txt .

# Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos contenido de la carpeta app al contenedor
COPY app/ ./app/

# Copiar el modelo al contenedor
COPY classifier_model.pkl /app/classifier_model.pkl
WORKDIR /app


# Exponemos el puerto
EXPOSE 8000

# Comando por defecto para lanzar la app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
