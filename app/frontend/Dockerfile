FROM python:3.10-slim
WORKDIR /app
COPY app/frontend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY . /app
CMD ["python", "-u", "app/frontend/user_intf.py"]