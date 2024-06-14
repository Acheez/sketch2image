FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev

WORKDIR /app

COPY app/requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY app/ .

EXPOSE 8501

# Set the default command to run the application with Streamlit
CMD ["streamlit", "run", "app.py"]
