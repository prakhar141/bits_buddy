FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential poppler-utils     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . . 
COPY data/ /app/data/

EXPOSE 8501
ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "cleaned_buddy.py", "--server.address=0.0.0.0", "--server.port=8501"]
