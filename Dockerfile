FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

RUN pip install huggingface_hub && \
    mkdir -p models && \
    python -c "
from huggingface_hub import hf_hub_download
import os
try:
    path = hf_hub_download(repo_id='Yug0911/rice-disease-model', filename='best_5class.h5', local_dir='models')
    print(f'Model downloaded to: {path}')
except Exception as e:
    print(f'Error: {e}')
"

EXPOSE 8080

CMD ["python", "app.py"]