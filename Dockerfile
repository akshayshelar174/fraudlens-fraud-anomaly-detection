FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt pyproject.toml ./
COPY fraudlens ./fraudlens
COPY dashboard.py ./dashboard.py
RUN pip install --no-cache-dir . \
    && python -m fraudlens.training --rows 12000

EXPOSE 8000 8501
CMD ["uvicorn", "fraudlens.api:app", "--host", "0.0.0.0", "--port", "8000"]
