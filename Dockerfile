FROM python:3.13.8

WORKDIR /app
COPY . /app

RUN pip install uv && uv pip install -r pyproject.toml

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
