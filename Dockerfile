FROM python:3.12-slim
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY demandwatch ./demandwatch
RUN python -m demandwatch.cli download && python -m demandwatch.cli train
RUN useradd --create-home appuser && chown -R appuser:appuser /app/artifacts
USER appuser
EXPOSE 8787
CMD ["python", "-m", "uvicorn", "demandwatch.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8787"]
