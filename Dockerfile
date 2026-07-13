# This Dockerfile does not build as-is. Fix it and note what was wrong in your README.
# Hints: there is more than one issue.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
