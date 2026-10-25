# CellBench live demo: interface + CNN scoring in one container. Qwen runs in Ollama on the host machine.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OLLAMA_URL=http://host.docker.internal:11434 QWEN_MODEL=qwen2.5:0.5b
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY cellbench/ cellbench/
COPY configs/experiment.json configs/experiment.json
COPY data/population_table.csv data/population_table.csv
COPY data/results_full_sweep_summary.csv data/detection_limits_all.csv data/
COPY data/final/cnn/weights_cnn_cells.pt data/final/cnn/weights_cnn_report.pt data/final/cnn/calibration.json data/final/cnn/final_metrics.csv data/final/cnn/normal_reference.json data/final/cnn/
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--app-dir", "cellbench", "--host", "0.0.0.0", "--port", "8000"]
