# Step 3: Containerize it.
#
# Why Docker here specifically? Your laptop might have Python 3.11, the
# server might have 3.9, onnxruntime versions might drift, a teammate's
# machine might be missing a system library. Docker freezes ALL of that
# into one reproducible image so "it works on my machine" stops being
# a valid excuse.

FROM python:3.11-slim

WORKDIR /app

# Copy dependency list first (before code) so Docker can cache this layer -
# if only your code changes, Docker won't need to reinstall dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code and the exported model.
COPY app.py .
COPY resnet18.onnx .
COPY imagenet_classes.txt .

EXPOSE 8000

# Run the FastAPI app via uvicorn when the container starts.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
