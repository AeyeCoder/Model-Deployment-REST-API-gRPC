TensorFlow Serving Deployment: REST & gRPC — End-to-End Guide

This repository demonstrates production-grade deployment of deep learning models using TensorFlow Serving, exposing the model through both REST and gRPC APIs, using Docker + Linux (Ubuntu VM), and querying the server from Windows client.

It documents the complete real-world ML deployment workflow, including:

Model training

SavedModel export

Dockerized serving

Linux server setup

Cross-OS inference

REST vs gRPC comparison

Performance-oriented serving pipeline

📌 Project Overview

We trained an MNIST digit classification model using TensorFlow/Keras, exported it in TensorFlow SavedModel format, and deployed it using TensorFlow Serving inside Docker on Ubuntu VM.

The model is accessed:

via REST API (HTTP + JSON)

via gRPC (binary protocol for high-performance serving)

The client runs on Windows, sending inference requests to the Linux VM server.

This simulates real industry AI deployment architecture.

🧠 System Architecture
Windows (Client)
     |
     |  REST / gRPC Requests
     |
Ubuntu VM (Docker + TensorFlow Serving)
     |
     |  Loads SavedModel
     |
TensorFlow Model Server

🛠️ Tech Stack

TensorFlow / Keras

TensorFlow Serving

Docker

Ubuntu Linux (VMware)

Python

REST API

gRPC API

Windows Client

📂 Project Structure
models/
└── mnist/
    └── 001/
        └── saved_model.pb + variables/

client/
├── rest_client.py
└── grpc_client.py

🧪 Model Training

A simple neural network is trained on MNIST:

Flatten → Dense(100) → Dense(10 softmax)

The trained model is exported in TensorFlow SavedModel format, which is the only supported format for TensorFlow Serving.

💾 Model Export (SavedModel Format)
model.save("mnist/001")


This produces:

mnist/
 └── 001/
      ├── saved_model.pb
      └── variables/


TensorFlow Serving loads model name + versioned folder automatically.

🐳 Docker + TensorFlow Serving Setup (Ubuntu VM)
Pull TensorFlow Serving Image
docker pull tensorflow/serving

Run TensorFlow Serving Container
docker run -p 8501:8501 -p 8500:8500 \
  --mount type=bind,source=/home/awais/models,target=/models \
  -e MODEL_NAME=mnist \
  tensorflow/serving

What this does:
Flag	Purpose
-p 8501	REST API port
-p 8500	gRPC port
--mount	Attach local model directory
MODEL_NAME	Model name inside TF Serving
🌐 REST API Inference (Windows Client)
Request Code
import requests, json

url = "http://VM_IP:8501/v1/models/mnist:predict"

data = {
    "signature_name": "serving_default",
    "instances": [x_test[0].tolist()]
}

response = requests.post(url, json=data)
print(response.json())

Clean Output Formatting
import numpy as np

pred = response.json()["predictions"][0]
digit = np.argmax(pred)
confidence = np.max(pred)

print("Prediction:", digit)
print("Confidence:", round(confidence, 4))

⚡ gRPC Inference (High Performance)

gRPC offers binary serialization + persistent connections, making it much faster and production-ready.

Why gRPC?
REST	gRPC
JSON	Binary Protobuf
Slower	Much Faster
Debug-friendly	High-performance
Human readable	Production grade
High latency	Low latency
gRPC Client Setup

TensorFlow Serving API

grpcio

protobuf

pip install grpcio tensorflow-serving-api protobuf

gRPC Inference Code
import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

channel = grpc.insecure_channel("VM_IP:8500")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = "mnist"
request.model_spec.signature_name = "serving_default"

x = x_test[0].reshape(1,28,28).astype(np.float32)

request.inputs["flatten"].CopyFrom(
    tf.make_tensor_proto(x, dtype=tf.float32)
)

response = stub.Predict(request, timeout=10.0)

Clean gRPC Output
probs = response.outputs["output_0"].float_val
probs = np.array(probs)

digit = probs.argmax()
confidence = probs.max()

print("Prediction:", digit)
print("Confidence:", round(confidence,4))

🚀 Why Docker + Linux + TensorFlow Serving?

This setup matches real production AI infrastructure.

Industry Deployment Stack:
Component	Reason
Linux	Stable, fast, secure
Docker	Reproducible deployments
TF Serving	High-performance inference
gRPC	Low latency, scalable
Versioned Models	Safe model updates

This is exactly how FAANG / AI startups deploy ML models.

🌍 Is This Model Online?

The model is locally hosted inside a VM

Accessible across different OS (Windows → Linux)

Not publicly accessible unless:

Exposed via public IP

Deployed on cloud (AWS / GCP / Azure)

🧠 What This Project Teaches

This project provides deep hands-on learning of:

Production ML deployment

Real server-client architecture

Dockerized inference pipelines

REST vs gRPC design tradeoffs

Industry-grade AI system design

🏁 Final Outcome

You now own full-stack AI deployment capability:

Model Training → Export → Docker → Linux Server → REST + gRPC → Windows Client


This is real-world applied AI engineering.

⭐ Recommended Next Steps

Load testing (latency benchmarking)

GPU serving

Kubernetes deployment

Model version switching

Canary rollout

Autoscaling inference

👨‍💻 Author

Awais
Applied AI Engineer
Pakistan 🇵🇰
