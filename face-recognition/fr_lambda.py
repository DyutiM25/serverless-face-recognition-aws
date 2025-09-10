import json
import boto3
import os
import torch
import base64
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import io

# Initialize AWS clients globally
sqs = boto3.client('sqs')

# Load model and weights globally (cold start optimization)
print("Model loading...")
resnet = InceptionResnetV1(pretrained='vggface2').eval()
MODEL_WT_PATH = 'resnetV1_video_weights.pt'
saved_data = torch.load(MODEL_WT_PATH)
embedding_list = saved_data[0]
name_list = saved_data[1]
print("Model loaded...")

# Get environment variable for response queue
RESPONSE_QUEUE_URL = f"https://sqs.us-east-1.amazonaws.com/198257650092/1232833006-resp-queue"

def handler(event, context):
    #print(event)
    for record in event['Records']:
        try:
            # Extract message from SQS
            body = json.loads(record['body'])

            # Required fields
            request_id = body['request_id']
            filename = body['filename']
            image_b64 = body['faces']

            # Decode base64 image
            image_bytes = base64.b64decode(image_b64)
            face_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Convert to tensor
            face_np = np.array(face_pil).astype(np.float32) / 255.0
            face_np = np.transpose(face_np, (2, 0, 1))
            face_tensor = torch.tensor(face_np, dtype=torch.float32).unsqueeze(0)

            # Compute embedding
            emb = resnet(face_tensor).detach()
            dist_list = [torch.dist(emb, db_emb).item() for db_emb in embedding_list]
            idx_min = dist_list.index(min(dist_list))
            result = name_list[idx_min]

            # Send result to response queue
            response_payload = {
                "request_id": request_id,
                "result": result
            }

            sqs.send_message(
                QueueUrl=RESPONSE_QUEUE_URL,
                MessageBody=json.dumps(response_payload)
            )

        except Exception as e:
            print(f"Error processing record: {e}")
            continue
