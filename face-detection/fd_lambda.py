import os
import json
import boto3
import base64
import numpy as np
from io import BytesIO
from facenet_pytorch import MTCNN
from PIL import Image
import torch

sqs = boto3.client('sqs')

class FaceDetector:
    def __init__(self):
        self.mtcnn          = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection

    def detect_faces(self, test_image_path):
        # Step-1: Read the image
        # img     = Image.open(test_image_path).convert("RGB")
        image_data = base64.b64decode(test_image_path)
        img = Image.open(BytesIO(image_data)).convert("RGB")
        img     = np.array(img)
        img     = Image.fromarray(img)

        #key = os.path.splitext(os.path.basename(test_image_path))[0].split(".")[0]

        # Step:2 Face detection
        face, prob = self.mtcnn(img, return_prob=True, save_path=None)

        if face != None:


            face_img = face - face.min()  # Shift min value to 0
            face_img = face_img / face_img.max()  # Normalize to range [0,1]
            face_img = (face_img * 255).byte().permute(1, 2, 0).numpy()  # Convert to uint8

            # Convert numpy array to PIL Image
            face_pil        = Image.fromarray(face_img, mode="RGB")
            #face_img_path   = os.path.join(output_path, f"{key}_face.jpg")

            # Save face image
            #face_pil.save(face_img_path)
            # return face_img_path
            # Encode the image to base64
            buffered = BytesIO()
            face_pil.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            return img_base64

        else:
            print(f"No face is detected")
            return

def handler(event, context):
    print("Event and Context")
    # Initialize face detector (stays warm between invocations)
    if not hasattr(context, 'face_detector'):
        context.face_detector = FaceDetector()
    print("Detector Mode")
    
    try:
        # Validate input
        if 'body' not in event:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No body in request'})
            }
            
        body = json.loads(event['body'])
        required_fields = ['content', 'request_id', 'filename']
        if not all(field in body for field in required_fields):
            return {
                'statusCode': 400,
                'body': json.dumps({'error': f'Missing required fields: {required_fields}'})
            }

        print("Before Processing Image")
        # Process image
        faces = context.face_detector.detect_faces(body['content'])
        
        # (Optional) Save for debugging
        # image.save(f"/tmp/{image_id}.jpg")

        # Send to SQS request queue
        queue_url = f"https://sqs.us-east-1.amazonaws.com/198257650092/1232833006-req-queue"
        
        sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps({
                'request_id': body['request_id'],
                'faces': faces,
                'filename': body['filename']
            })
        )

        

        return {
            'statusCode': 200,
            'body': json.dumps({
                'faces': faces,
                'request_id': body['request_id'],
                'filename': body['filename']
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'request_id': body.get('request_id', 'unknown')
            })
        }