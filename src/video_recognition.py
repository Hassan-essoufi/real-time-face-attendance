import os
import cv2
import csv
import numpy as np
from datetime import datetime
import torch

import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity

class VideoFaceRecognition():
    
    def __init__(self, embeddings_db_path="../embeddings.npy", threshold=0.5):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        self._load_models()
        

        self.embeddings_db = self._load_embeddings(embeddings_db_path)
        
        if self.embeddings_db:
            print(f"Embeddings : {len(self.embeddings_db)} persons")
        else:
            print("Embeddings base not found")
        
        self.attendance = {name: "Absent" for name in self.embeddings_db.keys()}
    
    def _load_models(self):
        """
        Loading models: FaceNet & MTCNN
        """

        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.detector = MTCNN(
            keep_all=True,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            device=self.device
        )
    
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _load_embeddings(self, path):
        """
        Loading embeddings
        """
        if os.path.exists(path):
            try:
                return np.load(path, allow_pickle=True).item()
            except:
                print(f"Downloading  error {path}")
                return {}
        return {}
    
    def _extract_face_embedding(self, face_image):
        """
        Extracting embedding from image
        """
        try:
            face_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
            

            with torch.no_grad():
                embedding = self.embedder(face_tensor)
            
            return embedding.cpu().numpy().flatten()
        except:
            return None
    
    def _recognize_face(self, embedding):
        """
        face recognition by embedding
        """
        if embedding is None or not self.embeddings_db:
            return "Unknown", 0.0
        
        best_match = "Unknown"
        best_score = -1
        
        for name, db_embedding in self.embeddings_db.items():
            score = cosine_similarity([embedding], [db_embedding])[0][0]
            if score > best_score:
                best_score = score
                best_match = name
        
        if best_score >= self.threshold:
            return best_match, best_score
        
        return "Unknown", best_score
    
    def process_video(self, input_path, output_path):
        """
        Video face recognition        
        """
        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            return False
        
        print(f"Video treatement: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"FPS: {fps}, Resolution: {width}x{height}")
        
        # Writer creation
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face detection
            boxes, _ = self.detector.detect(frame_rgb)
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    try:
                        face = frame_rgb[y1:y2, x1:x2]
                        embedding = self._extract_face_embedding(face)
                        
                        if embedding is not None:
                            name, score = self._recognize_face(embedding)
                            if name != "Unknown":
                                self.attendance[name] = "Present"
                            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{name} ({score:.2f})"
                            cv2.putText(frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    except:
                        continue
            
            out.write(frame)

            if frame_count % 100 == 0:
                print(f" Treated frames: {frame_count}")
        
        # Cleaning
        cap.release()
        out.release()
        
        print(f"treatement: {frame_count} frames")
        print(f"Saved video: {output_path}")

        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open("attendance.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Status", "Time"])
            for name, status in self.attendance.items():
                writer.writerow([name, status, time_now])
        
        return True



