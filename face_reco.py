import dlib
import numpy as np
import cv2
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid

class FaceRecognition():
    def __init__(self):
        # Initialize dlib face recognition
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
        
        self.qdrant_client = QdrantClient(
            url="https://eec77cfc-3fba-40e7-a969-00a8b4aef15b.us-east4-0.gcp.cloud.qdrant.io",
            api_key="m6-oeZPhiJFFOmD8E01cPZyh1XvgntHRIcZcsBnILyfVO6L0UfEr1A"
        )
        
        self.vector_size = 128
        self.collection_name = "face_embeddings"
  
        if not self.qdrant_client.collection_exists:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )

    def show_dets(self, dets, image, image_name, save_folder_name="output_imgs"):
        for k, d in enumerate(dets):
            print(f"Detected face {k+1}: Left: {d.left()} Top: {d.top()} Right: {d.right()} Bottom: {d.bottom()}")
            shape = self.sp(image, d)
            for i in range(0, 68):
                x = shape.part(i).x
                y = shape.part(i).y
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.imwrite(f"{save_folder_name}/{image_name.split('.')[0]}_landmarks.jpg", image)

    def encode_face(self, image):
        dets = self.detector(image, 1)
        #self.show_dets(dets, image)
        if len(dets) == 0:
            return None
        shape = self.sp(image, dets[0])
        face_descriptor = self.facerec.compute_face_descriptor(image, shape)
        return np.array(face_descriptor)

    def generate_unique_face_id(self):
        """Generate a unique face_id using UUID."""
        return str(uuid.uuid4())

    def store_face(self, embedding):
        """Store a new face embedding in Qdrant."""
        face_id = self.generate_unique_face_id()
        point = PointStruct(id=face_id, vector=embedding.tolist())
        self.qdrant_client.upsert(collection_name=self.collection_name, points=[point])

    def is_match(self, embedding, threshold=0.95):
        """Check if a given embedding matches any in the Qdrant collection."""
        # Perform a similarity search in Qdrant
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=1  # We only need the top result
        )
        # Check if similarity score meets the threshold
        if search_result and search_result[0].score >= threshold:
            return True, search_result[0].id
        return False, None
