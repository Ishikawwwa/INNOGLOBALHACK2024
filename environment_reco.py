import numpy as np
import cv2
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid
import torch
from torchvision import models, transforms

class EnvironmentRecognition():
    def __init__(self):
        # Load a pre-trained ResNet model for environment feature extraction
        self.model = models.resnet50(pretrained=True)
        # Remove the final classification layer to get the 2048-dimensional embedding
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()  # Set to evaluation mode

        # Set up Qdrant client
        self.qdrant_client = QdrantClient(
            url="https://eec77cfc-3fba-40e7-a969-00a8b4aef15b.us-east4-0.gcp.cloud.qdrant.io",
            api_key="m6-oeZPhiJFFOmD8E01cPZyh1XvgntHRIcZcsBnILyfVO6L0UfEr1A"
        )
        
        # Define vector size and collection name for environment embeddings
        self.vector_size = 2048  # Size of ResNet50 feature vectors
        self.collection_name = "environment_embeddings"
        
        # Create the collection if it doesn't exist
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )

        # Transformation for input images
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def encode_environment(self, image):
        """Extract an environment embedding from an image."""
        # Preprocess the image and convert it to a tensor
        input_tensor = self.preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            # Pass through the ResNet model to get feature embedding
            embedding = self.model(input_tensor).squeeze().numpy()
        
        return embedding

    def generate_unique_environment_id(self):
        """Generate a unique ID for an environment using UUID."""
        return str(uuid.uuid4())

    def store_environment(self, embedding):
        """Store a new environment embedding in Qdrant."""
        env_id = self.generate_unique_environment_id()
        point = PointStruct(id=env_id, vector=embedding.tolist())
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
