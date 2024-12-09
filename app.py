from fastapi import FastAPI, File, UploadFile, Form
import numpy as np
from PIL import Image
import pickle
import io
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sklearn.decomposition import PCA
import torch
from transformers import CLIPProcessor, CLIPModel

app = FastAPI()

app.mount("/images", StaticFiles(directory="coco_images_resized"), name="images")

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html") as f:
        return f.read()

# Load embeddings
with open('image_embeddings.pickle', 'rb') as f:
    df = pickle.load(f)

IMAGE_DATA = [{'path': row['file_name'], 'embedding': np.array(row['embedding'])} for _, row in df.iterrows()]

# Create PCA for image embeddings
embeddings_array = np.stack([x['embedding'] for x in IMAGE_DATA])
pca = PCA(n_components=50)
pca_embeddings = pca.fit_transform(embeddings_array)

for i, item in enumerate(IMAGE_DATA):
    item['pca_embedding'] = pca_embeddings[i]

def get_pca_embedding(image_embedding):
    """Transform the image embedding using PCA"""
    return pca.transform([image_embedding])[0]

def get_clip_image_embedding(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        return model.get_image_features(**inputs).cpu().numpy()[0]

def get_clip_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        return model.get_text_features(**inputs).cpu().numpy()[0]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_top_matches(query_embedding, n=5, use_pca=False):
    return sorted(
        [(img['path'], cosine_similarity(query_embedding, img['pca_embedding'] if use_pca else img['embedding'])) for img in IMAGE_DATA],
        key=lambda x: x[1], reverse=True
    )[:n]

@app.post("/search/text")
async def text_search(query: str = Form()):
    query_embedding = get_clip_text_embedding(query)
    matches = get_top_matches(query_embedding)
    return {"results": [{"image": m[0], "score": float(m[1])} for m in matches]}

@app.post("/search/image")
async def image_search(image: UploadFile = File(...), use_pca: bool = Form(False)):
    img = Image.open(io.BytesIO(await image.read())).convert('RGB')
    img_embedding = get_clip_image_embedding(img)
    if use_pca:
        img_embedding = get_pca_embedding(img_embedding)
    matches = get_top_matches(img_embedding, use_pca=use_pca)
    return {"results": [{"image": m[0], "score": float(m[1])} for m in matches]}

@app.post("/search/combined")
async def combined_search(
    image: UploadFile = File(...),
    text: str = Form(...),
    weight: float = Form(0.5)
):
    img = Image.open(io.BytesIO(await image.read())).convert('RGB')
    img_embedding = get_clip_image_embedding(img)
    text_embedding = get_clip_text_embedding(text)
    
    scores = [
        (img['path'], weight * cosine_similarity(text_embedding, img['embedding']) + (1 - weight) * cosine_similarity(img_embedding, img['embedding']))
        for img in IMAGE_DATA
    ]
    
    matches = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
    return {"results": [{"image": m[0], "score": float(m[1])} for m in matches]}
