import chromadb
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from sentence_transformers import SentenceTransformer
sentences = ["고양이", "먹었어"]

model = SentenceTransformer("bespin-global/klue-sroberta-base-continue-learning-by-mnr")
embeddings = model.encode(sentences)
embedding_list=embeddings.tolist()
print(embeddings.shape)
print(embeddings)