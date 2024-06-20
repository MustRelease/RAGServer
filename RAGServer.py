import chromadb
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import embedding_model.modelUpload as modelUpload
import numpy as np
from typing import List
import ast
from model.Item import Item
from model.User import User
from api.userAPI import userRouter
from api.memoryCrudAPI import memoryRouter

#uvicorn RAGServer:app --reload --host=0.0.0.0 --port=8800
#python -m uvicorn FastApi:app --reload

app=FastAPI()

app.include_router(userRouter)
app.include_router(memoryRouter)

client=chromadb.PersistentClient()

embed_model=modelUpload.model_upload()