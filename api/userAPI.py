from fastapi import APIRouter
import chromadb
from model.User import User 
import embedding_model.modelUpload as modelUpload


client=chromadb.PersistentClient()

embed_model=modelUpload.model_upload()

userRouter = APIRouter()

@userRouter.post("/make/collection")
async def make_collection(user: User):
    client.create_collection(name=user.userId, metadata={"hnsw:space":"cosine"})
    client.create_collection(name=user.userId+"_buffer",metadata={"hnsw:space":"cosine"})
    return 200; 
