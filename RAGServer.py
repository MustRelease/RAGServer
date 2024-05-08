import chromadb
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import modelUpload
import numpy as np

#uvicorn RAGServer:app --reload --host=0.0.0.0 --port=8800
#python -m uvicorn FastApi:app --reload

app=FastAPI()

client=chromadb.PersistentClient()

embed_model=modelUpload.model_upload()

class Item(BaseModel):
    userId: str
    timestamp: str
    observation: str
    importance: float
    
class User(BaseModel):
    userId: str
    
@app.post("/make/collection")
async def make_collection(user: User):
    client.create_collection(name=user.userId, metadata={"hnsw:space":"cosine"})
    return 200; 

@app.post("/memory/add")
async def add_memory(infomation: Item):
    collection=client.get_collection(name=infomation.userId)
    #기억 정보 임베딩
    db_embedding_word=[infomation.observation]
    embeddings=embed_model.encode(db_embedding_word)
    db_embedding=embeddings.tolist()
    
    #정보 모으기
    id=str(collection.count()+1) 
    print(id)
    userId=infomation.userId
    timestamp=infomation.timestamp
    observation=infomation.observation
    importance=infomation.importance
    
    db_metadatas=[  #바꿔야 하는 부분
        {"userId":userId, "timeStamp":timestamp, "observation": observation, "importance": importance}
    ]
    #DB 삽입
    collection.add(
        ids=id,
        metadatas=db_metadatas,
        embeddings=db_embedding
    )
    
    return db_metadatas;
    
@app.get("/memory/get/{query}/{userid}")
async def get_memory(query: str, userid: str):
    collection=client.get_collection(name=userid)
    n_result=collection.count() #결과 출력 개수
    #쿼리 임베딩
    print(query)
    query_embedding_word=[query] #바꿔야 하는 부분
    query_embedding=embed_model.encode(query_embedding_word)
    query_embedding=query_embedding.tolist()
    
    result=collection.query(
        query_embeddings=query_embedding[0],
        n_results=n_result,
        # where={"userId":"king123"} #유저 아이디 필터링
    )
    print(result)
    #3가지의 합 계산 코드 작성하기
    to_prompt_list=calculate(result)
    
    #Obeservation반환

    # print(result_meta)
    return to_prompt_list


def calculate(result_list):
    result_meta=(result_list["metadatas"])[0]
    print(result_meta)
    print(len(result_meta))
    result_distance=(result_list["distances"])[0]
    print(result_distance)
    
    #시간, 행동, 최근성, 중요도, 유사도, 3가지의 합 리스트 구성
    prompt_list=[]
    for i in range(0,len(result_meta)):
        prompt_dic={}
        prompt_dic['timeStamp']=(result_meta[i]).get('timeStamp')
        prompt_dic['observation']=(result_meta[i]).get('observation')
        prompt_dic['recency']=0.0
        prompt_dic['importance']=(result_meta[i]).get('importance')
        prompt_dic['similarity']=1-result_distance[i]
        prompt_list.append(prompt_dic)        
    calculate_recency(prompt_list)
    calculate_priority(prompt_list)
    sorted_list = sorted(prompt_list, key=lambda x: x['priority'], reverse=True)
    print(sorted_list)
    return sorted_list
    
    
def calculate_recency(prompt_list):
    timestamp_list=[]
    for i in range(0,len(prompt_list)):
        timestamp_list.append(int(prompt_list[i].get('timeStamp')))
    max_value=max(timestamp_list)
    min_value=min(timestamp_list)
    for i in range(0,len(prompt_list)):
        normal_num=(timestamp_list[i]-min_value)/(max_value-min_value)
        (prompt_list[i])['recency']=normal_num
    print(prompt_list)
    return
    
def calculate_priority(prompt_list):
    for i in range(0,len(prompt_list)):
        (prompt_list[i])['priority']=prompt_list[i].get('recency') + prompt_list[i].get('importance')+ prompt_list[i].get('similarity')
    return         
