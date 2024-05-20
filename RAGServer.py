import chromadb
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import modelUpload
import numpy as np
from typing import List

#uvicorn RAGServer:app --reload --host=0.0.0.0 --port=8800
#python -m uvicorn FastApi:app --reload

app=FastAPI()

client=chromadb.PersistentClient()

embed_model=modelUpload.model_upload()

class Item(BaseModel):
    userId: str
    timestamp: int
    observation: str
    importance: float
    isEventScene: bool
    
class User(BaseModel):
    userId: str
    
@app.post("/make/collection")
async def make_collection(user: User):
    client.create_collection(name=user.userId, metadata={"hnsw:space":"cosine"})
    client.create_collection(name=user.userId+"_buffer",metadata={"hnsw:space":"cosine"})
    return 200; 

@app.post("/memory/add")
async def add_memory(infomations: List[Item]):
    if(infomations[0].isEventScene):
        collection=client.get_collection(name=infomations[0].userId+"_buffer")
    else:
        collection=client.get_collection(name=infomations[0].userId)
    result_meta=[]
    for i in range(0,len(infomations)):
        #기억 정보 임베딩
        db_embedding_word=[infomations[i].observation]
        embeddings=embed_model.encode(db_embedding_word)
        db_embedding=embeddings.tolist()
        
        #id 새로운 방법 없으면 1부터, 있으면 최대 ids 찾아서 그 다음 id 부여
        n_result=collection.count()
        if(n_result!=0):
            id=str(get_ids_max(collection)+1)
        else:
            id=str(1)
            
        print(id)
        
        userId=infomations[i].userId
        timestamp=infomations[i].timestamp
        observation=infomations[i].observation
        importance=infomations[i].importance
        
        db_metadatas=[  #바꿔야 하는 부분
            {"userId":userId, "timestamp":timestamp, "observation": observation, "importance": importance}
        ]
        result_meta.append(db_metadatas)
        
        #DB 삽입
        collection.add(
            ids=id,
            metadatas=db_metadatas,
            embeddings=db_embedding
        )
        
    return result_meta;
    
@app.patch("/memory/relocate/{userId}")
async def relocate_memory(userId: str):
    collection=client.get_collection(name=userId)
    collection_buffer=client.get_collection(name=userId+"_buffer")
    buffer_memory=get_all_memory(collection_buffer)
    buffer_meta=(buffer_memory["metadatas"])[0]
    
    for i in range(0,len(buffer_meta)):
        (buffer_meta[i])["isEventScene"]=False
    result_meta=add_memory2(collection,buffer_meta)
    delete_memory2(userId+"_buffer")
    # for i in range(0,len(buffer_memory)):
    #     buffer_meta.append(buffer_memory[i])
    # result=collection_buffer.get(
    #     ids=buffer_id
    # )
    return result_meta
    
def add_memory2(collection,metalist):
    result_meta=[]
    print(metalist)
    for i in range(0,len(metalist)):
        #기억 정보 임베딩
        db_embedding_word=[(metalist[i])["observation"]]
        embeddings=embed_model.encode(db_embedding_word)
        db_embedding=embeddings.tolist()
        
        #id 새로운 방법 없으면 1부터, 있으면 최대 ids 찾아서 그 다음 id 부여
        n_result=collection.count()
        if(n_result!=0):
            id=str(get_ids_max(collection)+1)
        else:
            id=str(1)
            
        print(id)
        
        userId=(metalist[i])["userId"]
        timestamp=(metalist[i])["timestamp"]
        observation=(metalist[i])["observation"]
        importance=(metalist[i])["importance"]
        
        db_metadatas=[  #바꿔야 하는 부분
            {"userId":userId, "timestamp":timestamp, "observation": observation, "importance": importance}
        ]
        result_meta.append(db_metadatas)
        
        #DB 삽입
        collection.add(
            ids=id,
            metadatas=db_metadatas,
            embeddings=db_embedding
        )
    return result_meta
        
@app.get("/memory/get/{query}/{userid}")
async def get_memory(query: str, userid: str):
    collection=client.get_collection(name=userid)
    n_result=collection.count() #결과 출력 개수
    print(n_result)
    if(n_result==0):
        return 410
    elif(n_result==1):
        return 420
        
    #쿼리 임베딩
    print(query)
    query_embedding_word=[query] #바꿔야 하는 부분
    query_embedding=embed_model.encode(query_embedding_word)
    query_embedding=query_embedding.tolist()
    
    result=collection.query(
        query_embeddings=query_embedding[0],
        n_results=n_result,
    )
    print(result)
    
    #3가지의 합
    to_prompt_list=calculate(result)
    
    return to_prompt_list

def calculate(result_list):
    result_meta=(result_list["metadatas"])[0]
    print(result_meta)
    print(len(result_meta))
    result_distance=(result_list["distances"])[0]
    result_ids=(result_list["ids"])[0]
    print(result_ids)
    print(result_distance)
    
    #시간, 행동, 최근성, 중요도, 유사도, 3가지의 합 리스트 구성
    prompt_list=[]
    for i in range(0,len(result_meta)):
        prompt_dic={}
        prompt_dic['timestamp']=(result_meta[i]).get('timestamp')
        prompt_dic['observation']=(result_meta[i]).get('observation')
        prompt_dic['recency']=0.0
        prompt_dic['ids']=(result_ids[i])
        prompt_dic['importance']=(result_meta[i]).get('importance')
        prompt_dic['similarity']=1-result_distance[i]
        prompt_list.append(prompt_dic)        
    calculate_recency(prompt_list)
    calculate_priority(prompt_list)
    sorted_list = sorted(prompt_list, key=lambda x: x['priority'], reverse=True)
    print(sorted_list)
    return sorted_list
    
    
def calculate_recency(prompt_list):
    # timestamp 방식
    # timestamp_list=[]
    # for i in range(0,len(prompt_list)):
    #     timestamp_list.append(prompt_list[i].get('timestamp'))
    # max_value=max(timestamp_list)
    # min_value=min(timestamp_list)
    # for i in range(0,len(prompt_list)):
    #     normal_num=(timestamp_list[i]-min_value)/(max_value-min_value)
    #     (prompt_list[i])['recency']=normal_num
    # print(prompt_list)
    
    #ids 방식
    ids_list=[]
    for i in range(0,len(prompt_list)):
        ids_list.append(int(prompt_list[i].get('ids')))
    max_value=max(ids_list)
    min_value=min(ids_list)
    for i in range(0,len(prompt_list)):
        normal_num=(ids_list[i]-min_value)/(max_value-min_value)
        (prompt_list[i])['recency']=normal_num    
    return
    
def calculate_priority(prompt_list):
    for i in range(0,len(prompt_list)):
        (prompt_list[i])['priority']=prompt_list[i].get('recency') + prompt_list[i].get('importance')+ prompt_list[i].get('similarity')
    return         


@app.delete("/memory/delete/all/{userid}/{start}/{end}")
async def delete_memory(userid: str, start: int, end: int):
    collection=client.get_collection(name=userid)
    ids=[]
    for i in range(start,end+1):
        ids.append(str(i))
    print(ids)
    collection.delete(ids=ids)
    return 200
    

@app.delete("/memory/delete/all/{userid}")
async def delete_memory(userid: str):
    collection=client.get_collection(name=userid)
    ids=[]
    for i in range(1,get_ids_max(collection)+1):
        ids.append(str(i))
    print(ids)
    collection.delete(ids=ids)
    return 200

def delete_memory2(userid):
    collection=client.get_collection(name=userid)
    ids=[]
    for i in range(1,get_ids_max(collection)+1):
        ids.append(str(i))
    print(ids)
    collection.delete(ids=ids)
    return 200

#기억에서 ids 최대값 찾아서 int로 반환
def get_ids_max(collection):
    n_result=collection.count()
    query_embedding_word=[" "] #바꿔야 하는 부분
    query_embedding=embed_model.encode(query_embedding_word)
    query_embedding=query_embedding.tolist()
    result=collection.query(
            query_embeddings=query_embedding[0],
            n_results=n_result
    )
    result_ids=(result["ids"])[0]
    id_int=[]
    for j in range(0,len(result_ids)):
        id_int.append(int(result_ids[j]))
    return max(id_int)

def get_all_memory(collection):
    n_result=collection.count()
    query_embedding_word=[" "] #바꿔야 하는 부분
    query_embedding=embed_model.encode(query_embedding_word)
    query_embedding=query_embedding.tolist()
    result=collection.query(
            query_embeddings=query_embedding[0],
            n_results=n_result
    )
    return result