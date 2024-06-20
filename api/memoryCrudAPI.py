from fastapi import APIRouter
import chromadb
from model.Item import Item
import embedding_model.modelUpload as modelUpload
from typing import List
import ast

client=chromadb.PersistentClient()

embed_model=modelUpload.model_upload()

memoryRouter = APIRouter()

@memoryRouter.post("/memory/add")
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
        print(collection.name+"에 저장되었습니다." )
        
        userId=infomations[i].userId
        timestamp=infomations[i].timestamp
        observation=infomations[i].observation
        importance=infomations[i].importance
        isEventScene=infomations[i].isEventScene
        reasonIds=infomations[i].reasonIds
        
        db_metadatas=[  #바꿔야 하는 부분
            {"userId":userId, "timestamp":timestamp, "observation": observation, "importance": importance,"isEventScene": isEventScene, "reasonIds": reasonIds}
        ]
        result_meta.append(db_metadatas)
        
        #DB 삽입
        collection.add(
            ids=id,
            metadatas=db_metadatas,
            embeddings=db_embedding
        )
        
    return result_meta;

@memoryRouter.patch("/memory/relocate/{userId}")
async def relocate_memory(userId: str):
    collection=client.get_collection(name=userId)
    collection_buffer=client.get_collection(name=userId+"_buffer")
    buffer_meta=get_all_memory_byId(collection_buffer)
    for i in range(0,len(buffer_meta)):
        (buffer_meta[i])["isEventScene"]=False
    #print(buffer_meta)
    result_meta=add_memory2(collection,buffer_meta)
    delete_memory2(userId+"_buffer")
    return result_meta
    
def add_memory2(collection,metalist):
    result_meta=[]
    #print(metalist)
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
            
        if(i==0):
            add_ids=int(id)-1
        
        print(id)
        print("Buffer -> Memory 저장되었습니다.")
        
        userId=(metalist[i])["userId"]
        timestamp=(metalist[i])["timestamp"]
        observation=(metalist[i])["observation"]
        importance=(metalist[i])["importance"]
        isEventScene=(metalist[i])["isEventScene"]
        reasonIds=(metalist[i])["reasonIds"]
        
        if(reasonIds!="null"):
            print(reasonIds)
            int_list = ast.literal_eval(reasonIds)
            int_list=list(int_list)
            for j in range(0,len(int_list)):
                int_list[j]=int_list[j]+add_ids
            reasonIds=str(int_list) 
        
        db_metadatas=[  #바꿔야 하는 부분
            {"userId":userId, "timestamp":timestamp, "observation": observation, "importance": importance,"isEventScene" : isEventScene ,"reasonIds": reasonIds}
        ]
        result_meta.append(db_metadatas)
        
        #DB 삽입
        collection.add(
            ids=id,
            metadatas=db_metadatas,
            embeddings=db_embedding
        )
    return result_meta
        
@memoryRouter.get("/memory/get/buffer/{userid}")
async def get_memory(userid: str):
    collection=client.get_collection(name=userid+"_buffer")
    return get_all_memory_byId(collection)

@memoryRouter.get("/memory/get/id/{userId}/{memoryId}")
async def get_memory_byId(userId: str, memoryId: str):
    collection=client.get_collection(name=userId)
    ids=[]
    ids.append(memoryId)
    result=collection.get(ids=ids)
    return result["metadatas"]

@memoryRouter.get("/memory/get/{query}/{userid}/{count}")
async def get_memory(query: str, userid: str, count: int):
    collection=client.get_collection(name=userid)
    n_result=collection.count() #결과 출력 개수
    print(n_result)
    if(n_result==0):
        return 410
        
    #쿼리 임베딩
    print(query)
    query_embedding_word=[query] #바꿔야 하는 부분
    query_embedding=embed_model.encode(query_embedding_word)
    query_embedding=query_embedding.tolist()
    
    result=collection.query(
        query_embeddings=query_embedding[0],
        n_results=n_result,
    )
    #print(result)
    
    #2가지의 합
    to_prompt_list=calculate(result,count)
    
    return to_prompt_list

def calculate(result_list,count):
    result_meta=(result_list["metadatas"])[0]
    #print(result_meta)
    #print(len(result_meta))
    result_distance=(result_list["distances"])[0]
    result_ids=(result_list["ids"])[0]
    #print(result_ids)
    #print(result_distance)
    
    #시간, 행동, 최근성, 중요도, 유사도, 3가지의 합 리스트 구성
    prompt_list=[]
    for i in range(0,len(result_meta)):
        prompt_dic={}
        prompt_dic['timestamp']=(result_meta[i]).get('timestamp')
        prompt_dic['observation']=(result_meta[i]).get('observation')
        prompt_dic['recency']=0.0;
        prompt_dic['ids']=(result_ids[i])
        prompt_dic['importance']=(result_meta[i]).get('importance')
        prompt_dic['similarity']=1-result_distance[i]
        prompt_dic['reasonIds']=(result_meta[i]).get('reasonIds')
        prompt_list.append(prompt_dic)        
    calculate_recency(prompt_list)
    calculate_priority(prompt_list)
    sorted_list = sorted(prompt_list, key=lambda x: x['priority'], reverse=True)
    sorted_count_list=[]
    for i in range(0,count):
        if(i == len(sorted_list)):
            break
        sorted_count_list.append(sorted_list[i])
    sorted_recency_list = sorted(sorted_count_list,key=lambda x: x['recency'])
    print(sorted_recency_list)
    return sorted_recency_list
    
    
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
        if(min_value==max_value):
            normal_num=1.0
        else:
            normal_num=0.7+(ids_list[i]-min_value)*0.3/(max_value-min_value)
        (prompt_list[i])['recency']=normal_num    
    return
    
def calculate_priority(prompt_list):
    for i in range(0,len(prompt_list)):
        (prompt_list[i])['priority']= prompt_list[i].get('importance')+ prompt_list[i].get('similarity')+prompt_list[i].get('recency')
    return         


@memoryRouter.delete("/memory/delete/all/{userid}/{start}/{end}")
async def delete_memory(userid: str, start: int, end: int):
    collection=client.get_collection(name=userid)
    ids=[]
    for i in range(start,end+1):
        ids.append(str(i))
    #print(ids)
    collection.delete(ids=ids)
    return 200
    

@memoryRouter.delete("/memory/delete/all/{userid}")
async def delete_memory(userid: str):
    collection=client.get_collection(name=userid)
    ids=[]
    for i in range(1,get_ids_max(collection)+1):
        ids.append(str(i))
    #print(ids)
    collection.delete(ids=ids)
    return 200

@memoryRouter.delete("/memory/delete/buffer/all/{userId}")
async def delete_buffer_memory(userId: str):
    collection=client.get_collection(name=userId+"_buffer")
    ids=[]
    for i in range(1,get_ids_max(collection)+1):
        ids.append(str(i))
    print("버퍼 삭제 리스트:")
    print(ids)
    collection.delete(ids=ids)
    return 200

def delete_memory2(userid):
    collection=client.get_collection(name=userid)
    ids=[]
    for i in range(1,get_ids_max(collection)+1):
        ids.append(str(i))
    print("버퍼 삭제 리스트:")
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

def get_all_memory_byId(collection):
    n_result=collection.count()
    if(n_result==0):
        return 410
    # query_embedding_word=[" "] #바꿔야 하는 부분
    # query_embedding=embed_model.encode(query_embedding_word)
    # query_embedding=query_embedding.tolist()
    # result=collection.query(
    #         query_embeddings=query_embedding[0],
    #         n_results=n_result
    # )
    #ids=(result["ids"])[0]
    #print(ids)
    ids=[]
    result=[]
    for i in range(1,n_result+1):
        ids.append(str(i))
        relist=collection.get(ids=ids)
        ids.clear()
        #print(relist)
        result.append((relist['metadatas'])[0])
    #print(result)
    return result