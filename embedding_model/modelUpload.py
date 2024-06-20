from sentence_transformers import SentenceTransformer

def model_upload():
    return SentenceTransformer("bespin-global/klue-sroberta-base-continue-learning-by-mnr")

