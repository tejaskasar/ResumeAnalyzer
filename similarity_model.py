from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(jd_text, resumes):
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    resume_embs = model.encode(resumes, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(jd_emb, resume_embs).cpu().numpy()[0]
    return similarities
