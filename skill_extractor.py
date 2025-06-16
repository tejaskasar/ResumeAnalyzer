import spacy

nlp = spacy.load("en_core_web_sm")

def extract_skills(text):
    doc = nlp(text)
    skills = []
    for token in doc:
        if token.pos_ == "NOUN" and len(token.text) > 2:
            skills.append(token.text.lower())
    return list(set(skills))
