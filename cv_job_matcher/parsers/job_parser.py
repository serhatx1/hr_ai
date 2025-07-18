import re
from typing import Dict, Optional
from sentence_transformers import SentenceTransformer, util
import numpy as np

JOB_SECTION_HEADERS = {
    "about_company": (
        'about us', 'hakkımızda', 'company info', 'şirket hakkında', 'about the company', 'about',
    'who we are', 'biz kimiz', 'our story', 'hikayemiz', 'company overview', 'şirket genel bakış',
    'company profile', 'şirket profili', 'background', 'geçmiş', 'mission', 'misyon',
    'vision', 'vizyon', 'what we do', 'ne yapıyoruz', 'who we are as a company', 'şirket olarak biz kimiz'
    ),
    "mission": (
    'mission', 'misyon', 'görevimiz', 'amaç', 'hedefimiz', 'bizim görevimiz', 'bizim amacımız'
    ),
    "position": (
    'position', 'pozisyon', 'job title', 'unvan', 'title', 'rol', 'role', 'görev', 'mevki', 'statü'
    ),
    "required_skills": (
'required skills', 'zorunlu beceriler', 'aranan nitelikler', 'gerekli beceriler',
    'must have', 'must-have', 'must-have qualifications', 'must have qualifications',
    'requirements', 'qualifications', 'aranan özellikler', 'gerekli nitelikler',
    'requirements and skills', 'olmazsa olmaz'    ),
    "preferred_skills": (
'preferred skills', 'tercih edilen beceriler', 'nice to have', 'nice-to-have',
    'plus', 'artı', 'tercihen', 'tercih edilen nitelikler', 'tercih edilen özellikler',
    'nice-to-have / plus'    ),
    "soft_skills": (
'soft skills', 'kişisel yetkinlikler', 'kişisel beceriler', 'kişisel özellikler',
    'interpersonal skills', 'communication skills', 'teamwork', 'leadership',
    'problem solving', 'adaptability', 'analytical thinking'    ),

    "job_description": (
        'job description', 'iş tanımı', 'görevler', 'sorumluluklar', 'responsibilities', 'tasks', 'main duties', 'main responsibilities', 'what you will do', 'what will you do'
    ),

    "responsibilities": (
        'responsibilities', 'görevler', 'sorumluluklar', 'main responsibilities', 'main duties', 'what you will do', 'key responsibilities', 'ana sorumluluklar', 'primary responsibilities'
    ),
    
    "requirements": (
        'requirements', 'gereksinimler', 'aranan nitelikler', 'must have', 'qualifications', 'gerekli nitelikler', 'requirements and skills'
    ),

    "benefits": (
        'benefits', 'yan haklar', 'imkanlar', 'sunduklarımız', 'what we offer', 'offerings'
    ),
    
    "project_details": (
        'project duration', 'contract type', 'project type', 'duration', 'kontrat tipi', 'proje süresi', 'çalışma şekli', 'employment type', 'iş tipi', 'iş süresi', 'kontrat süresi', 'kontrat'
    ),
    "ideal_candidate": (
        'ideal aday', 'tercih edilen aday', 'aradığımız aday', 'en uygun aday', 'kimler başvurmalı',
        'who should apply', 'ideal candidate', 'preferred candidate', 'the right person', 'who you are', 'what we look for', 'candidate profile', 'aranan aday profili', 'aday profili', 'bizim için ideal aday'
    )
}

def find_header_with_transformer(line: str, all_headers: list, model, threshold: float = 0.7) -> Optional[str]:
    """
    Verilen satırı, header adayları ile karşılaştırır ve benzerliği en yüksek olanı döner.
    Eşik değerin altında ise None döner.
    """
    line_emb = model.encode(line, convert_to_tensor=True)
    headers_emb = model.encode(all_headers, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(line_emb, headers_emb)[0].cpu()
    best_idx = int(np.argmax(cos_scores))
    best_score = float(cos_scores[best_idx])
    if best_score >= threshold:
        return all_headers[best_idx]
    return None

# Sentence transformer ile header bulma destekli section extraction

def extract_job_sections_transformer(text: str, model_name: str = "paraphrase-MiniLM-L6-v2", threshold: float = 0.93) -> Dict[str, str]:
    model = SentenceTransformer(model_name)
    all_headers = []
    header_to_section = {}
    for section, headers in JOB_SECTION_HEADERS.items():
        for h in headers:
            all_headers.append(h)
            header_to_section[h.lower()] = section
    sections = {}
    current_section = "general"
    sections[current_section] = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        found_header = find_header_with_transformer(line, all_headers, model, threshold)
        if found_header:
            current_section = header_to_section.get(found_header.lower(), found_header)
            if current_section not in sections:
                sections[current_section] = ""
        else:
            sections[current_section] += line + "\n"
    return sections

def parse_job_posting(text: str) -> Dict[str, str]:
 
    return extract_job_sections_transformer(text)
