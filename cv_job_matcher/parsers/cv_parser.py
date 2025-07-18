import os
from typing import Dict, Any, Optional
import pdfplumber
import docx
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np

SECTION_HEADERS = {
    "objective": (
        'career goal', 'objective', 'career objective', 'employment objective', 'professional objective',
        'summary', 'summary of qualifications', 'amaç', 'hedef', 'kariyer hedefi', 'özgeçmiş özeti'
    ),
    "work_and_employment": (
        'employment history', 'employment data', 'career summary', 'work history', 'work experience', 'experience',
        'professional experience', 'professional background', 'professional employment', 'additional experience',
        'career related experience', "professional employment history", 'related experience', 'programming experience',
        'freelance', 'freelance experience', 'army experience', 'military experience', 'military background',
        'iş deneyimi', 'deneyim', 'profesyonel deneyim', 'çalışma geçmişi'
    ),
    "education_and_training": (
        'academic background', 'academic experience', 'programs', 'courses', 'related courses', 'education',
        'educational background', 'educational qualifications', 'educational training', 'education and training',
        'training', 'academic training', 'professional training', 'course project experience', 'related course projects',
        'internship experience', 'internships', 'apprenticeships', 'college activities', 'certifications',
        'special training', 'eğitim', 'staj', 'sertifikalar', 'kurslar', 'akademik geçmiş'
    ),
    "skills": (
        'credentials', 'qualifications', 'areas of experience', 'areas of expertise', 'areas of knowledge', 'skills',
        "other skills", "other abilities", 'digital skills', 'career related skills', 'professional skills',
        'specialized skills', 'technical skills', 'computer skills', 'personal skills', 'computer knowledge',
        'technologies', 'technical experience', 'proficiencies', 'languages', 'language competencies and skills',
        'programming languages', 'competencies', 'yetenekler', 'beceriler', 'diller', 'teknik beceriler'
    ),
    "misc": (
        'activities and honors', 'activities', 'affiliations', 'professional affiliations', 'associations',
        'professional associations', 'memberships', 'professional memberships', 'athletic involvement',
        'community involvement', 'refere', 'civic activities', 'extra-Curricular activities', 'professional activities',
        'volunteer work', 'volunteer experience', 'additional information', 'interests', 'ilgi alanları', 'üyelikler'
    ),
    "accomplishments": (
        'achievement', 'awards and achievements', 'licenses', 'presentations', 'conference presentations', 'conventions',
        'dissertations', 'exhibits', 'papers', 'publications', 'professional publications', 'research experience',
        'research grants', 'project', 'research projects', 'personal projects', 'current research interests', 'thesis',
        'theses', 'başarılar', 'ödüller', 'projeler', 'yayınlar', 'araştırma'
    )
}

def parse_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def parse_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def parse_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def detect_filetype(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return 'pdf'
    elif ext == '.docx':
        return 'docx'
    elif ext in ['.txt', '.text']:
        return 'txt'
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def find_header_with_transformer(line: str, all_headers: list, model, threshold: float = 0.93) -> Optional[str]:

    line_emb = model.encode(line, convert_to_tensor=True)
    headers_emb = model.encode(all_headers, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(line_emb, headers_emb)[0].cpu()
    best_idx = int(np.argmax(cos_scores))
    best_score = float(cos_scores[best_idx])
    if best_score >= threshold:
        return all_headers[best_idx]
    return None

def extract_sections_transformer(text: str, model_name: str = "paraphrase-MiniLM-L6-v2", threshold: float = 0.7) -> Dict[str, str]:
    model = SentenceTransformer(model_name)
    all_headers = []
    header_to_section = {}
    for section, headers in SECTION_HEADERS.items():
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

def parse_cv(file_path: str) -> Dict[str, Any]:
    filetype = detect_filetype(file_path)
    if filetype == 'pdf':
        text = parse_pdf(file_path)
    elif filetype == 'docx':
        text = parse_docx(file_path)
    elif filetype == 'txt':
        text = parse_txt(file_path)
    else:
        raise ValueError("Unsupported file type")
    sections = extract_sections_transformer(text)
    return {
        'raw_text': text,
        'sections': sections
    }
