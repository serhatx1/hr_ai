import re
from difflib import SequenceMatcher
from cv_job_matcher.keywords import KEYWORDS

# Normalizasyon fonksiyonu
def normalize_keyword(kw):
    return re.sub(r'[^a-zA-Z0-9]', '', kw).lower()

# KEYWORDS listesini normalize edilmiş set olarak hazırla
NORMALIZED_KEYWORDS = [normalize_keyword(kw) for kw in KEYWORDS]
KEYWORD_MAP = {normalize_keyword(kw): kw for kw in KEYWORDS}  # normalize -> orijinal

# Benzerlik oranı fonksiyonu
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# En iyi whitelist eşleşmesini bul
def find_best_keyword_match(word, normalized_keywords, threshold=0.8):
    best_match = None
    best_score = 0
    for kw in normalized_keywords:
        score = similarity(word, kw)
        if score > best_score:
            best_score = score
            best_match = kw
    if best_score >= threshold:
        return best_match
    return None

def extract_keywords_from_job(job_sections, threshold=0.8):
    """
    İş tanımındaki önemli anahtar kelimeleri çıkarır.
    Sadece whitelist'te benzerlik oranı eşik değerini aşanları döndürür.
    """
    keywords = set()
    job_keyword_sections = [
        'required_skills', 'responsibilities', 'requirements',
        'preferred_skills', 'soft_skills', 'job_description'
    ]
    for section in job_keyword_sections:
        text = job_sections.get(section, '')
        for kw in re.split(r'[\,\n;•\-]', text):
            norm_kw = normalize_keyword(kw.strip())
            if norm_kw and len(norm_kw) > 1:
                match = find_best_keyword_match(norm_kw, NORMALIZED_KEYWORDS, threshold)
                if match:
                    keywords.add(match)
    return list(keywords)

def score_keyword_in_cv_progressive(keyword, cv_sections, threshold=0.8):
    """
    Anahtar kelimeyi CV'de arar ve progressive puanlama uygular.
    Benzerlik oranı eşik değerini aşan kelimeler için çalışır.
    """
    norm_keyword = normalize_keyword(keyword)
    score = 0
    # 3: skills kısmında geçiyorsa
    if find_best_keyword_match(norm_keyword, [normalize_keyword(w) for w in re.split(r'[\,\n;•\-]', cv_sections.get('skills', ''))], threshold) is not None:
        score = max(score, 3)
    # 5: work_and_employment kısmında detaylı geçiyorsa, daha önce 3 aldıysa +2 ekle
    if find_best_keyword_match(norm_keyword, [normalize_keyword(w) for w in re.split(r'[\,\n;•\-]', cv_sections.get('work_and_employment', ''))], threshold) is not None:
        if score == 3:
            score += 2  # +2 ekle
        else:
            score = max(score, 5)
    # 2: education_and_training, misc, accomplishments bölümlerinde geçiyorsa
    for sec in ['education_and_training', 'misc', 'accomplishments']:
        if find_best_keyword_match(norm_keyword, [normalize_keyword(w) for w in re.split(r'[\,\n;•\-]', cv_sections.get(sec, ''))], threshold) is not None:
            score = max(score, 2)
    # 1: herhangi bir yerde geçiyorsa, daha önce puan almadıysa
    if score == 0:
        for sec in cv_sections:
            if find_best_keyword_match(norm_keyword, [normalize_keyword(w) for w in re.split(r'[\,\n;•\-]', cv_sections.get(sec, ''))], threshold) is not None:
                score = 1
                break
    return score

def match_and_score(job_sections, cv_sections, threshold=0.8):
  
    keywords = extract_keywords_from_job(job_sections, threshold)
    total_score = 0
    keyword_scores = {}
    for kw in keywords:
        score = score_keyword_in_cv_progressive(kw, cv_sections, threshold)
        keyword_scores[KEYWORD_MAP[kw]] = score
        total_score += score
    return total_score, keyword_scores

def build_gemini_prompt(job_sections, cv_sections, keywords=None):

    if keywords is None:
        keywords = KEYWORDS
    prompt = (
        "Below are the sections of a job posting and a CV, a list of keywords, and scoring rules.\n"
        "For each keyword:\n"
        "- Assign a score (0, 1, 3, 5).\n"
        "- Indicate the source: 'work experience', 'skills', 'other' (e.g., if found in work experience section, use 'work experience'; if in skills, use 'skills'; if only in CV but not in those sections, use 'other'; if not found, use 'none').\n"
        "Additionally, evaluate education:\n"
        "- If the department(s) required by the job match the candidate's degree, add a department score.\n"
        "- Score the university based on the following tier list:\n"
        "  * Tier 1 (top universities): 10 points\n"
        "  * Tier 2: 8 points\n"
        "  * Tier 3: 5 points\n"
        "  * Tier 4: 3 points\n"
        "  * Tier 5: 2 points\n"
        "- Indicate the university's tier and whether the department matches.\n"
        "Also, evaluate work experience years:\n"
        "- Only consider full-time, professional work experience (do NOT include internships, education, part-time, or freelance jobs).\n"
        "- For each job in the CV's work experience section, extract the start and end dates (month and year).\n"
        "- For each job, calculate the duration in months by subtracting the start date from the end date. If the end date is 'Present' or 'Current', use the current date.\n"
        "- Sum up all durations to get the total professional work experience in years (round to the nearest half year).\n"
        "- If dates are missing or unclear, ignore that job.\n"
        "- Do NOT guess or assume experience if dates are not explicit.\n"
        "- Report the total professional work experience in years.\n"
        "- If the job posting specifies required years of experience, compare it to the candidate's total years of professional work experience.\n"
        "- Scoring for experience years:\n"
        "  * Exact match (e.g., 5 years required, 5 years in CV): 30 points\n"
        "  * Close match (e.g., 5 years required, 4 or 6 years in CV): 20 points\n"
        "  * Distant match (e.g., 5 years required, 2-3 or 7-8 years in CV): 5 points\n"
        "  * Very distant or missing (1 year or none): 0 points\n"
        "- Indicate both the required and found years, and the match type.\n"
        "Return a very short and clear JSON. Example format:\n"
        "{\n"
        "  \"keywords\": {\n"
        "    \"React\": {\"score\": 5, \"source\": \"work experience\"},\n"
        "    ...\n"
        "  },\n"
        "  \"education\": {\n"
        "    \"university\": \"Bogazici University\",\n"
        "    \"tier\": 1,\n"
        "    \"score\": 10,\n"
        "    \"department_match\": true,\n"
        "    \"department_score\": 5\n"
        "  },\n"
        "  \"experience_years\": {\n"
        "    \"required\": 5,\n"
        "    \"found\": 5,\n"
        "    \"match_type\": \"exact\",\n"
        "    \"score\": 30\n"
        "  },\n"
        "  \"total_score\": 65,\n"
        "  \"summary\": \"The candidate scored 65 points. The university and department are a great fit. Experience matches exactly. Technically strong, should be invited for an interview.\"\n"
        "}\n"
        "Also, add a short summary and a recommendation such as 'should be hired or not'.\n\n"
        f"Job Posting Sections:\n{job_sections}\n\n"
        f"CV Sections:\n{cv_sections}\n\n"
        f"Keywords:\n{keywords}\n"
        "University tier list example:\n"
        "Tier 1: Bogazici University, Middle East Technical University (METU/ODTU), Istanbul Technical University (ITU), Bilkent University, Koc University\n"
        "Tier 2: Sabanci University, Hacettepe University, Yildiz Technical University, Ankara University\n"
        "Tier 3: ...\n"
        "Tier 4: ...\n"
        "Tier 5: ...\n"
    )
    return prompt

def build_gemini_payload(job_sections, cv_sections, keywords=None):
 
    prompt = build_gemini_prompt(job_sections, cv_sections, keywords)
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
