from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from cv_job_matcher.parsers.cv_parser import parse_cv
from cv_job_matcher.parsers.job_parser import parse_job_posting
from cv_job_matcher.matching.matcher import match_and_score
from cv_job_matcher.utils.file_utils import save_upload_file
from cv_job_matcher.matching.matcher import build_gemini_payload
from cv_job_matcher.llm.gemini_client import call_gemini_flash_api

app = FastAPI()

class MatchRequest(BaseModel):
    job_sections: dict
    cv_sections: dict

@app.post("/upload_cv")
async def upload_cv(file: UploadFile = File(...)):
    file_path = await save_upload_file(file)
    result = parse_cv(file_path)
    return result

@app.post("/upload_job")
async def upload_job(text: str = Form(...)):
    try:
        result = parse_job_posting(text)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    return result

@app.post("/match")
def match(request: MatchRequest):
    cv_sections = request.cv_sections.get("sections", request.cv_sections)
    job_sections = request.job_sections.get("sections", request.job_sections)
    payload = build_gemini_payload(job_sections, cv_sections)
    llm_response = call_gemini_flash_api(payload)
    return llm_response

if __name__ == "__main__":
    uvicorn.run("cv_job_matcher.main:app", host="0.0.0.0", port=8152, reload=True)
