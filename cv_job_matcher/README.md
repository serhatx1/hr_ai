# CV-Job Matcher: AI-Powered Recruitment Decision Support System

## Technologies Used

- **Python 3.x**  
  Core programming language for all modules.

- **FastAPI**  
  For building the RESTful API endpoints.

- **Uvicorn**  
  ASGI server for running FastAPI applications.

- **pdfplumber**  
  For extracting and parsing text from PDF files.

- **python-docx**  
  For extracting and parsing text from DOCX files.

- **sentence-transformers**  
  For semantic search, section header detection, and advanced text similarity (used in both CV and job description parsing).

- **scikit-learn**  
  For machine learning utilities and text processing.

- **NumPy**  
  For numerical operations and vector calculations.

- **requests**  
  For making HTTP requests, including to the Gemini 2.0 Flash API.

- **OpenAI**  
  (If used, for LLM integration or as an alternative to Gemini.)

- **Gemini 2.0 Flash API**  
  Large Language Model (LLM) integration for advanced analysis and suggestion generation.

- **Custom Keyword List**  
  For skill and technology extraction and matching.

- **difflib (SequenceMatcher)**  
  For fuzzy keyword and skill matching.

- **tempfile, os**  
  For secure file handling and uploads.

## Project Overview

CV-Job Matcher is an advanced AI-driven decision support system designed to streamline and objectify recruitment processes for human resources professionals. The system compares a candidate’s CV with a target job description, analyzes strengths and weaknesses, identifies missing skills, and provides optimization suggestions using a Large Language Model (LLM). The final output is a clear, data-driven “Hired” or “Not Hired” recommendation.

## Key Features

- Automated parsing of CVs and job descriptions (PDF and text formats supported)
- Skill extraction: identification of both technical and soft skills
- Keyword and requirement matching between candidate and job description
- Detection of missing or underdeveloped skills
- LLM-powered suggestions for candidate improvement
- Decision engine providing a definitive “Hired” or “Not Hired” result
- Modular, extensible Python codebase
- API-ready design for seamless integration with HR platforms

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/cv-job-matcher.git
cd cv-job-matcher
pip install -r requirements.txt
```

## Folder Structure

- `parsers/` : CV and job description parsers
- `extractors/` : Keyword and skill extractors
- `matching/` : Matching and analysis modules
- `llm/` : LLM (Gemini 2.0 Flash) integration
- `utils/` : Utility functions
- `tests/` : Test scenarios

## Usage

To run the main workflow:

```bash
python main.py
```

Upload the candidate’s CV and the job description. The system will analyze both documents and provide a hiring decision along with actionable suggestions.

## Why CV-Job Matcher?

- Objective and consistent decision-making, reducing human bias
- Comprehensive skill-based evaluation beyond simple keyword matching
- Actionable, AI-driven feedback for both candidates and HR professionals
- Efficient processing of large volumes of applications
- Designed for easy integration into existing HR systems via API

## Target Audience

- Human resources professionals
- Recruitment consultants
- Enterprises and startups
- Candidates seeking self-assessment 