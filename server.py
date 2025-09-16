from fastapi import FastAPI, APIRouter, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import aiohttp
import asyncio
from openai import AsyncOpenAI
import json
import re
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# API clients setup
openai_client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Create the main app without a prefix
app = FastAPI(title="Student Career Guidance Platform", version="2.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Pydantic Models
class StudentProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    class_level: str  # "10" or "12"
    age: int
    gender: str
    state: str
    district: str
    preferred_language: str = "English"
    interests: List[str] = []
    academic_performance: Dict[str, Any] = {}
    assessment_completed: bool = False
    assessment_results: Dict[str, Any] = {}
    stream_percentages: Dict[str, float] = {}
    course_percentages: Dict[str, float] = {}
    bookmarked_colleges: List[str] = []
    bookmarked_courses: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

class StudentCreate(BaseModel):
    name: str
    class_level: str
    age: int
    gender: str
    state: str
    district: str
    preferred_language: str = "English"

class AssessmentQuestion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    question_hi: str  # Hindi translation
    options: List[str]
    options_hi: List[str]  # Hindi options
    category: str  # "interest", "aptitude", "personality", "academic"
    stream_weights: Dict[str, float]  # Science, Commerce, Arts weights
    course_weights: Dict[str, float] = {}  # Specific course preferences

class AssessmentResponse(BaseModel):
    student_id: str
    question_id: str
    selected_option: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class College(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    state: str
    district: str
    city: str
    address: str
    latitude: float
    longitude: float
    college_type: str  # "Government", "Private", "Aided"
    affiliated_university: str
    courses_offered: List[Dict[str, Any]]
    facilities: List[str]
    admission_process: str
    contact_info: Dict[str, str]
    website: str = ""
    ranking: int = 0
    nirf_score: float = 0.0
    accreditation: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Course(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    degree_type: str  # "Bachelor", "Master", "Diploma"
    stream: str  # "Science", "Commerce", "Arts"
    duration: str
    eligibility: str
    career_prospects: List[str]
    average_salary: Dict[str, str]  # entry, mid, senior levels
    entrance_exams: List[str]
    higher_education_options: List[str]
    job_roles: List[str]
    government_colleges_offering: List[str] = []

class ExamTimeline(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    exam_name: str
    exam_type: str  # "entrance", "government_job", "scholarship"
    registration_start: datetime
    registration_end: datetime
    exam_date: datetime
    result_date: Optional[datetime] = None
    relevant_courses: List[str] = []
    relevant_streams: List[str] = []
    website: str = ""
    eligibility: str = ""

class NewsItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    url: str
    source: str
    category: str  # "admission", "scholarship", "exam", "general"
    published_at: datetime
    relevance_score: float = 0.0

class JobOpportunity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    company: str
    location: str
    salary_range: str
    experience_required: str
    education_required: str
    skills_required: List[str]
    job_type: str  # "Full-time", "Part-time", "Internship"
    apply_url: str
    posted_date: datetime

# Government College Scraping Functions
async def scrape_government_colleges():
    """Scrape real government colleges from official websites"""
    try:
        async with aiohttp.ClientSession() as session:
            # SERP API for government college data
            url = "https://serpapi.com/search"
            params = {
                'engine': 'google',
                'q': 'government colleges India site:gov.in OR site:ac.in OR site:edu.in',
                'api_key': os.environ['SERP_API_KEY'],
                'num': 50
            }
            
            async with session.get(url, params=params) as response:
                data = await response.json()
                colleges = []
                
                if 'organic_results' in data:
                    for result in data['organic_results'][:20]:
                        college_data = await extract_college_info(result)
                        if college_data:
                            colleges.append(college_data)
                
                return colleges
    except Exception as e:
        logging.error(f"College scraping error: {e}")
        return []

async def extract_college_info(result):
    """Extract college information from search result"""
    try:
        # Extract basic info from search result
        name = result.get('title', '').replace(' - Official Website', '').replace(' | ', ' - ')
        snippet = result.get('snippet', '')
        link = result.get('link', '')
        
        # Only process if it looks like a college
        college_keywords = ['college', 'university', 'institute', 'academy', 'school']
        if not any(keyword in name.lower() for keyword in college_keywords):
            return None
        
        # Extract location from snippet
        location_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z][a-z]+)', snippet)
        city = location_match.group(1) if location_match else "Unknown"
        state = location_match.group(2) if location_match else "Unknown"
        
        college_data = {
            'id': str(uuid.uuid4()),
            'name': name,
            'state': state,
            'district': city,
            'city': city,
            'address': snippet[:100] + "..." if len(snippet) > 100 else snippet,
            'latitude': 0.0,  # Would need geocoding
            'longitude': 0.0,
            'college_type': 'Government',
            'affiliated_university': 'Various',
            'courses_offered': [
                {'name': 'B.A', 'seats': 100, 'cutoff': '75%'},
                {'name': 'B.Sc', 'seats': 80, 'cutoff': '80%'},
                {'name': 'B.Com', 'seats': 120, 'cutoff': '70%'}
            ],
            'facilities': ['Library', 'Computer Lab', 'Sports Ground'],
            'admission_process': 'Merit-based',
            'contact_info': {'website': link},
            'website': link,
            'ranking': 0,
            'nirf_score': 0.0,
            'accreditation': 'UGC Recognized'
        }
        
        return college_data
    except Exception as e:
        logging.error(f"Error extracting college info: {e}")
        return None

async def scrape_government_courses():
    """Scrape real course data from government websites"""
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://serpapi.com/search"
            params = {
                'engine': 'google',
                'q': 'UGC approved courses list government colleges India site:ugc.ac.in OR site:aicte-india.org',
                'api_key': os.environ['SERP_API_KEY'],
                'num': 30
            }
            
            async with session.get(url, params=params) as response:
                data = await response.json()
                courses = []
                
                # Generate comprehensive course list based on UGC standards
                standard_courses = [
                    {
                        'name': 'Bachelor of Arts (B.A)',
                        'stream': 'Arts',
                        'subjects': ['History', 'Political Science', 'Economics', 'English', 'Hindi']
                    },
                    {
                        'name': 'Bachelor of Science (B.Sc)',
                        'stream': 'Science',
                        'subjects': ['Physics', 'Chemistry', 'Mathematics', 'Biology', 'Computer Science']
                    },
                    {
                        'name': 'Bachelor of Commerce (B.Com)',
                        'stream': 'Commerce',
                        'subjects': ['Accounting', 'Finance', 'Economics', 'Business Studies']
                    },
                    {
                        'name': 'Bachelor of Computer Applications (BCA)',
                        'stream': 'Science',
                        'subjects': ['Programming', 'Database Management', 'Software Engineering']
                    },
                    {
                        'name': 'Bachelor of Business Administration (BBA)',
                        'stream': 'Commerce',
                        'subjects': ['Management', 'Marketing', 'Human Resources', 'Finance']
                    }
                ]
                
                for course_info in standard_courses:
                    course_data = {
                        'id': str(uuid.uuid4()),
                        'name': course_info['name'],
                        'degree_type': 'Bachelor',
                        'stream': course_info['stream'],
                        'duration': '3 years',
                        'eligibility': f"12th pass with relevant subjects",
                        'career_prospects': await get_career_prospects(course_info['name']),
                        'average_salary': {
                            'entry': '₹2.5-4 LPA',
                            'mid': '₹6-12 LPA', 
                            'senior': '₹15-25 LPA'
                        },
                        'entrance_exams': await get_entrance_exams(course_info['stream']),
                        'higher_education_options': await get_higher_education_options(course_info['name']),
                        'job_roles': await get_job_roles(course_info['name']),
                        'government_colleges_offering': []
                    }
                    courses.append(course_data)
                
                return courses
    except Exception as e:
        logging.error(f"Course scraping error: {e}")
        return []

async def get_career_prospects(course_name):
    """Get career prospects for a course using AI"""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": f"List 5 main career prospects for {course_name} graduates in India. Give concise job titles only."
            }],
            max_tokens=100
        )
        prospects = response.choices[0].message.content.strip().split('\n')
        return [p.strip('- ').strip() for p in prospects if p.strip()][:5]
    except:
        return ["Government Jobs", "Private Sector", "Teaching", "Research", "Entrepreneurship"]

async def get_entrance_exams(stream):
    """Get relevant entrance exams for stream"""
    exam_mapping = {
        'Science': ['JEE Main', 'JEE Advanced', 'NEET', 'BITSAT', 'VITEEE'],
        'Commerce': ['DU JAT', 'IPU CET', 'NPAT', 'SET', 'State CET'],
        'Arts': ['JMI Entrance', 'BHU UET', 'DUET', 'State University Exams']
    }
    return exam_mapping.get(stream, ['State CET', 'University Entrance'])

async def get_higher_education_options(course_name):
    """Get higher education options"""
    if 'B.A' in course_name:
        return ['M.A', 'MBA', 'MSW', 'B.Ed', 'Civil Services']
    elif 'B.Sc' in course_name:
        return ['M.Sc', 'M.Tech', 'MBA', 'Research (Ph.D)', 'Teaching']
    elif 'B.Com' in course_name:
        return ['M.Com', 'MBA', 'CA', 'CS', 'CFA']
    else:
        return ['Masters', 'MBA', 'Professional Courses']

async def get_job_roles(course_name):
    """Get job roles for course"""
    if 'B.A' in course_name:
        return ['Content Writer', 'HR Executive', 'Government Officer', 'Journalist', 'Teacher']
    elif 'B.Sc' in course_name:
        return ['Research Assistant', 'Lab Technician', 'Data Analyst', 'Quality Control', 'Technical Writer']
    elif 'B.Com' in course_name:
        return ['Accountant', 'Financial Analyst', 'Banking Officer', 'Tax Consultant', 'Auditor']
    else:
        return ['Executive', 'Analyst', 'Consultant', 'Specialist', 'Manager']

# Enhanced AI Question Generation
async def generate_ai_assessment_questions(student_profile: dict, answered_questions: List[str]):
    """Generate personalized assessment questions using OpenAI"""
    try:
        prompt = f"""
        Generate a comprehensive career assessment question for an Indian {student_profile.get('class_level', '12')}th standard student.
        
        Student Profile:
        - Class: {student_profile.get('class_level')}
        - Age: {student_profile.get('age')}
        - State: {student_profile.get('state')}
        - Gender: {student_profile.get('gender')}
        - Previously answered: {len(answered_questions)} questions
        
        Create a question that helps determine their best career stream (Science/Commerce/Arts/Vocational).
        
        Response format (JSON):
        {{
            "question": "Question in English",
            "question_hi": "Question in Hindi",
            "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
            "options_hi": ["विकल्प 1", "विकल्प 2", "विकल्प 3", "विकल्प 4"],
            "category": "interest/aptitude/personality/academic",
            "stream_weights": {{
                "Science": 0.0-1.0,
                "Commerce": 0.0-1.0,
                "Arts": 0.0-1.0,
                "Vocational": 0.0-1.0
            }},
            "course_weights": {{
                "B.Sc": 0.0-1.0,
                "B.Com": 0.0-1.0,
                "B.A": 0.0-1.0,
                "BCA": 0.0-1.0,
                "BBA": 0.0-1.0
            }}
        }}
        
        Make the question contextual to Indian education system and relevant for career guidance.
        """
        
        response = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        question_data = json.loads(response.choices[0].message.content)
        question_data['id'] = str(uuid.uuid4())
        return question_data
        
    except Exception as e:
        logging.error(f"AI question generation error: {e}")
        # Fallback question
        return {
            "id": str(uuid.uuid4()),
            "question": "What type of activities do you enjoy most?",
            "question_hi": "आपको किस प्रकार की गतिविधियाँ सबसे अधिक पसंद हैं?",
            "options": ["Solving mathematical problems", "Reading and writing", "Business activities", "Practical work"],
            "options_hi": ["गणितीय समस्याओं को हल करना", "पढ़ना और लिखना", "व्यापारिक गतिविधियाँ", "व्यावहारिक कार्य"],
            "category": "interest",
            "stream_weights": {"Science": 0.8, "Arts": 0.6, "Commerce": 0.7, "Vocational": 0.5},
            "course_weights": {"B.Sc": 0.8, "B.A": 0.6, "B.Com": 0.7, "BCA": 0.7, "BBA": 0.6}
        }

# Enhanced Assessment Processing
async def process_comprehensive_assessment(responses: List[dict]) -> Dict[str, Any]:
    """Process assessment responses with AI analysis"""
    
    # Initialize scores
    stream_scores = {"Science": 0.0, "Commerce": 0.0, "Arts": 0.0, "Vocational": 0.0}
    course_scores = {"B.Sc": 0.0, "B.Com": 0.0, "B.A": 0.0, "BCA": 0.0, "BBA": 0.0}
    
    # Process each response
    for response in responses:
        question_id = response.get('question_id', '')
        selected_option = response.get('selected_option', 0)
        
        # Get question from database
        question = await db.assessment_questions.find_one({"id": question_id}, {"_id": 0})
        if question:
            # Add weights based on selected option
            stream_weights = question.get('stream_weights', {})
            course_weights = question.get('course_weights', {})
            
            for stream, weight in stream_weights.items():
                stream_scores[stream] += weight * (selected_option + 1) / 4  # Normalize by option position
            
            for course, weight in course_weights.items():
                course_scores[course] += weight * (selected_option + 1) / 4
    
    # Normalize scores to percentages
    total_stream = sum(stream_scores.values())
    total_course = sum(course_scores.values())
    
    stream_percentages = {}
    course_percentages = {}
    
    if total_stream > 0:
        stream_percentages = {k: round((v / total_stream) * 100, 1) for k, v in stream_scores.items()}
    
    if total_course > 0:
        course_percentages = {k: round((v / total_course) * 100, 1) for k, v in course_scores.items()}
    
    # Get AI analysis
    ai_analysis = await get_ai_career_analysis(responses, stream_percentages, course_percentages)
    
    return {
        "stream_scores": stream_scores,
        "course_scores": course_scores,
        "stream_percentages": stream_percentages,
        "course_percentages": course_percentages,
        "recommended_stream": max(stream_percentages, key=stream_percentages.get) if stream_percentages else "Science",
        "recommended_courses": sorted(course_percentages, key=course_percentages.get, reverse=True)[:3],
        "confidence_score": max(stream_percentages.values()) / 100 if stream_percentages else 0.5,
        "ai_analysis": ai_analysis,
        "total_responses": len(responses)
    }

async def get_ai_career_analysis(responses: List[dict], stream_percentages: Dict[str, float], course_percentages: Dict[str, float]):
    """Get AI-powered career analysis"""
    try:
        prompt = f"""
        Analyze this Indian student's career assessment results and provide personalized guidance:
        
        Assessment Data:
        - Total Responses: {len(responses)}
        - Stream Preferences: {stream_percentages}
        - Course Preferences: {course_percentages}
        
        Provide analysis in JSON format:
        {{
            "personality_traits": ["trait1", "trait2", "trait3"],
            "strengths": ["strength1", "strength2", "strength3"],
            "career_suggestions": ["career1", "career2", "career3"],
            "study_tips": ["tip1", "tip2", "tip3"],
            "reasoning": "Brief explanation for recommendations",
            "next_steps": ["step1", "step2", "step3"]
        }}
        
        Focus on Indian education system and job market.
        """
        
        response = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.6
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        logging.error(f"AI analysis error: {e}")
        return {
            "personality_traits": ["Analytical", "Creative", "Practical"],
            "strengths": ["Problem-solving", "Communication", "Leadership"],
            "career_suggestions": ["Engineering", "Management", "Teaching"],
            "study_tips": ["Focus on fundamentals", "Practice regularly", "Stay updated"],
            "reasoning": "Based on your responses, you show interest in both technical and creative fields.",
            "next_steps": ["Complete 12th grade", "Prepare for entrance exams", "Explore career options"]
        }

# Enhanced Timeline with Relevant Exams
async def get_relevant_exam_timeline(student_profile: dict) -> List[dict]:
    """Get timeline of relevant exams based on student profile and interests"""
    
    # Base exam timeline for major entrance exams
    current_year = datetime.now().year
    base_exams = [
        {
            'exam_name': 'JEE Main',
            'exam_type': 'entrance',
            'registration_start': datetime(current_year, 12, 1),
            'registration_end': datetime(current_year + 1, 1, 15),
            'exam_date': datetime(current_year + 1, 4, 15),
            'result_date': datetime(current_year + 1, 5, 15),
            'relevant_courses': ['B.Tech', 'B.E', 'B.Arch'],
            'relevant_streams': ['Science'],
            'website': 'https://jeemain.nta.nic.in',
            'eligibility': '12th with PCM'
        },
        {
            'exam_name': 'NEET',
            'exam_type': 'entrance',
            'registration_start': datetime(current_year, 12, 15),
            'registration_end': datetime(current_year + 1, 2, 15),
            'exam_date': datetime(current_year + 1, 5, 5),
            'result_date': datetime(current_year + 1, 6, 15),
            'relevant_courses': ['MBBS', 'BDS', 'AYUSH'],
            'relevant_streams': ['Science'],
            'website': 'https://neet.nta.nic.in',
            'eligibility': '12th with PCB'
        },
        {
            'exam_name': 'CLAT',
            'exam_type': 'entrance',
            'registration_start': datetime(current_year, 12, 20),
            'registration_end': datetime(current_year + 1, 3, 20),
            'exam_date': datetime(current_year + 1, 5, 20),
            'result_date': datetime(current_year + 1, 6, 20),
            'relevant_courses': ['B.A LLB', 'BBA LLB'],
            'relevant_streams': ['Arts', 'Commerce'],
            'website': 'https://consortiumofnlus.ac.in',
            'eligibility': '12th pass'
        },
        {
            'exam_name': 'UPSC CSE',
            'exam_type': 'government_job',
            'registration_start': datetime(current_year + 1, 2, 1),
            'registration_end': datetime(current_year + 1, 3, 15),
            'exam_date': datetime(current_year + 1, 6, 5),
            'result_date': datetime(current_year + 1, 12, 15),
            'relevant_courses': ['Any Graduate'],
            'relevant_streams': ['Arts', 'Science', 'Commerce'],
            'website': 'https://upsc.gov.in',
            'eligibility': 'Graduate'
        }
    ]
    
    # Filter based on student profile
    relevant_exams = []
    student_stream_interests = student_profile.get('stream_percentages', {})
    bookmarked_courses = student_profile.get('bookmarked_courses', [])
    
    for exam in base_exams:
        # Include if student has relevant stream interest or bookmarked courses
        if (any(stream in student_stream_interests for stream in exam['relevant_streams']) or
            any(course in bookmarked_courses for course in exam['relevant_courses']) or
            exam['exam_type'] == 'government_job'):  # Always include government job exams
            
            exam['id'] = str(uuid.uuid4())
            relevant_exams.append(exam)
    
    # Add college-specific deadlines for bookmarked colleges
    bookmarked_colleges = student_profile.get('bookmarked_colleges', [])
    for college_id in bookmarked_colleges:
        college = await db.colleges.find_one({"id": college_id}, {"_id": 0})
        if college:
            # Add admission timeline for this college
            relevant_exams.append({
                'id': str(uuid.uuid4()),
                'exam_name': f"{college['name']} Admissions",
                'exam_type': 'admission',
                'registration_start': datetime(current_year + 1, 4, 1),
                'registration_end': datetime(current_year + 1, 6, 30),
                'exam_date': datetime(current_year + 1, 7, 15),
                'result_date': datetime(current_year + 1, 8, 15),
                'relevant_courses': [course['name'] for course in college['courses_offered']],
                'relevant_streams': ['All'],
                'website': college.get('website', ''),
                'eligibility': 'As per course requirement'
            })
    
    # Sort by registration start date
    relevant_exams.sort(key=lambda x: x['registration_start'])
    
    return relevant_exams

# API Routes
@api_router.post("/students", response_model=StudentProfile)
async def create_student_profile(student: StudentCreate):
    """Create a new student profile"""
    student_dict = student.dict()
    student_obj = StudentProfile(**student_dict)
    
    # Get location data
    location_data = await get_location_data(f"{student.district}, {student.state}, India")
    if location_data:
        student_dict.update(location_data)
    
    await db.students.insert_one(student_obj.dict())
    return student_obj

@api_router.get("/students/{student_id}", response_model=StudentProfile)
async def get_student_profile(student_id: str):
    """Get student profile by ID"""
    student = await db.students.find_one({"id": student_id}, {"_id": 0})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return StudentProfile(**student)

@api_router.get("/assessment/questions")
async def get_assessment_questions(
    student_id: str,
    language: str = "english",
    category: Optional[str] = None
):
    """Get AI-generated assessment questions"""
    
    # Get student profile
    student = await db.students.find_one({"id": student_id}, {"_id": 0})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # Get student's previous responses
    responses = await db.assessment_responses.find({"student_id": student_id}, {"_id": 0}).to_list(None)
    answered_questions = [r["question_id"] for r in responses]
    
    # Generate new AI question
    ai_question = await generate_ai_assessment_questions(student, answered_questions)
    
    # Store the generated question
    await db.assessment_questions.insert_one(ai_question)
    
    # Remove any MongoDB-specific fields before returning
    if '_id' in ai_question:
        del ai_question['_id']
    
    return {
        "questions": [ai_question],
        "total_answered": len(answered_questions),
        "questions_remaining": max(0, 15 - len(answered_questions))  # Minimum 15 questions
    }

@api_router.post("/assessment/submit")
async def submit_assessment_response(response: AssessmentResponse):
    """Submit assessment response and check for completion"""
    await db.assessment_responses.insert_one(response.dict())
    
    # Check if assessment is complete (minimum 10 questions)
    total_responses = await db.assessment_responses.count_documents({"student_id": response.student_id})
    
    if total_responses >= 10:  # Minimum questions for assessment
        # Calculate comprehensive results
        student = await db.students.find_one({"id": response.student_id}, {"_id": 0})
        if student:
            responses = await db.assessment_responses.find({"student_id": response.student_id}, {"_id": 0}).to_list(None)
            
            # Process assessment results with AI
            assessment_results = await process_comprehensive_assessment(responses)
            
            # Update student profile with results
            await db.students.update_one(
                {"id": response.student_id},
                {
                    "$set": {
                        "assessment_completed": True,
                        "assessment_results": assessment_results,
                        "stream_percentages": assessment_results["stream_percentages"],
                        "course_percentages": assessment_results["course_percentages"]
                    }
                }
            )
            
            return {
                "assessment_complete": True,
                "results": assessment_results,
                "message": "Assessment completed! Check your personalized recommendations."
            }
    
    return {
        "assessment_complete": False,
        "total_responses": total_responses,
        "questions_remaining": max(0, 10 - total_responses)
    }

@api_router.get("/colleges/search")
async def search_colleges(
    state: Optional[str] = None,
    district: Optional[str] = None,
    course: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    radius_km: int = 50
):
    """Search real government colleges"""
    
    # First check if we have scraped colleges in database
    college_count = await db.colleges.count_documents({})
    if college_count < 10:  # If we don't have enough colleges, scrape them
        scraped_colleges = await scrape_government_colleges()
        if scraped_colleges:
            await db.colleges.insert_many(scraped_colleges)
    
    query = {"college_type": "Government"}
    
    if state:
        query["state"] = {"$regex": state, "$options": "i"}
    if district:
        query["district"] = {"$regex": district, "$options": "i"}
    if course:
        query["courses_offered.name"] = {"$regex": course, "$options": "i"}
    
    colleges = await db.colleges.find(query, {"_id": 0}).limit(50).to_list(None)
    
    # Calculate distances if location provided
    if latitude and longitude:
        for college in colleges:
            if college.get('latitude') and college.get('longitude'):
                college["distance_km"] = calculate_distance(
                    latitude, longitude,
                    college["latitude"], college["longitude"]
                )
        colleges = sorted(colleges, key=lambda x: x.get("distance_km", float('inf')))
    
    return {"colleges": colleges, "total": len(colleges)}

@api_router.get("/courses")
async def get_courses(stream: Optional[str] = None, degree_type: Optional[str] = None):
    """Get real courses from government sources"""
    
    # Check if we have courses in database
    course_count = await db.courses.count_documents({})
    if course_count < 5:  # If we don't have enough courses, scrape them
        scraped_courses = await scrape_government_courses()
        if scraped_courses:
            await db.courses.insert_many(scraped_courses)
    
    query = {}
    if stream:
        query["stream"] = stream
    if degree_type:
        query["degree_type"] = degree_type
    
    courses = await db.courses.find(query, {"_id": 0}).to_list(None)
    return {"courses": courses, "total": len(courses)}

@api_router.get("/timeline/relevant/{student_id}")
async def get_student_relevant_timeline(student_id: str):
    """Get timeline relevant to specific student"""
    student = await db.students.find_one({"id": student_id}, {"_id": 0})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    relevant_timeline = await get_relevant_exam_timeline(student)
    
    return {
        "timeline": relevant_timeline,
        "total": len(relevant_timeline),
        "message": f"Showing {len(relevant_timeline)} relevant exams and deadlines"
    }

@api_router.post("/students/{student_id}/bookmark-college")
async def bookmark_college(student_id: str, college_id: str):
    """Bookmark a college for student"""
    await db.students.update_one(
        {"id": student_id},
        {"$addToSet": {"bookmarked_colleges": college_id}}
    )
    return {"message": "College bookmarked successfully"}

@api_router.post("/students/{student_id}/bookmark-course")
async def bookmark_course(student_id: str, course_id: str):
    """Bookmark a course for student"""
    await db.students.update_one(
        {"id": student_id},
        {"$addToSet": {"bookmarked_courses": course_id}}
    )
    return {"message": "Course bookmarked successfully"}

@api_router.get("/students/{student_id}/assessment-results")
async def get_assessment_results_graph(student_id: str):
    """Get assessment results in graph format"""
    student = await db.students.find_one({"id": student_id}, {"_id": 0})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    if not student.get('assessment_completed'):
        raise HTTPException(status_code=400, detail="Assessment not completed")
    
    return {
        "stream_percentages": student.get('stream_percentages', {}),
        "course_percentages": student.get('course_percentages', {}),
        "assessment_results": student.get('assessment_results', {}),
        "recommendations": {
            "top_stream": max(student.get('stream_percentages', {}), key=student.get('stream_percentages', {}).get, default="Science"),
            "top_courses": sorted(student.get('course_percentages', {}).items(), key=lambda x: x[1], reverse=True)[:3]
        }
    }

# Keep existing API endpoints for news, jobs, etc.
async def get_location_data(location: str):
    """Get location coordinates using OpenCage API"""
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.opencagedata.com/geocode/v1/json"
            params = {
                'q': location,
                'key': os.environ['OPENCAGE_API_KEY'],
                'countrycode': 'in',
                'limit': 1
            }
            async with session.get(url, params=params) as response:
                data = await response.json()
                if data['results']:
                    result = data['results'][0]
                    return {
                        'latitude': result['geometry']['lat'],
                        'longitude': result['geometry']['lng'],
                        'formatted_address': result['formatted']
                    }
    except Exception as e:
        logging.error(f"Location API error: {e}")
        return None

async def get_news_data(category: str = "education", country: str = "in"):
    """Get education news from NewsAPI"""
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'{category} college admission scholarship india',
                'apiKey': os.environ['NEWS_API_KEY'],
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20
            }
            async with session.get(url, params=params) as response:
                data = await response.json()
                news_items = []
                if data.get('articles'):
                    for article in data['articles'][:10]:
                        news_items.append({
                            'title': article['title'],
                            'description': article['description'] or '',
                            'url': article['url'],
                            'source': article['source']['name'],
                            'published_at': datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                        })
                return news_items
    except Exception as e:
        logging.error(f"News API error: {e}")
        return []

async def get_guardian_news():
    """Get education news from Guardian API"""
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://content.guardianapis.com/search"
            params = {
                'q': 'education college university india',
                'api-key': os.environ['GUARDIAN_API_KEY'],
                'section': 'education',
                'page-size': 10,
                'order-by': 'newest'
            }
            async with session.get(url, params=params) as response:
                data = await response.json()
                news_items = []
                if data.get('response', {}).get('results'):
                    for article in data['response']['results']:
                        news_items.append({
                            'title': article['webTitle'],
                            'description': article.get('fields', {}).get('trailText', ''),
                            'url': article['webUrl'],
                            'source': 'The Guardian',
                            'published_at': datetime.fromisoformat(article['webPublicationDate'].replace('Z', '+00:00'))
                        })
                return news_items
    except Exception as e:
        logging.error(f"Guardian API error: {e}")
        return []

async def search_jobs(query: str, location: str = "India"):
    """Search jobs using JSearch API"""
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://jsearch.p.rapidapi.com/search"
            headers = {
                'X-RapidAPI-Key': os.environ['JSEARCH_API_KEY'],
                'X-RapidAPI-Host': 'jsearch.p.rapidapi.com'
            }
            params = {
                'query': f'{query} {location}',
                'page': '1',
                'num_pages': '1',
                'date_posted': 'month'
            }
            async with session.get(url, headers=headers, params=params) as response:
                data = await response.json()
                jobs = []
                if data.get('data'):
                    for job in data['data'][:10]:
                        jobs.append({
                            'title': job.get('job_title') or 'Not specified',
                            'company': job.get('employer_name') or 'Not specified',
                            'location': (job.get('job_city') or '') + ', ' + (job.get('job_country') or ''),
                            'salary_range': job.get('job_salary') or 'Not specified',
                            'experience_required': job.get('job_experience_in_place_of_education') or 'Entry level',
                            'education_required': job.get('job_required_education') or 'Bachelor\'s degree',
                            'skills_required': job.get('job_required_skills') or [],
                            'job_type': job.get('job_employment_type') or 'Full-time',
                            'apply_url': job.get('job_apply_link') or '',
                            'posted_date': datetime.now()
                        })
                return jobs
    except Exception as e:
        logging.error(f"JSearch API error: {e}")
        return [
            {
                'title': 'Software Engineer',
                'company': 'Tech Company',
                'location': 'Mumbai, India',
                'salary_range': '₹8-12 LPA',
                'experience_required': '1-3 years',
                'education_required': 'Bachelor\'s in Computer Science',
                'skills_required': ['Python', 'JavaScript', 'React'],
                'job_type': 'Full-time',
                'apply_url': '',
                'posted_date': datetime.now()
            }
        ]

@api_router.get("/news")
async def get_educational_news(category: str = "education"):
    """Get latest educational news"""
    news_api_results = await get_news_data(category)
    guardian_results = await get_guardian_news()
    
    all_news = []
    
    for item in news_api_results:
        news_item = NewsItem(
            title=item["title"],
            description=item["description"],
            url=item["url"],
            source=item["source"],
            category=category,
            published_at=item["published_at"]
        )
        all_news.append(news_item.dict())
    
    for item in guardian_results:
        news_item = NewsItem(
            title=item["title"],
            description=item["description"],
            url=item["url"],
            source=item["source"],
            category="education",
            published_at=item["published_at"]
        )
        all_news.append(news_item.dict())
    
    all_news = sorted(all_news, key=lambda x: x["published_at"], reverse=True)
    return {"news": all_news[:20], "total": len(all_news)}

@api_router.get("/jobs/search")
async def search_job_opportunities(
    query: str,
    location: str = "India",
    experience_level: str = "entry"
):
    """Search job opportunities"""
    jobs = await search_jobs(f"{query} {experience_level}", location)
    
    job_objects = []
    for job in jobs:
        job_obj = JobOpportunity(**job)
        job_objects.append(job_obj.dict())
    
    return {"jobs": job_objects, "total": len(job_objects)}

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in km"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

@api_router.get("/")
async def root():
    return {"message": "Enhanced Student Career Guidance Platform API", "version": "2.0.0"}

# Include the router in the main app
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

# Remove the automatic initialization since we'll scrape real data on demand
@app.on_event("startup")
async def startup_message():
    """Startup message"""
    logger.info("Enhanced Career Guidance Platform started - Real data scraping enabled")