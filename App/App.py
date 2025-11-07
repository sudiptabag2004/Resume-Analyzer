# AI-Powered Resume Analyzer with ML Models
# Enhanced version with spaCy NER and sklearn classification

import streamlit as st
import pandas as pd
import random
import time
import io
import re
from streamlit_tags import st_tags
from PIL import Image

# ML Libraries
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# PDF parsing libraries
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from io import StringIO

# pre stored data for prediction purposes
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos

import nltk
nltk.download('stopwords')

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load('en_core_web_sm')
        return nlp
    except:
        st.warning("Installing spaCy model... This will only happen once.")
        import os
        os.system('python -m spacy download en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
        return nlp

nlp = load_spacy_model()


###### ML-Enhanced Functions ######

def pdf_reader(file):
    """Reads PDF file and extracts text"""
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()

    converter.close()
    fake_file_handle.close()
    return text


def extract_name_from_pdf_by_fontsize(file):
    """Extract name by finding the largest font size text in the PDF"""
    try:
        from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTChar
        from pdfminer.pdfpage import PDFPage
        from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
        from pdfminer.converter import PDFPageAggregator
        
        resource_manager = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(resource_manager, laparams=laparams)
        interpreter = PDFPageInterpreter(resource_manager, device)
        
        text_with_sizes = []
        
        with open(file, 'rb') as fh:
            # Only process first page where name usually appears
            for page_num, page in enumerate(PDFPage.get_pages(fh)):
                if page_num > 0:  # Only check first page
                    break
                    
                interpreter.process_page(page)
                layout = device.get_result()
                
                for element in layout:
                    if isinstance(element, (LTTextBox, LTTextLine)):
                        for text_line in element:
                            if hasattr(text_line, '__iter__'):
                                for character in text_line:
                                    if isinstance(character, LTChar):
                                        text_with_sizes.append({
                                            'text': element.get_text().strip(),
                                            'size': character.height,
                                            'y_position': character.y0
                                        })
                                        break  # Only need one char to get size
                                break
        
        if text_with_sizes:
            # Sort by font size (descending) and y-position (top first)
            text_with_sizes.sort(key=lambda x: (x['size'], x['y_position']), reverse=True)
            
            # Get top candidates with largest font
            max_size = text_with_sizes[0]['size']
            candidates = [item for item in text_with_sizes if item['size'] >= max_size * 0.9]
            
            # Filter candidates to find the most likely name
            for candidate in candidates[:5]:  # Check top 5 largest text items
                text = candidate['text']
                words = text.split()
                text_lower = text.lower()
                
                # Check if it looks like a name
                if (2 <= len(words) <= 4 and
                    3 < len(text) < 50 and
                    not '@' in text and
                    not any(word.lower() in ['resume', 'cv', 'curriculum', 'phone', 'email', 'address'] for word in words)):
                    
                    # Clean up the text - remove special characters and extra parts
                    cleaned_text = re.split(r'[§|•·:\t]', text)[0].strip()
                    cleaned_words = cleaned_text.split()
                    
                    # Validate cleaned text
                    if (2 <= len(cleaned_words) <= 4 and
                        not any(char.isdigit() for char in cleaned_text)):
                        return cleaned_text
        
        return None
        
    except Exception as e:
        return None


def extract_entities_with_spacy(text):
    """Extract entities using spaCy NER (Named Entity Recognition)"""
    doc = nlp(text)
    
    entities = {
        'persons': [],
        'organizations': [],
        'locations': [],
        'dates': []
    }
    
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            entities['persons'].append(ent.text)
        elif ent.label_ == 'ORG':
            entities['organizations'].append(ent.text)
        elif ent.label_ in ['GPE', 'LOC']:
            entities['locations'].append(ent.text)
        elif ent.label_ == 'DATE':
            entities['dates'].append(ent.text)
    
    return entities


def extract_skills_ml(text):
    """Extract skills using ML-based approach with expanded skill database"""
    # Comprehensive skill database
    skill_database = {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'swift', 'kotlin', 'golang', 'rust', 'scala',
        'react', 'angular', 'vue', 'svelte', 'node', 'nodejs', 'django', 'flask', 'fastapi', 'spring', 'express', 'nextjs', 'nuxt',
        'html', 'css', 'sass', 'scss', 'tailwind', 'bootstrap', 'material ui', 'mui',
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'dynamodb', 'oracle', 'sqlite',
        'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible',
        'git', 'github', 'gitlab', 'bitbucket', 'agile', 'scrum', 'jira', 'confluence',
        'machine learning', 'deep learning', 'neural networks', 'tensorflow', 'keras', 'pytorch', 'scikit-learn', 'opencv',
        'data analysis', 'data science', 'data visualization', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
        'nlp', 'natural language processing', 'computer vision', 'reinforcement learning',
        'android', 'ios', 'flutter', 'react native', 'xamarin', 'swift ui', 'jetpack compose',
        'figma', 'adobe xd', 'sketch', 'photoshop', 'illustrator', 'after effects', 'premier pro',
        'excel', 'powerpoint', 'word', 'tableau', 'power bi', 'looker',
        'rest api', 'graphql', 'microservices', 'serverless', 'websockets',
        'testing', 'unit testing', 'pytest', 'jest', 'selenium', 'cypress',
        'linux', 'unix', 'bash', 'shell scripting', 'powershell'
    }
    
    text_lower = text.lower()
    doc = nlp(text_lower)
    
    # Extract skills using both keyword matching and noun phrases
    found_skills = set()
    
    # Keyword matching
    for skill in skill_database:
        if skill in text_lower:
            found_skills.add(skill.title())
    
    # Extract technical noun phrases (potential skills)
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        if chunk_text in skill_database:
            found_skills.add(chunk_text.title())
    
    return list(found_skills)


def detect_ai_generated_content(text):
    """Detect if resume content is AI-generated using multiple ML-based indicators"""
    
    doc = nlp(text)
    
    # Initialize scores
    ai_indicators = {
        'repetitive_patterns': 0,
        'perfect_grammar': 0,
        'generic_phrases': 0,
        'sentence_uniformity': 0,
        'buzzword_density': 0
    }
    
    # 1. Check for AI-common phrases and patterns
    ai_phrases = [
        'results-oriented professional', 'proven track record', 'dynamic individual',
        'team player', 'detail-oriented', 'self-motivated', 'fast-paced environment',
        'excellent communication skills', 'think outside the box', 'go above and beyond',
        'synergy', 'leverage', 'spearheaded', 'orchestrated', 'championed',
        'robust', 'cutting-edge', 'state-of-the-art', 'innovative solutions',
        'passionate about', 'dedicated professional', 'highly motivated'
    ]
    
    text_lower = text.lower()
    generic_count = sum(1 for phrase in ai_phrases if phrase in text_lower)
    ai_indicators['generic_phrases'] = min(generic_count / 5, 1.0)  # Normalize to 0-1
    
    # 2. Check sentence length uniformity (AI tends to generate uniform sentences)
    sentences = [sent.text for sent in doc.sents if len(sent.text.split()) > 3]
    if len(sentences) > 5:
        sentence_lengths = [len(sent.split()) for sent in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((x - avg_length) ** 2 for x in sentence_lengths) / len(sentence_lengths)
        # Low variance indicates uniform sentence lengths (AI characteristic)
        if variance < 20:
            ai_indicators['sentence_uniformity'] = 0.8
        elif variance < 40:
            ai_indicators['sentence_uniformity'] = 0.5
        else:
            ai_indicators['sentence_uniformity'] = 0.2
    
    # 3. Check for repetitive sentence structures
    sentence_starts = [sent.text.split()[0].lower() for sent in doc.sents if len(sent.text.split()) > 0]
    if len(sentence_starts) > 0:
        unique_starts = len(set(sentence_starts))
        repetition_ratio = 1 - (unique_starts / len(sentence_starts))
        ai_indicators['repetitive_patterns'] = repetition_ratio
    
    # 4. Check buzzword density (AI loves buzzwords)
    buzzwords = [
        'innovative', 'strategic', 'dynamic', 'synergy', 'leverage', 'optimize',
        'streamline', 'enhance', 'facilitate', 'implement', 'execute', 'deliver',
        'drive', 'transform', 'revolutionize', 'cutting-edge', 'scalable'
    ]
    
    words = text_lower.split()
    if len(words) > 0:
        buzzword_count = sum(1 for word in words if word in buzzwords)
        ai_indicators['buzzword_density'] = min((buzzword_count / len(words)) * 100, 1.0)
    
    # 5. Check for perfect grammar and structure (somewhat indicative)
    # Count sentences without common human errors
    total_sentences = len(list(doc.sents))
    if total_sentences > 0:
        # Simple heuristic: AI text tends to have consistent punctuation and capitalization
        proper_sentences = sum(1 for sent in doc.sents if sent.text[0].isupper() and sent.text.strip()[-1] in '.!?')
        ai_indicators['perfect_grammar'] = proper_sentences / total_sentences
    
    # Calculate overall AI probability
    weights = {
        'repetitive_patterns': 0.25,
        'perfect_grammar': 0.15,
        'generic_phrases': 0.30,
        'sentence_uniformity': 0.20,
        'buzzword_density': 0.10
    }
    
    ai_probability = sum(ai_indicators[key] * weights[key] for key in ai_indicators)
    
    return ai_probability, ai_indicators


def predict_career_field_ml(text, skills):
    """Predict career field using TF-IDF and cosine similarity"""
    
    # Define career field patterns with more keywords
    field_patterns = {
        'Data Science': [
            'machine learning', 'deep learning', 'data science', 'tensorflow', 'keras', 'pytorch',
            'pandas', 'numpy', 'scikit-learn', 'data analysis', 'statistics', 'neural networks',
            'predictive modeling', 'data mining', 'big data', 'hadoop', 'spark', 'nlp', 'computer vision'
        ],
        'Web Development': [
            'html', 'css', 'javascript', 'react', 'angular', 'vue', 'node', 'django', 'flask',
            'php', 'laravel', 'wordpress', 'web development', 'frontend', 'backend', 'full stack',
            'rest api', 'graphql', 'express', 'nextjs', 'typescript'
        ],
        'Mobile Development': [
            'android', 'ios', 'flutter', 'react native', 'swift', 'kotlin', 'java', 'mobile app',
            'xamarin', 'mobile development', 'app development', 'swift ui', 'jetpack compose'
        ],
        'UI/UX Design': [
            'ui', 'ux', 'user experience', 'user interface', 'figma', 'adobe xd', 'sketch',
            'wireframe', 'prototype', 'design', 'photoshop', 'illustrator', 'user research',
            'interaction design', 'visual design'
        ],
        'DevOps': [
            'docker', 'kubernetes', 'jenkins', 'ci/cd', 'aws', 'azure', 'gcp', 'terraform',
            'ansible', 'devops', 'cloud', 'infrastructure', 'monitoring', 'deployment'
        ],
        'Data Engineering': [
            'etl', 'data pipeline', 'airflow', 'kafka', 'spark', 'hadoop', 'data warehouse',
            'sql', 'nosql', 'data engineering', 'big data', 'data lake'
        ]
    }
    
    # Prepare documents for TF-IDF
    documents = []
    field_names = []
    
    for field, keywords in field_patterns.items():
        documents.append(' '.join(keywords))
        field_names.append(field)
    
    # Add resume text
    resume_text = text.lower() + ' ' + ' '.join([s.lower() for s in skills])
    documents.append(resume_text)
    
    # Calculate TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Calculate cosine similarity
    resume_vector = tfidf_matrix[-1]
    field_vectors = tfidf_matrix[:-1]
    
    similarities = cosine_similarity(resume_vector, field_vectors)[0]
    
    # Get the best matching field
    best_match_idx = np.argmax(similarities)
    confidence = similarities[best_match_idx]
    
    if confidence > 0.1:  # Threshold for confidence
        return field_names[best_match_idx], confidence
    else:
        return 'General', confidence


def extract_resume_info_ml(text):
    """Extract resume information using ML models"""
    resume_data = {
        'name': 'User',
        'email': '',
        'mobile_number': '',
        'skills': [],
        'degree': '',
        'no_of_pages': 1,
        'organizations': [],
        'locations': []
    }
    
    # Extract entities using spaCy NER
    entities = extract_entities_with_spacy(text)
    
    # Comprehensive list of Indian states and major cities to filter out
    indian_locations = {
        'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh', 'goa', 'gujarat',
        'haryana', 'himachal pradesh', 'jharkhand', 'karnataka', 'kerala', 'madhya pradesh',
        'maharashtra', 'manipur', 'meghalaya', 'mizoram', 'nagaland', 'odisha', 'punjab',
        'rajasthan', 'sikkim', 'tamil nadu', 'telangana', 'tripura', 'uttar pradesh', 'uttarakhand',
        'west bengal', 'delhi', 'mumbai', 'bangalore', 'hyderabad', 'chennai', 'kolkata', 'pune',
        'ahmedabad', 'jaipur', 'surat', 'lucknow', 'kanpur', 'nagpur', 'indore', 'bhopal', 'visakhapatnam',
        'andhra', 'pradesh', 'tamil', 'nadu', 'west', 'bengal', 'himachal', 'madhya', 'uttar', 'arunachal'
    }
    
    # Common location indicators to filter out
    location_indicators = {
        'city', 'state', 'country', 'street', 'avenue', 'road', 'pin', 'pincode', 
        'zip', 'postal', 'india', 'usa', 'uk', 'district', 'town', 'village'
    }
    
    # Get all location keywords in lowercase
    location_keywords = set([loc.lower() for loc in entities['locations']])
    location_keywords.update(indian_locations)
    
    # Try to find valid person name from spaCy entities
    
    
    # Extract organizations and locations
    resume_data['organizations'] = list(set(entities['organizations']))[:5]
    resume_data['locations'] = list(set(entities['locations']))[:3]
    
    # Post-process name: Clean up any remaining separators or unwanted text
    if resume_data['name'] != 'User':
        # Split by separators and take first valid part
        name_parts = re.split(r'[§|•·:\t]', resume_data['name'])
        for part in name_parts:
            part = part.strip()
            words = part.split()
            # Take the part that looks like a proper name (2-3 words, all capitalized)
            if (2 <= len(words) <= 3 and 
                all(word[0].isupper() for word in words) and 
                not any(char.isdigit() for char in part)):
                resume_data['name'] = part
                break
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        resume_data['email'] = emails[0]
    
    # Extract phone number with improved pattern and multiple attempts
    phone_patterns = [
        r'\+?\d[\d\s\-\(\)]{8,}\d',  # General pattern
        r'(?:\+91[\-\s]?)?[6789]\d{9}',  # Indian mobile pattern
        r'\b\d{10}\b',  # Simple 10-digit pattern
        r'\b\d{3}[\-\s]?\d{3}[\-\s]?\d{4}\b'  # US format
    ]
    
    phone_found = False
    for pattern in phone_patterns:
        phones = re.findall(pattern, text)
        if phones:
            for phone in phones:
                # Clean the phone number - keep only digits
                cleaned_phone = re.sub(r'[^\d]', '', phone)
                
                # Validate: should be 10-12 digits (10 for local, 12 for +91)
                if 10 <= len(cleaned_phone) <= 12:
                    # If it's 12 digits and starts with 91, remove country code
                    if len(cleaned_phone) == 12 and cleaned_phone.startswith('91'):
                        cleaned_phone = cleaned_phone[2:]
                    # If it's 11 digits and starts with 91, remove country code
                    elif len(cleaned_phone) == 11 and cleaned_phone.startswith('91'):
                        cleaned_phone = cleaned_phone[2:]
                    
                    # Final validation: 10 digits starting with 6, 7, 8, or 9 (Indian mobile)
                    if len(cleaned_phone) == 10 and cleaned_phone[0] in '6789':
                        resume_data['mobile_number'] = cleaned_phone
                        phone_found = True
                        break
                    # Or any 10-digit number
                    elif len(cleaned_phone) == 10:
                        resume_data['mobile_number'] = cleaned_phone
                        phone_found = True
                        break
        
        if phone_found:
            break
    
    # Extract skills using ML
    resume_data['skills'] = extract_skills_ml(text)
    
    # Extract degree keywords
    degree_keywords = ['bachelor', 'master', 'phd', 'b.tech', 'm.tech', 'bsc', 'msc', 'ba', 'ma', 'mba', 'bba', 'b.e', 'm.e']
    text_lower = text.lower()
    for keyword in degree_keywords:
        if keyword in text_lower:
            resume_data['degree'] = keyword.upper()
            break
    
    # Count pages
    resume_data['no_of_pages'] = max(1, text.count('\f') + 1)
    
    return resume_data


def course_recommender(course_list):
    """Recommend courses based on field"""
    st.subheader("**📚 Courses & Certificates Recommendations**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"**{c}.** [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course


###### Streamlit Configuration ######

st.set_page_config(
   page_title="AI Resume Analyzer",
   page_icon='🤖',
   layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 4rem;
        font-weight: 900;
        margin: 2rem 0;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    .info-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
        color: #2c3e50;
        font-size: 1.1rem;
        line-height: 1.8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .success-item {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
        font-size: 1rem;
    }
    
    .warning-item {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
        font-size: 1rem;
    }
    
    .ml-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    .level-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    
    .level-fresher {
        background-color: #d73b5c;
        color: white;
    }
    
    .level-intermediate {
        background-color: #1ed760;
        color: white;
    }
    
    .level-experienced {
        background-color: #fba171;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


###### Main Application ######

def run():
    
    # Main Title with Gradient
    st.markdown('<h1 class="main-title">🤖 AI Resume Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Powered by Machine Learning & Natural Language Processing</p>', unsafe_allow_html=True)
    
    st.sidebar.markdown("# 📋 About")
    st.sidebar.info("""
    **AI Resume Analyzer**
    
    🤖 **ML Models Used:**
    - spaCy NER (Named Entity Recognition)
    - TF-IDF Vectorization
    - Cosine Similarity Classification
    
    **Features:**
    - ✅ Smart Resume Analysis
    - 💡 AI Skill Extraction
    - 📚 Course Suggestions
    - 📊 Resume Score
    - 🎯 Career Field Prediction
    
    Built with Machine Learning
    """)
    
    # Main Content
    st.markdown("<br>", unsafe_allow_html=True)
    
    # File upload
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pdf_file = st.file_uploader("Choose your Resume (PDF format)", type=["pdf"])
    
    if pdf_file is not None:
        with st.spinner('🤖 AI Models are analyzing your resume... Please wait...'):
            time.sleep(2)
        
        # Save uploaded file temporarily
        import os
        os.makedirs('./Uploaded_Resumes', exist_ok=True)
        save_image_path = './Uploaded_Resumes/' + pdf_file.name
        with open(save_image_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        try:
            # Extract text from PDF
            resume_text = pdf_reader(save_image_path)
            
            # Extract resume data using ML models
            resume_data = extract_resume_info_ml(resume_text)
            
            if resume_data and resume_text:
                
                st.markdown("---")
                st.header("**📊 AI-Powered Resume Analysis**")
                st.success(f"👋 Hello **{resume_data['name']}**!")
                
                # Basic Info Section
                st.subheader("**👤 Your Basic Information**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class='info-box'>
                        <strong>📛 Name:</strong> {resume_data['name']}<br>
                        <strong>📧 Email:</strong> {resume_data['email'] if resume_data['email'] else 'Not found'}<br>
                        <strong>📱 Contact:</strong> {resume_data['mobile_number'] if resume_data['mobile_number'] else 'Not found'}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='info-box'>
                        <strong>🎓 Degree:</strong> {resume_data['degree'] if resume_data['degree'] else 'Not found'}<br>
                        <strong>📄 Resume Pages:</strong> {resume_data['no_of_pages']}<br>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show extracted organizations and locations
                if resume_data['organizations'] or resume_data['locations']:
                    st.subheader("**🏢 ML-Extracted Entities**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if resume_data['organizations']:
                            st.info(f"**Organizations:** {', '.join(resume_data['organizations'][:3])}")
                    
                    with col2:
                        if resume_data['locations']:
                            st.info(f"**Locations:** {', '.join(resume_data['locations'][:3])}")
                
                # Experience Level Prediction
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("**💼 Experience Level**")
                
                cand_level = ''
                if resume_data['no_of_pages'] < 1:                
                    cand_level = "Fresher"
                    st.markdown("<div class='level-badge level-fresher'>🌱 Fresher Level</div>", unsafe_allow_html=True)
                elif 'INTERNSHIP' in resume_text or 'Internship' in resume_text or 'INTERNSHIPS' in resume_text or 'Internships' in resume_text:
                    cand_level = "Intermediate"
                    st.markdown("<div class='level-badge level-intermediate'>📈 Intermediate Level</div>", unsafe_allow_html=True)
                elif 'EXPERIENCE' in resume_text or 'Experience' in resume_text or 'WORK EXPERIENCE' in resume_text or 'Work Experience' in resume_text:
                    cand_level = "Experienced"
                    st.markdown("<div class='level-badge level-experienced'>⭐ Experienced Level</div>", unsafe_allow_html=True)
                else:
                    cand_level = "Fresher"
                    st.markdown("<div class='level-badge level-fresher'>🌱 Fresher Level</div>", unsafe_allow_html=True)

                # ML-Based Career Field Prediction
                st.markdown("---")
                st.header("**🤖 AI Career Field Prediction**")
                
                predicted_field, confidence = predict_career_field_ml(resume_text, resume_data['skills'])
                
                st.success(f"**🎯 Predicted Career Field:** {predicted_field}")
                st.info(f"**🎲 Confidence Score:** {confidence:.2%}")
                
                # AI Detection Analysis
                st.markdown("---")
                st.header("**🔍 AI Content Detection Analysis**")
                
                ai_probability, ai_indicators = detect_ai_generated_content(resume_text)
                
                # Display AI Detection Results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### 🤖 AI Generation Probability")
                    
                    # Color code based on probability
                    if ai_probability < 0.3:
                        color = "#28a745"
                        emoji = "✅"
                        verdict = "Likely Human-Written"
                        message = "Your resume shows natural writing patterns with good authenticity."
                    elif ai_probability < 0.6:
                        color = "#ffc107"
                        emoji = "⚠️"
                        verdict = "Possibly AI-Assisted"
                        message = "Some sections may have been enhanced with AI tools. Consider adding more personal touches."
                    else:
                        color = "#dc3545"
                        emoji = "🚨"
                        verdict = "Likely AI-Generated"
                        message = "Strong indicators of AI-generated content detected. Recruiters prefer authentic, personalized resumes."
                    
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                                color: white; padding: 1.5rem; border-radius: 15px; text-align: center; 
                                margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                        <h1 style='margin: 0; font-size: 2.5rem;'>{emoji} {ai_probability*100:.1f}%</h1>
                        <h3 style='margin: 0.5rem 0 0 0;'>{verdict}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info(f"💡 **Analysis:** {message}")
                
                with col2:
                    st.markdown("### 📊 Detection Metrics")
                    st.metric("Generic Phrases", f"{ai_indicators['generic_phrases']*100:.0f}%")
                    st.metric("Sentence Uniformity", f"{ai_indicators['sentence_uniformity']*100:.0f}%")
                    st.metric("Repetitive Patterns", f"{ai_indicators['repetitive_patterns']*100:.0f}%")
                    st.metric("Buzzword Density", f"{ai_indicators['buzzword_density']*100:.0f}%")
                
                # Detailed breakdown
                with st.expander("📋 View Detailed AI Detection Report"):
                    st.markdown("#### What We Analyzed:")
                    st.markdown(f"""
                    - **Generic Phrases Score:** {ai_indicators['generic_phrases']*100:.1f}% - Checks for overused AI-generated phrases like "results-oriented professional", "proven track record"
                    - **Sentence Uniformity:** {ai_indicators['sentence_uniformity']*100:.1f}% - AI tends to generate sentences of similar length
                    - **Repetitive Patterns:** {ai_indicators['repetitive_patterns']*100:.1f}% - Checks if sentences start similarly (AI characteristic)
                    - **Buzzword Density:** {ai_indicators['buzzword_density']*100:.1f}% - High concentration of business buzzwords
                    - **Grammar Perfection:** {ai_indicators['perfect_grammar']*100:.1f}% - Unnaturally perfect grammar can indicate AI
                    """)
                    
                    st.markdown("#### 💡 Tips for More Authentic Resumes:")
                    st.markdown("""
                    - Use specific examples and numbers from your real experience
                    - Vary your sentence structure and length naturally
                    - Replace generic phrases with concrete achievements
                    - Include personal projects and genuine interests
                    - Write in your own voice, not corporate buzzwords
                    - Add unique details that only you can provide
                    """)
                
                # Skills Analysis
                st.markdown("---")
                st.header("**💡 ML-Extracted Skills Analysis**")
                
                if resume_data['skills']:
                    st.markdown("### 🎯 Your Current Skills (ML Detected)")
                    st.info(f"Found **{len(resume_data['skills'])}** technical skills using AI")
                    keywords = st_tags(
                        label='',
                        text='Skills extracted using spaCy NLP',
                        value=resume_data['skills'],
                        key='1'
                    )
                else:
                    st.warning("⚠️ No skills detected. Please make sure your resume has a skills section.")
                    keywords = []

                # Skill Recommendations based on predicted field
                st.markdown("### ⭐ Recommended Skills for You")
                
                field_recommendations = {
                    'Data Science': {
                        'skills': ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling', 'Data Mining', 
                                   'Clustering & Classification', 'Data Analytics', 'Quantitative Analysis', 'Web Scraping', 
                                   'ML Algorithms', 'Keras', 'Pytorch', 'Probability', 'Scikit-learn', 'Tensorflow', "Flask", 'Streamlit'],
                        'courses': ds_course
                    },
                    'Web Development': {
                        'skills': ['React', 'Django', 'Node JS', 'React JS', 'php', 'laravel', 'Magento', 'wordpress', 
                                   'Javascript', 'Angular JS', 'c#', 'Flask', 'SDK', 'TypeScript', 'Next.js'],
                        'courses': web_course
                    },
                    'Mobile Development': {
                        'skills': ['Android', 'Android development', 'Flutter', 'Kotlin', 'XML', 'Java', 'Kivy', 'GIT', 
                                   'SDK', 'SQLite', 'Swift', 'iOS', 'React Native'],
                        'courses': android_course
                    },
                    'UI/UX Design': {
                        'skills': ['UI', 'User Experience', 'Adobe XD', 'Figma', 'Zeplin', 'Balsamiq', 'Prototyping', 
                                   'Wireframes', 'Storyframes', 'Adobe Photoshop', 'Editing', 'Illustrator', 'After Effects', 
                                   'Premier Pro', 'Indesign', 'Wireframe', 'Solid', 'Grasp', 'User Research'],
                        'courses': uiux_course
                    },
                    'DevOps': {
                        'skills': ['Docker', 'Kubernetes', 'Jenkins', 'CI/CD', 'AWS', 'Azure', 'GCP', 'Terraform', 
                                   'Ansible', 'GitLab', 'GitHub Actions', 'Monitoring'],
                        'courses': ds_course  # Using ds_course as placeholder
                    },
                    'Data Engineering': {
                        'skills': ['ETL', 'Apache Airflow', 'Kafka', 'Spark', 'Hadoop', 'Data Pipeline', 'SQL', 
                                   'NoSQL', 'Data Warehouse', 'Big Data'],
                        'courses': ds_course  # Using ds_course as placeholder
                    }
                }
                
                if predicted_field in field_recommendations:
                    recommended_skills = field_recommendations[predicted_field]['skills']
                    recommended_keywords = st_tags(
                        label='',
                        text=f'AI-recommended skills for {predicted_field}',
                        value=recommended_skills,
                        key='2'
                    )
                    st.info("💡 **Tip:** Adding these skills to your resume will boost your chances of getting a Job!")
                    
                    # Course Recommendations
                    rec_course = course_recommender(field_recommendations[predicted_field]['courses'])
                else:
                    st.info("**ℹ️ General recommendations - AI couldn't confidently predict a specific field.**")

                # Resume Scoring
                st.markdown("---")
                st.header("**✨ Resume Content Analysis**")
                resume_score = 0
                
                # Check for key sections
                if 'Objective' in resume_text or 'Summary' in resume_text or 'OBJECTIVE' in resume_text or 'SUMMARY' in resume_text:
                    resume_score += 6
                    st.markdown("<div class='success-item'>✅ <strong>Awesome!</strong> You have added Objective/Summary</div>", unsafe_allow_html=True)                
                else:
                    st.markdown("<div class='warning-item'>⚠️ <strong>Suggestion:</strong> Add your career objective - it will give your career intention to the Recruiters</div>", unsafe_allow_html=True)

                if 'Education' in resume_text or 'EDUCATION' in resume_text or 'School' in resume_text or 'College' in resume_text or 'University' in resume_text:
                    resume_score += 12
                    st.markdown("<div class='success-item'>✅ <strong>Awesome!</strong> You have added Education Details</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='warning-item'>⚠️ <strong>Suggestion:</strong> Add Education - it will show your qualification level to the recruiter</div>", unsafe_allow_html=True)

                if 'EXPERIENCE' in resume_text or 'Experience' in resume_text:
                    resume_score += 16
                    st.markdown("<div class='success-item'>✅ <strong>Awesome!</strong> You have added Experience</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='warning-item'>⚠️ <strong>Suggestion:</strong> Add Experience - it will help you stand out from the crowd</div>", unsafe_allow_html=True)

                if 'INTERNSHIP' in resume_text or 'Internship' in resume_text or 'INTERNSHIPS' in resume_text or 'Internships' in resume_text:
                    resume_score += 6
                    st.markdown("<div class='success-item'>✅ <strong>Awesome!</strong> You have added Internships</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='warning-item'>⚠️ <strong>Suggestion:</strong> Add Internships - it will help you stand out from the crowd</div>", unsafe_allow_html=True)

                if 'SKILLS' in resume_text or 'SKILL' in resume_text or 'Skills' in resume_text or 'Skill' in resume_text:
                    resume_score += 7
                    st.markdown("<div class='success-item'>✅ <strong>Awesome!</strong> You have added Skills</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='warning-item'>⚠️ <strong>Suggestion:</strong> Add Skills section - it will help you a lot</div>", unsafe_allow_html=True)

                if 'HOBBIES' in resume_text or 'Hobbies' in resume_text:
                    resume_score += 4
                    st.markdown("<div class='success-item'>✅ <strong>Awesome!</strong> You have added your Hobbies</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='warning-item'>⚠️ <strong>Suggestion:</strong> Add Hobbies - it will show your personality to the Recruiters</div>", unsafe_allow_html=True)

                if 'INTERESTS' in resume_text or 'Interests' in resume_text:
                    resume_score += 5
                    st.markdown("<div class='success-item'>✅ <strong>Awesome!</strong> You have added your Interests</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='warning-item'>⚠️ <strong>Suggestion:</strong> Add Interests - it will show your interests other than job</div>", unsafe_allow_html=True)

                if 'ACHIEVEMENTS' in resume_text or 'Achievements' in resume_text:
                    resume_score += 13
                    st.markdown("<div class='success-item'>✅ <strong>Awesome!</strong> You have added your Achievements</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='warning-item'>⚠️ <strong>Suggestion:</strong> Add Achievements - it shows you are capable for the position</div>", unsafe_allow_html=True)

                if 'CERTIFICATIONS' in resume_text or 'Certifications' in resume_text or 'Certification' in resume_text:
                    resume_score += 12
                    st.markdown("<div class='success-item'>✅ <strong>Awesome!</strong> You have added your Certifications</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='warning-item'>⚠️ <strong>Suggestion:</strong> Add Certifications - it shows you have specialization</div>", unsafe_allow_html=True)

                if 'PROJECTS' in resume_text or 'PROJECT' in resume_text or 'Projects' in resume_text or 'Project' in resume_text:
                    resume_score += 19
                    st.markdown("<div class='success-item'>✅ <strong>Awesome!</strong> You have added your Projects</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='warning-item'>⚠️ <strong>Suggestion:</strong> Add Projects - it shows relevant work experience</div>", unsafe_allow_html=True)

                # Display Resume Score
                st.markdown("---")
                st.header("**📊 Your Resume Score**")
                
                # Score Bar
                my_bar = st.progress(0)
                for percent_complete in range(resume_score):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1)
                
                # Display Score with color coding
                if resume_score >= 80:
                    score_color = "#28a745"
                    score_emoji = "🌟"
                    score_text = "Excellent!"
                elif resume_score >= 60:
                    score_color = "#ffc107"
                    score_emoji = "👍"
                    score_text = "Good!"
                else:
                    score_color = "#dc3545"
                    score_emoji = "📈"
                    score_text = "Needs Improvement"
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {score_color} 0%, {score_color}dd 100%); 
                            color: white; padding: 2rem; border-radius: 15px; text-align: center; 
                            margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h1 style='margin: 0; font-size: 3rem;'>{score_emoji} {resume_score}/100</h1>
                    <h3 style='margin: 0.5rem 0 0 0;'>{score_text}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.info("💡 **Note:** This score is calculated based on the content in your Resume. ML models analyzed structure and completeness.")

                # Recommending Resume Writing Video
                st.markdown("---")
                st.header("**🎥 Bonus: Resume Writing Tips**")
                resume_vid = random.choice(resume_videos)
                st.video(resume_vid)

                # Recommending Interview Preparation Video
                st.header("**🎥 Bonus: Interview Preparation Tips**")
                interview_vid = random.choice(interview_videos)
                st.video(interview_vid)

                # Success message
                st.success("✅ Analysis Complete!")

            else:
                st.error('❌ Could not extract text from the PDF. Please make sure you uploaded a valid PDF resume.')
                
        except Exception as e:
            st.error(f'❌ Error processing resume: {str(e)}')
            st.info('Please make sure you uploaded a valid PDF file with text content.')
            st.info('If you see spaCy errors, run: `python -m spacy download en_core_web_sm`')


# Run the application
if __name__ == '__main__':
    run()