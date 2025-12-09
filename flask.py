from flask import Flask, render_template, request, jsonify, session
import os
import json
import random
import numpy as np
import pandas as pd
import chromadb
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import shutil
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import TreebankWordTokenizer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

# ==================== FLASK APP ====================
app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'  # Change this!
app.config['JSON_SORT_KEYS'] = False

# ==================== CONFIGURATION ====================
ORIGINAL_CSV = "science_qa.csv"
PREPROCESSED_CSV = "preprocessed_science_qa.csv"
VECTOR_DB_DIR = "vector_db"
MODEL_NAME = "allenai/scibert_scivocab_uncased"
PROGRESS_DIR = "student_progress"

# Quiz parameters
MASTERY_THRESHOLD = 0.80
TOPIC_SIMILARITY_THRESHOLD = 0.70
MIN_TOPIC_MASTERY = 3
TOPIC_MASTERY_RATIO = 0.75

os.makedirs(PROGRESS_DIR, exist_ok=True)

# ==================== DOWNLOAD NLTK RESOURCES ====================
print("Checking NLTK resources...")
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("✓ NLTK resources ready")
except Exception as e:
    print(f"⚠ NLTK download warning: {e}")

# ==================== TEXT PREPROCESSOR ====================
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = TreebankWordTokenizer()
    
    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    def preprocess(self, text):
        try:
            if pd.isna(text) or not text:
                return ""
            
            text = text.lower()
            text = re.sub(r'[^a-z\s]', '', text)
            tokens = self.tokenizer.tokenize(text)
            tokens = [word for word in tokens if word not in self.stop_words]
            pos_tags = pos_tag(tokens)
            lemmatized = [
                self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos)) 
                for word, pos in pos_tags
            ]
            
            return ' '.join(lemmatized)
        except Exception as e:
            print(f"⚠ Preprocessing error: {e}")
            return text

# ==================== LOAD MODELS AND DATA ====================
print("Loading datasets and models...")

# Load CSVs
df_original = pd.read_csv(ORIGINAL_CSV)
df_preprocessed = pd.read_csv(PREPROCESSED_CSV)

# Find columns
def find_column(df, name):
    for col in df.columns:
        if col.lower() == name.lower():
            return col
    return None

question_col = find_column(df_original, 'question')
answer_col = find_column(df_original, 'answer')
question_col_prep = find_column(df_preprocessed, 'question')
answer_col_prep = find_column(df_preprocessed, 'answer')

print(f"✓ Loaded {len(df_original)} questions")

# Initialize preprocessor
preprocessor = TextPreprocessor()

# Load SciBERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()
print("✓ SciBERT loaded")

def get_embedding(text):
    if not text or text.strip() == "":
        return np.zeros(768)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    embedding = sum_embeddings / sum_mask
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    
    return embedding.cpu().numpy()[0]

# Connect to ChromaDB
client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
answers_collection = client.get_collection("science_qa_answers")
questions_collection = client.get_collection("science_qa_questions")

print(f"✓ ChromaDB connected ({questions_collection.count()} questions)")

# ==================== QUIZ SYSTEM ====================
class AdaptiveQuizSystem:
    def __init__(self):
        self.all_questions = self._build_question_index()
        print(f"✓ Quiz system initialized with {len(self.all_questions)} questions")
    
    def _build_question_index(self):
        all_questions = []
        
        for row_idx in range(len(df_original)):
            question_text = str(df_original[question_col].iloc[row_idx]).strip()
            answer_text = str(df_original[answer_col].iloc[row_idx]).strip()
            question_prep = str(df_preprocessed[question_col_prep].iloc[row_idx]).strip()
            answer_prep = str(df_preprocessed[answer_col_prep].iloc[row_idx]).strip()
            
            if not question_text or question_text.lower() == 'nan':
                continue
            if not answer_text or answer_text.lower() == 'nan':
                continue
            if not question_prep or question_prep.lower() == 'nan':
                continue
            if not answer_prep or answer_prep.lower() == 'nan':
                continue
            
            all_questions.append({
                'row_idx': row_idx,
                'question': question_text,
                'ideal_answer': answer_text,
                'question_preprocessed': question_prep,
                'answer_preprocessed': answer_prep
            })
        
        return all_questions
    
    def get_random_question(self, progress):
        available = [
            q for q in self.all_questions
            if q['row_idx'] not in progress.get('asked_questions', [])
        ]
        
        if not available:
            return None
        
        return random.choice(available)
    
    def evaluate_answer(self, question_data, student_answer):
        # Preprocess student answer
        preprocessed_student_answer = preprocessor.preprocess(student_answer)
        preprocessed_ideal_answer = question_data['answer_preprocessed']
        
        # Generate fresh embeddings
        ideal_embedding = get_embedding(preprocessed_ideal_answer)
        student_embedding = get_embedding(preprocessed_student_answer)
        
        # Compare
        similarity = cosine_similarity(
            student_embedding.reshape(1, -1),
            ideal_embedding.reshape(1, -1)
        )[0][0]
        
        grade = self._get_grade(similarity)
        is_mastered = similarity >= MASTERY_THRESHOLD
        
        return {
            'question': question_data['question'],
            'ideal_answer': question_data['ideal_answer'],
            'student_answer': student_answer,
            'preprocessed_student': preprocessed_student_answer,
            'preprocessed_ideal': preprocessed_ideal_answer,
            'similarity': float(similarity),
            'grade': grade,
            'mastered': is_mastered,
            'row_idx': question_data['row_idx']
        }
    
    def _get_grade(self, similarity):
        if similarity >= 0.95:
            return 'A+'
        elif similarity >= 0.90:
            return 'A'
        elif similarity >= 0.85:
            return 'A-'
        elif similarity >= MASTERY_THRESHOLD:
            return 'B+'
        elif similarity >= 0.75:
            return 'B'
        elif similarity >= 0.70:
            return 'C+'
        elif similarity >= 0.60:
            return 'C'
        elif similarity >= 0.50:
            return 'D'
        else:
            return 'F'
    
    def get_next_similar_question(self, current_question_data, progress):
        preprocessed_question = current_question_data['question_preprocessed']
        current_embedding = get_embedding(preprocessed_question)
        
        results = questions_collection.query(
            query_embeddings=[current_embedding.tolist()],
            n_results=20,
            include=['distances', 'metadatas']
        )
        
        if not results['ids'][0]:
            return None
        
        for distance, metadata in zip(results['distances'][0], results['metadatas'][0]):
            similarity = 1 - (distance ** 2) / 2
            
            if similarity < TOPIC_SIMILARITY_THRESHOLD:
                continue
            
            if 'row_index' not in metadata:
                continue
            
            row_idx = metadata['row_index']
            
            if row_idx in progress.get('asked_questions', []):
                continue
            if row_idx in progress.get('mastered_questions', []):
                continue
            if row_idx == current_question_data['row_idx']:
                continue
            
            question_text = str(df_original[question_col].iloc[row_idx]).strip()
            answer_text = str(df_original[answer_col].iloc[row_idx]).strip()
            question_prep = str(df_preprocessed[question_col_prep].iloc[row_idx]).strip()
            answer_prep = str(df_preprocessed[answer_col_prep].iloc[row_idx]).strip()
            
            if not question_text or question_text.lower() == 'nan':
                continue
            
            return {
                'row_idx': row_idx,
                'question': question_text,
                'ideal_answer': answer_text,
                'question_preprocessed': question_prep,
                'answer_preprocessed': answer_prep,
                'similarity': float(similarity)
            }
        
        return None
    
    def check_topic_mastery(self, current_question_data, progress):
        preprocessed_question = current_question_data['question_preprocessed']
        current_embedding = get_embedding(preprocessed_question)
        
        results = questions_collection.query(
            query_embeddings=[current_embedding.tolist()],
            n_results=50,
            include=['distances', 'metadatas']
        )
        
        similar_questions = []
        for distance, metadata in zip(results['distances'][0], results['metadatas'][0]):
            similarity = 1 - (distance ** 2) / 2
            
            if similarity >= TOPIC_SIMILARITY_THRESHOLD:
                if 'row_index' in metadata:
                    similar_questions.append(metadata['row_index'])
        
        similar_questions.append(current_question_data['row_idx'])
        
        mastered_count = sum(
            1 for row_idx in similar_questions
            if row_idx in progress.get('mastered_questions', [])
        )
        
        cluster_size = len(similar_questions)
        mastery_ratio = mastered_count / cluster_size if cluster_size > 0 else 0
        
        topic_mastered = (
            mastered_count >= MIN_TOPIC_MASTERY and 
            mastery_ratio >= TOPIC_MASTERY_RATIO
        )
        
        return {
            'topic_mastered': topic_mastered,
            'cluster_size': cluster_size,
            'mastered_count': mastered_count,
            'mastery_ratio': mastery_ratio
        }

# Initialize quiz system
quiz_system = AdaptiveQuizSystem()

# ==================== PROGRESS MANAGEMENT ====================
def get_progress_file(student_id):
    return os.path.join(PROGRESS_DIR, f"{student_id}_progress.json")

def load_progress(student_id):
    progress_file = get_progress_file(student_id)
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return {
        'asked_questions': [],
        'mastered_questions': [],
        'question_scores': {},
        'session_history': []
    }

def save_progress(student_id, progress):
    progress_file = get_progress_file(student_id)
    temp_file = progress_file + '.tmp'
    
    try:
        with open(temp_file, 'w') as f:
            json.dump(progress, f, indent=2)
        shutil.move(temp_file, progress_file)
    except Exception as e:
        print(f"Error saving progress: {e}")

# ==================== ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_quiz():
    """Start a new quiz session."""
    data = request.json
    student_id = data.get('student_id', 'default_student')
    
    session['student_id'] = student_id
    progress = load_progress(student_id)
    
    # Get first question
    question_data = quiz_system.get_random_question(progress)
    
    if not question_data:
        return jsonify({'error': 'No questions available'}), 400
    
    session['current_question'] = question_data
    
    return jsonify({
        'question': question_data['question'],
        'row_idx': question_data['row_idx'],
        'total_available': len(quiz_system.all_questions),
        'total_asked': len(progress.get('asked_questions', [])),
        'total_mastered': len(progress.get('mastered_questions', []))
    })

@app.route('/api/submit', methods=['POST'])
def submit_answer():
    """Submit and evaluate an answer."""
    data = request.json
    student_answer = data.get('answer', '')
    student_id = session.get('student_id', 'default_student')
    current_question = session.get('current_question')
    
    if not current_question:
        return jsonify({'error': 'No active question'}), 400
    
    if not student_answer.strip():
        return jsonify({'error': 'Answer cannot be empty'}), 400
    
    # Evaluate answer
    evaluation = quiz_system.evaluate_answer(current_question, student_answer)
    
    # Update progress
    progress = load_progress(student_id)
    row_idx = evaluation['row_idx']
    key_str = f"row{row_idx}"
    
    if key_str not in progress['question_scores']:
        progress['question_scores'][key_str] = []
    
    progress['question_scores'][key_str].append(evaluation['similarity'])
    
    if evaluation['mastered'] and row_idx not in progress['mastered_questions']:
        progress['mastered_questions'].append(row_idx)
    
    if row_idx not in progress['asked_questions']:
        progress['asked_questions'].append(row_idx)
    
    progress['session_history'].append({
        'row_idx': row_idx,
        'question': current_question['question'],
        'similarity': evaluation['similarity'],
        'mastered': evaluation['mastered'],
        'timestamp': datetime.now().isoformat()
    })
    
    save_progress(student_id, progress)
    
    # Calculate progress stats
    total_asked = len(set(progress['asked_questions']))
    total_mastered = len(set(progress['mastered_questions']))
    all_scores = [s for scores in progress['question_scores'].values() for s in scores]
    avg_score = np.mean(all_scores) if all_scores else 0
    
    evaluation['progress'] = {
        'total_asked': total_asked,
        'total_mastered': total_mastered,
        'mastery_rate': total_mastered / total_asked if total_asked > 0 else 0,
        'average_score': float(avg_score),
        'attempts': len(progress['question_scores'][key_str])
    }
    
    return jsonify(evaluation)

@app.route('/api/next', methods=['POST'])
def next_question():
    """Get the next question based on adaptive logic."""
    student_id = session.get('student_id', 'default_student')
    current_question = session.get('current_question')
    
    progress = load_progress(student_id)
    
    if not current_question:
        # First question
        question_data = quiz_system.get_random_question(progress)
    else:
        # Check topic mastery
        topic_status = quiz_system.check_topic_mastery(current_question, progress)
        
        if topic_status['topic_mastered']:
            # Move to new topic
            question_data = quiz_system.get_random_question(progress)
            strategy = 'new_topic'
        else:
            # Continue current topic
            question_data = quiz_system.get_next_similar_question(current_question, progress)
            strategy = 'continue_topic'
            
            if not question_data:
                question_data = quiz_system.get_random_question(progress)
                strategy = 'random'
    
    if not question_data:
        return jsonify({'completed': True, 'message': 'All questions completed!'})
    
    session['current_question'] = question_data
    
    response = {
        'question': question_data['question'],
        'row_idx': question_data['row_idx']
    }
    
    if 'similarity' in question_data:
        response['similarity_to_previous'] = question_data['similarity']
    
    return jsonify(response)

@app.route('/api/progress', methods=['GET'])
def get_progress():
    """Get current progress summary."""
    student_id = session.get('student_id', 'default_student')
    progress = load_progress(student_id)
    
    total_asked = len(set(progress.get('asked_questions', [])))
    total_mastered = len(set(progress.get('mastered_questions', [])))
    all_scores = [s for scores in progress.get('question_scores', {}).values() for s in scores]
    avg_score = np.mean(all_scores) if all_scores else 0
    
    return jsonify({
        'total_available': len(quiz_system.all_questions),
        'total_asked': total_asked,
        'total_mastered': total_mastered,
        'mastery_rate': total_mastered / total_asked if total_asked > 0 else 0,
        'average_score': float(avg_score),
        'total_attempts': len(progress.get('session_history', []))
    })

@app.route('/api/reset', methods=['POST'])
def reset_progress():
    """Reset student progress."""
    student_id = session.get('student_id', 'default_student')
    progress_file = get_progress_file(student_id)
    
    if os.path.exists(progress_file):
        os.remove(progress_file)
    
    session.pop('current_question', None)
    
    return jsonify({'message': 'Progress reset successfully'})

# ==================== HTML TEMPLATE ====================
@app.route('/template')
def get_template():
    """Return the HTML template as string for reference."""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Adaptive Science Quiz</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        h1 {
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        
        .question {
            font-size: 1.3em;
            color: #333;
            margin-bottom: 20px;
            line-height: 1.6;
        }
        
        textarea {
            width: 100%;
            min-height: 120px;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1em;
            font-family: inherit;
            resize: vertical;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .button {
            background: #667eea;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 1.1em;
            cursor: pointer;
            margin: 10px 5px;
            transition: all 0.3s;
        }
        
        .button:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .button-secondary {
            background: #6c757d;
        }
        
        .button-secondary:hover {
            background: #5a6268;
        }
        
        .evaluation {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        
        .score {
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }
        
        .score.high { color: #28a745; }
        .score.medium { color: #ffc107; }
        .score.low { color: #dc3545; }
        
        .grade {
            text-align: center;
            font-size: 1.5em;
            color: #666;
            margin-bottom: 20px;
        }
        
        .status {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 5px;
        }
        
        .status.mastered {
            background: #d4edda;
            color: #155724;
        }
        
        .status.practice {
            background: #fff3cd;
            color: #856404;
        }
        
        .progress-bar {
            background: #e0e0e0;
            border-radius: 10px;
            height: 30px;
            overflow: hidden;
            margin: 20px 0;
        }
        
        .progress-fill {
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            transition: width 0.5s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 2px solid #e0e0e0;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        
        .hidden {
            display: none;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #667eea;
        }
        
        .answer-section {
            margin-top: 15px;
        }
        
        .answer-label {
            font-weight: bold;
            color: #555;
            margin-top: 15px;
            margin-bottom: 5px;
        }
        
        .answer-text {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin-bottom: 10px;
        }
        
        .buttons-container {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Welcome Screen -->
        <div id="welcomeScreen" class="card">
            <h1>🎓 Adaptive Science Quiz</h1>
            <p class="subtitle">AI-powered learning that adapts to your knowledge</p>
            
            <div style="margin: 30px 0;">
                <label for="studentId" style="display: block; margin-bottom: 10px; font-weight: bold;">Student ID:</label>
                <input type="text" id="studentId" placeholder="Enter your student ID" 
                       style="width: 100%; padding: 12px; border: 2px solid #e0e0e0; border-radius: 8px; font-size: 1em;">
            </div>
            
            <div class="buttons-container">
                <button class="button" onclick="startQuiz()">Start Quiz</button>
            </div>
        </div>
        
        <!-- Quiz Screen -->
        <div id="quizScreen" class="hidden">
            <div class="card">
                <h1>❓ Question</h1>
                
                <div id="questionText" class="question"></div>
                
                <textarea id="answerInput" placeholder="Type your answer here..."></textarea>
                
                <div class="buttons-container">
                    <button class="button" onclick="submitAnswer()">Submit Answer</button>
                    <button class="button button-secondary" onclick="skipQuestion()">Skip</button>
                    <button class="button button-secondary" onclick="showProgress()">View Progress</button>
                </div>
            </div>
            
            <!-- Evaluation Results -->
            <div id="evaluationCard" class="card hidden">
                <h2 style="text-align: center; color: #667eea; margin-bottom: 20px;">📊 Evaluation Results</h2>
                
                <div id="scoreDisplay" class="score"></div>
                <div id="gradeDisplay" class="grade"></div>
                
                <div class="evaluation">
                    <div class="answer-section">
                        <div class="answer-label">✍️ Your Answer:</div>
                        <div id="studentAnswerText" class="answer-text"></div>
                        
                        <div class="answer-label">📚 Ideal Answer:</div>
                        <div id="idealAnswerText" class="answer-text"></div>
                        
                        <div id="statusDisplay" style="text-align: center; margin-top: 20px;"></div>
                    </div>
                </div>
                
                <div class="buttons-container">
                    <button class="button" onclick="nextQuestion()">Next Question</button>
                </div>
            </div>
        </div>
        
        <!-- Progress Screen -->
        <div id="progressScreen" class="hidden">
            <div class="card">
                <h1>📈 Your Progress</h1>
                
                <div class="progress-bar">
                    <div id="progressFill" class="progress-fill">0%</div>
                </div>
                
                <div class="stats">
                    <div class="stat-card">
                        <div id="totalAsked" class="stat-value">0</div>
                        <div class="stat-label">Questions Asked</div>
                    </div>
                    <div class="stat-card">
                        <div id="totalMastered" class="stat-value">0</div>
                        <div class="stat-label">Questions Mastered</div>
                    </div>
                    <div class="stat-card">
                        <div id="avgScore" class="stat-value">0%</div>
                        <div class="stat-label">Average Score</div>
                    </div>
                    <div class="stat-card">
                        <div id="totalAttempts" class="stat-value">0</div>
                        <div class="stat-label">Total Attempts</div>
                    </div>
                </div>
                
                <div class="buttons-container">
                    <button class="button" onclick="backToQuiz()">Back to Quiz</button>
                    <button class="button button-secondary" onclick="resetProgress()">Reset Progress</button>
                </div>
            </div>
        </div>
        
        <!-- Loading Indicator -->
        <div id="loadingIndicator" class="hidden">
            <div class="card loading">
                <h2>⏳ Processing...</h2>
            </div>
        </div>
    </div>
    
    <script>
        let currentProgress = null;
        
        // Start Quiz
        async function startQuiz() {
            const studentId = document.getElementById('studentId').value.trim();
            
            if (!studentId) {
                alert('Please enter a student ID');
                return;
            }
            
            showLoading();
            
            try {
                const response = await fetch('/api/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ student_id: studentId })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert(data.error);
                    return;
                }
                
                document.getElementById('questionText').textContent = data.question;
                document.getElementById('answerInput').value = '';
                
                hideLoading();
                showScreen('quizScreen');
                document.getElementById('evaluationCard').classList.add('hidden');
                
            } catch (error) {
                console.error('Error:', error);
                alert('Failed to start quiz');
                hideLoading();
            }
        }
        
        // Submit Answer
        async function submitAnswer() {
            const answer = document.getElementById('answerInput').value.trim();
            
            if (!answer) {
                alert('Please enter an answer');
                return;
            }
            
            showLoading();
            
            try {
                const response = await fetch('/api/submit', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ answer: answer })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert(data.error);
                    hideLoading();
                    return;
                }
                
                // Display evaluation
                displayEvaluation(data);
                hideLoading();
                
            } catch (error) {
                console.error('Error:', error);
                alert('Failed to submit answer');
                hideLoading();
            }
        }
        
        // Display Evaluation Results
        function displayEvaluation(data) {
            const similarity = data.similarity * 100;
            const scoreDisplay = document.getElementById('scoreDisplay');
            
            scoreDisplay.textContent = similarity.toFixed(1) + '%';
            
            // Color based on score
            scoreDisplay.className = 'score';
            if (similarity >= 80) {
                scoreDisplay.classList.add('high');
            } else if (similarity >= 60) {
                scoreDisplay.classList.add('medium');
            } else {
                scoreDisplay.classList.add('low');
            }
            
            document.getElementById('gradeDisplay').textContent = 'Grade: ' + data.grade;
            document.getElementById('studentAnswerText').textContent = data.student_answer;
            document.getElementById('idealAnswerText').textContent = data.ideal_answer;
            
            const statusDisplay = document.getElementById('statusDisplay');
            if (data.mastered) {
                statusDisplay.innerHTML = '<span class="status mastered">✅ MASTERED!</span>';
            } else {
                statusDisplay.innerHTML = '<span class="status practice">⏳ Keep Practicing</span>';
            }
            
            if (data.progress) {
                statusDisplay.innerHTML += `<div style="margin-top: 15px; color: #666;">
                    Attempts: ${data.progress.attempts} | 
                    Mastered: ${data.progress.total_mastered}/${data.progress.total_asked}
                </div>`;
            }
            
            document.getElementById('evaluationCard').classList.remove('hidden');
            document.getElementById('answerInput').disabled = true;
        }
        
        // Next Question
        async function nextQuestion() {
            showLoading();
            
            try {
                const response = await fetch('/api/next', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                
                const data = await response.json();
                
                if (data.completed) {
                    alert(data.message);
                    showProgress();
                    hideLoading();
                    return;
                }
                
                document.getElementById('questionText').textContent = data.question;
                document.getElementById('answerInput').value = '';
                document.getElementById('answerInput').disabled = false;
                document.getElementById('evaluationCard').classList.add('hidden');
                
                if (data.similarity_to_previous) {
                    const sim = (data.similarity_to_previous * 100).toFixed(1);
                    document.getElementById('questionText').textContent += 
                        ` (${sim}% similar to previous)`;
                }
                
                hideLoading();
                
            } catch (error) {
                console.error('Error:', error);
                alert('Failed to get next question');
                hideLoading();
            }
        }
        
        // Skip Question
        async function skipQuestion() {
            if (!confirm('Skip this question?')) {
                return;
            }
            
            await nextQuestion();
        }
        
        // Show Progress
        async function showProgress() {
            showLoading();
            
            try {
                const response = await fetch('/api/progress');
                const data = await response.json();
                
                currentProgress = data;
                
                const masteryPercent = (data.mastery_rate * 100).toFixed(1);
                const avgScorePercent = (data.average_score * 100).toFixed(1);
                
                document.getElementById('progressFill').style.width = masteryPercent + '%';
                document.getElementById('progressFill').textContent = masteryPercent + '% Mastered';
                
                document.getElementById('totalAsked').textContent = data.total_asked;
                document.getElementById('totalMastered').textContent = data.total_mastered;
                document.getElementById('avgScore').textContent = avgScorePercent + '%';
                document.getElementById('totalAttempts').textContent = data.total_attempts;
                
                hideLoading();
                showScreen('progressScreen');
                
            } catch (error) {
                console.error('Error:', error);
                alert('Failed to load progress');
                hideLoading();
            }
        }
        
        // Back to Quiz
        function backToQuiz() {
            showScreen('quizScreen');
        }
        
        // Reset Progress
        async function resetProgress() {
            if (!confirm('Are you sure you want to reset all progress? This cannot be undone.')) {
                return;
            }
            
            showLoading();
            
            try {
                const response = await fetch('/api/reset', {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                alert(data.message);
                hideLoading();
                showScreen('welcomeScreen');
                
            } catch (error) {
                console.error('Error:', error);
                alert('Failed to reset progress');
                hideLoading();
            }
        }
        
        // Screen Management
        function showScreen(screenId) {
            const screens = ['welcomeScreen', 'quizScreen', 'progressScreen'];
            screens.forEach(id => {
                document.getElementById(id).classList.add('hidden');
            });
            document.getElementById(screenId).classList.remove('hidden');
        }
        
        function showLoading() {
            document.getElementById('loadingIndicator').classList.remove('hidden');
        }
        
        function hideLoading() {
            document.getElementById('loadingIndicator').classList.add('hidden');
        }
        
        // Enter key to submit
        document.addEventListener('DOMContentLoaded', function() {
            const answerInput = document.getElementById('answerInput');
            const studentIdInput = document.getElementById('studentId');
            
            if (answerInput) {
                answerInput.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && e.ctrlKey) {
                        submitAnswer();
                    }
                });
            }
            
            if (studentIdInput) {
                studentIdInput.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        startQuiz();
                    }
                });
            }
        });
    </script>
</body>
</html>
    """
    return html

if __name__ == '__main__':
    print("\n" + "="*70)
    print("ADAPTIVE SCIENCE QUIZ - FLASK SERVER")
    print("="*70)
    print("\n🚀 Starting Flask server...")
    print("📍 Access the quiz at: http://localhost:5000")
    print("💾 Student progress saved in:", PROGRESS_DIR)
    print("\nPress Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)