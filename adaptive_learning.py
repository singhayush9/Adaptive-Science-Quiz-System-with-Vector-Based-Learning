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

# ==================== DOWNLOAD NLTK RESOURCES ====================
print("Checking NLTK resources...")
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("NLTK resources ready")
except Exception as e:
    print(f"NLTK download warning: {e}")

# ==================== CONFIGURATION ====================
original_csv = "science_qa.csv"  # Original clean questions
preprocessed_csv = "preprocessed_science_qa.csv"  # For embeddings reference
vector_db_dir = "vector_db"
model_name = "allenai/scibert_scivocab_uncased"
progress_file = "student_progress.json"

# Quiz parameters
MASTERY_THRESHOLD = 0.80  # 80% similarity = mastered
TOPIC_SIMILARITY_THRESHOLD = 0.70  # Questions above this are "similar topic"
MIN_TOPIC_MASTERY = 3  # Must master at least 3 questions in a topic
TOPIC_MASTERY_RATIO = 0.75  # 75% of similar questions must be mastered

# ==================== TEXT PREPROCESSING ====================
class TextPreprocessor:
    """Text preprocessing matching the embedding generation pipeline."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = TreebankWordTokenizer()
        print("Text preprocessor initialized")
    
    def get_wordnet_pos(self, tag):
        """Convert POS tags to WordNet format."""
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
        """
        Preprocess text using the same pipeline as embedding generation:
        1. Lowercase
        2. Remove non-alphabetic characters
        3. Tokenize
        4. Remove stopwords
        5. POS tagging
        6. Lemmatization
        """
        try:
            if pd.isna(text) or not text:
                return ""
            
            # 1. Lowercase
            text = text.lower()
            
            # 2. Remove non-alphabetic characters (keep spaces)
            text = re.sub(r'[^a-z\s]', '', text)
            
            # 3. Tokenize using TreebankWordTokenizer
            tokens = self.tokenizer.tokenize(text)
            
            # 4. Remove stopwords
            tokens = [word for word in tokens if word not in self.stop_words]
            
            # 5. POS tagging
            pos_tags = pos_tag(tokens)
            
            # 6. Lemmatize with POS
            lemmatized = [
                self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos)) 
                for word, pos in pos_tags
            ]
            
            # Return clean, lemmatized text string
            return ' '.join(lemmatized)
        
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return text  # Return original if preprocessing fails

# ==================== LOAD DATASETS ====================
print("="*70)
print("LOADING Q&A DATASETS")
print("="*70)

print(f"\nOriginal (for display): {original_csv}")
print(f"Preprocessed (for embeddings): {preprocessed_csv}")

df_original = pd.read_csv(original_csv)
print(f"Loaded original: {len(df_original)} rows")
print(f"  Columns: {df_original.columns.tolist()}")

df_preprocessed = pd.read_csv(preprocessed_csv)
print(f"Loaded preprocessed: {len(df_preprocessed)} rows")
print(f"  Columns: {df_preprocessed.columns.tolist()}")

# Verify both CSVs have same length
if len(df_original) != len(df_preprocessed):
    print(f"WARNING: CSV length mismatch!")
    print(f"   Original: {len(df_original)}, Preprocessed: {len(df_preprocessed)}")

# Detect column names (case-insensitive)
def find_column(df, name):
    for col in df.columns:
        if col.lower() == name.lower():
            return col
    return None

question_col = find_column(df_original, 'question')
answer_col = find_column(df_original, 'answer')

question_col_prep = find_column(df_preprocessed, 'question')
answer_col_prep = find_column(df_preprocessed, 'answer')

if not question_col or not answer_col:
    raise ValueError(f"Could not find 'question' and 'answer' columns in original CSV!")

if not question_col_prep or not answer_col_prep:
    raise ValueError(f"Could not find 'question' and 'answer' columns in preprocessed CSV!")

print(f"\nOriginal columns: '{question_col}' and '{answer_col}'")
print(f"Preprocessed columns: '{question_col_prep}' and '{answer_col_prep}'")

# ==================== INITIALIZE PREPROCESSOR ====================
print("\n" + "="*70)
print("INITIALIZING TEXT PREPROCESSOR")
print("="*70)

preprocessor = TextPreprocessor()

# ==================== INITIALIZE SCIBERT ====================
print("\n" + "="*70)
print("LOADING SCIBERT MODEL")
print("="*70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

print("Loading SciBERT...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.to(device)
model.eval()
print("Model loaded successfully")

def get_embedding(text):
    """
    Generate SciBERT embedding for text.
    Text should ALREADY be preprocessed.
    """
    if not text or text.strip() == "":
        print("Warning: Empty text for embedding")
        return np.zeros(768)  # Return zero vector
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    embedding = sum_embeddings / sum_mask
    
    # Normalize
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding.cpu().numpy()[0]

# ==================== INITIALIZE CHROMADB ====================
print("\n" + "="*70)
print("CONNECTING TO CHROMADB")
print("="*70)

client = chromadb.PersistentClient(path=vector_db_dir)
answers_collection = client.get_collection("science_qa_answers")
questions_collection = client.get_collection("science_qa_questions")

print(f"Answers collection: {answers_collection.count()} documents")
print(f"Questions collection: {questions_collection.count()} documents")

# ==================== ADAPTIVE QUIZ SYSTEM ====================
class AdaptiveQuizSystem:
    """
    Adaptive Quiz System for Science Q&A (FRESH EMBEDDINGS VERSION)
    
    KEY CHANGE: Generate BOTH embeddings fresh for fair comparison!
    
    WORKFLOW:
    1. Pick random question from original CSV (clean text for display)
    2. Get preprocessed question from preprocessed CSV (same row index)
    3. Embed preprocessed question
    4. Query answer DB with question embedding -> gets PAIRED answer TEXT
    5. Student provides answer
    6. Preprocess student answer at runtime
    7. Generate FRESH embeddings for BOTH ideal answer and student answer
    8. Compare fresh embeddings (apples-to-apples)
    9. Grade and provide feedback
    10. Find similar questions for topic clustering
    11. Repeat until topic mastered
    """
    
    def __init__(self, df_original, df_preprocessed, answers_collection, questions_collection, get_embedding_func):
        self.df_original = df_original
        self.df_preprocessed = df_preprocessed
        self.answers_collection = answers_collection
        self.questions_collection = questions_collection
        self.get_embedding = get_embedding_func
        self.progress = self._load_progress()
        
        # Build question index
        self._build_question_index()
        
        print(f"\nTotal valid questions: {len(self.all_questions)}")
    
    def _build_question_index(self):
        """Build index of all valid questions from BOTH CSVs."""
        self.all_questions = []
        
        for row_idx in range(len(self.df_original)):
            # Get from ORIGINAL (for display)
            question_text = str(self.df_original[question_col].iloc[row_idx]).strip()
            answer_text = str(self.df_original[answer_col].iloc[row_idx]).strip()
            
            # Get from PREPROCESSED (for embedding)
            question_prep = str(self.df_preprocessed[question_col_prep].iloc[row_idx]).strip()
            answer_prep = str(self.df_preprocessed[answer_col_prep].iloc[row_idx]).strip()
            
            # Skip if EITHER is empty/NaN
            if not question_text or question_text.lower() == 'nan':
                continue
            if not answer_text or answer_text.lower() == 'nan':
                continue
            if not question_prep or question_prep.lower() == 'nan':
                continue
            if not answer_prep or answer_prep.lower() == 'nan':
                continue
            
            self.all_questions.append({
                'row_idx': row_idx,
                'question': question_text,  # Original for display
                'ideal_answer': answer_text,  # Original for display
                'question_preprocessed': question_prep,  # For embedding
                'answer_preprocessed': answer_prep  # For embedding
            })
    
    def _create_new_progress(self):
        return {
            'asked_questions': [],
            'mastered_questions': [],
            'question_scores': {},
            'session_history': []
        }
    
    def _load_progress(self):
        """Load student progress."""
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        return self._create_new_progress()
                    return json.loads(content)
            except:
                return self._create_new_progress()
        return self._create_new_progress()
    
    def _save_progress(self):
        """Save progress."""
        try:
            temp_file = progress_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
            shutil.move(temp_file, progress_file)
        except Exception as e:
            print(f"Error saving progress: {e}")
    
    def get_random_question(self, avoid_asked=True):
        """Get random question from CSV."""
        available = [
            q for q in self.all_questions
            if not avoid_asked or q['row_idx'] not in self.progress['asked_questions']
        ]
        
        if not available:
            return None
        
        return random.choice(available)
    
    def evaluate_answer(self, question_data, student_answer):
        """Evaluate student answer - FRESH EMBEDDINGS VERSION."""
        print(f"\n{'='*70}")
        print("EVALUATING ANSWER (FRESH EMBEDDINGS)")
        print(f"{'='*70}")
        
        # STEP 1: Preprocess student answer
        print("  -> Preprocessing student answer...")
        preprocessed_student_answer = preprocessor.preprocess(student_answer)
        print(f"    Original: {student_answer[:100]}...")
        print(f"    Preprocessed: {preprocessed_student_answer[:100]}...")
        
        # STEP 2: Get preprocessed answer from CSV (same row index)
        print("  -> Getting preprocessed answer from CSV...")
        preprocessed_ideal_answer = question_data['answer_preprocessed']
        print(f"    Preprocessed ideal answer: {preprocessed_ideal_answer[:100]}...")
        
        # STEP 3: Generate FRESH embeddings for BOTH answers
        print("  -> Generating FRESH embeddings for comparison...")
        print("     - Embedding ideal answer (from CSV)...")
        ideal_answer_embedding = self.get_embedding(preprocessed_ideal_answer)
        
        print("     - Embedding student answer (preprocessed)...")
        student_embedding = self.get_embedding(preprocessed_student_answer)
        
        # STEP 4: Compare the fresh embeddings (apples-to-apples!)
        print("  -> Comparing student answer with ideal answer (fresh embeddings)...")
        similarity = cosine_similarity(
            student_embedding.reshape(1, -1),
            ideal_answer_embedding.reshape(1, -1)
        )[0][0]
        
        print(f"  -> Answer Similarity: {similarity:.2%}")
        
        # STEP 5: Debug check - are the texts identical?
        texts_identical = preprocessed_student_answer == preprocessed_ideal_answer
        if texts_identical:
            print(f"  [CHECK] Texts are IDENTICAL - similarity should be ~99%+")
        else:
            print(f"  [INFO] Texts are different")
        
        # Grade and update progress
        grade = self._get_grade(similarity)
        is_mastered = similarity >= MASTERY_THRESHOLD
        
        row_idx = question_data['row_idx']
        key_str = f"row{row_idx}"
        
        if key_str not in self.progress['question_scores']:
            self.progress['question_scores'][key_str] = []
        
        self.progress['question_scores'][key_str].append(float(similarity))
        
        if is_mastered and row_idx not in self.progress['mastered_questions']:
            self.progress['mastered_questions'].append(row_idx)
        
        if row_idx not in self.progress['asked_questions']:
            self.progress['asked_questions'].append(row_idx)
        
        self.progress['session_history'].append({
            'row_idx': row_idx,
            'question': question_data['question'],
            'similarity': float(similarity),
            'mastered': is_mastered,
            'timestamp': datetime.now().isoformat()
        })
        
        self._save_progress()
        
        return {
            'question': question_data['question'],
            'question_preprocessed': question_data['question_preprocessed'],
            'ideal_answer_csv': question_data['ideal_answer'],
            'ideal_answer_preprocessed': preprocessed_ideal_answer,
            'student_answer': student_answer,
            'preprocessed_answer': preprocessed_student_answer,
            'texts_identical': texts_identical,
            'similarity': similarity,
            'grade': grade,
            'mastered': is_mastered,
            'attempts': len(self.progress['question_scores'][key_str])
        }
    
    def _get_grade(self, similarity):
        """Convert similarity to letter grade."""
        if similarity >= 0.95:
            return 'A+ (Outstanding)'
        elif similarity >= 0.90:
            return 'A (Excellent)'
        elif similarity >= 0.85:
            return 'A- (Very Good)'
        elif similarity >= MASTERY_THRESHOLD:
            return 'B+ (Good - Mastered!)'
        elif similarity >= 0.75:
            return 'B (Above Average)'
        elif similarity >= 0.70:
            return 'C+ (Satisfactory)'
        elif similarity >= 0.60:
            return 'C (Adequate)'
        elif similarity >= 0.50:
            return 'D (Needs Improvement)'
        else:
            return 'F (Unsatisfactory)'
    
    def get_next_similar_question(self, current_question_data):
        """Get next similar question using vector DB."""
        print("\n  -> Finding next similar question...")
        
        # Use preprocessed question from CSV
        preprocessed_question = current_question_data['question_preprocessed']
        current_embedding = self.get_embedding(preprocessed_question)
        
        # Query questions collection for similar questions
        results = self.questions_collection.query(
            query_embeddings=[current_embedding.tolist()],
            n_results=20,
            include=['embeddings', 'distances', 'metadatas']
        )
        
        if not results['ids'][0]:
            return None
        
        # Try each result
        for distance, metadata in zip(results['distances'][0], results['metadatas'][0]):
            similarity = 1 - (distance ** 2) / 2
            
            # Skip if not similar enough
            if similarity < TOPIC_SIMILARITY_THRESHOLD:
                continue
            
            # Get row index from metadata
            if 'row_index' not in metadata:
                continue
            
            row_idx = metadata['row_index']
            
            # Skip if already asked or mastered
            if row_idx in self.progress['asked_questions']:
                continue
            if row_idx in self.progress['mastered_questions']:
                continue
            
            # Skip if it's the current question
            if row_idx == current_question_data['row_idx']:
                continue
            
            # Get question from BOTH CSVs
            question_text = str(self.df_original[question_col].iloc[row_idx]).strip()
            answer_text = str(self.df_original[answer_col].iloc[row_idx]).strip()
            question_prep = str(self.df_preprocessed[question_col_prep].iloc[row_idx]).strip()
            answer_prep = str(self.df_preprocessed[answer_col_prep].iloc[row_idx]).strip()
            
            if not question_text or question_text.lower() == 'nan':
                continue
            if not question_prep or question_prep.lower() == 'nan':
                continue
            
            print(f"  -> Found similar question (similarity: {similarity:.1%})")
            
            return {
                'row_idx': row_idx,
                'question': question_text,
                'ideal_answer': answer_text,
                'question_preprocessed': question_prep,
                'answer_preprocessed': answer_prep,
                'similarity': similarity
            }
        
        return None
    
    def check_topic_mastery(self, current_question_data):
        """Check if current topic cluster is mastered."""
        # Use preprocessed question from CSV
        preprocessed_question = current_question_data['question_preprocessed']
        current_embedding = self.get_embedding(preprocessed_question)
        
        results = self.questions_collection.query(
            query_embeddings=[current_embedding.tolist()],
            n_results=50,
            include=['distances', 'metadatas']
        )
        
        # Count similar questions
        similar_questions = []
        for distance, metadata in zip(results['distances'][0], results['metadatas'][0]):
            similarity = 1 - (distance ** 2) / 2
            
            if similarity >= TOPIC_SIMILARITY_THRESHOLD:
                if 'row_index' in metadata:
                    similar_questions.append(metadata['row_index'])
        
        # Add current question
        similar_questions.append(current_question_data['row_idx'])
        
        # Count mastered
        mastered_count = sum(
            1 for row_idx in similar_questions
            if row_idx in self.progress['mastered_questions']
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
    
    def get_next_question(self, current_question_data=None):
        """
        Adaptive question selection:
        1. If first question -> random
        2. Check topic mastery
        3. If mastered -> new random topic
        4. If not mastered -> similar question
        """
        if current_question_data is None:
            print("\n-> Selecting random first question...")
            return self.get_random_question(avoid_asked=True)
        
        # Check topic mastery
        topic_status = self.check_topic_mastery(current_question_data)
        
        print(f"\n{'='*70}")
        print("TOPIC STATUS")
        print(f"{'='*70}")
        print(f"  Cluster size: {topic_status['cluster_size']}")
        print(f"  Mastered: {topic_status['mastered_count']}")
        print(f"  Ratio: {topic_status['mastery_ratio']:.1%}")
        
        if topic_status['topic_mastered']:
            print(f"  [SUCCESS] Topic mastered!")
            print("\n-> Moving to NEW TOPIC (random question)")
            return self.get_random_question(avoid_asked=True)
        else:
            print(f"  [CONTINUE] Continue practicing this topic")
            print("\n-> Finding similar question from vector DB...")
            
            next_q = self.get_next_similar_question(current_question_data)
            
            if next_q:
                return next_q
            else:
                print("  No more similar questions, picking random...")
                return self.get_random_question(avoid_asked=True)
    
    def print_evaluation(self, evaluation):
        """Print formatted evaluation."""
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print(f"{'='*70}")
        
        print(f"\n[?] Question Asked (Original):")
        print(f"   {evaluation['question']}")
        
        print(f"\n[PREP] Question (Preprocessed from CSV):")
        print(f"   {evaluation['question_preprocessed'][:200]}...")
        
        print(f"\n[INPUT] Your Answer (Original):")
        print(f"   {evaluation['student_answer'][:200]}...")
        
        print(f"\n[PREP] Your Answer (Preprocessed at runtime):")
        print(f"   {evaluation['preprocessed_answer'][:200]}...")
        
        print(f"\n[IDEAL] Ideal Answer (Original CSV):")
        print(f"   {evaluation['ideal_answer_csv'][:200]}...")
        
        print(f"\n[PREP] Ideal Answer (Preprocessed from CSV):")
        print(f"   {evaluation['ideal_answer_preprocessed'][:200]}...")
        
        # Show text comparison
        if evaluation['texts_identical']:
            print(f"\n[MATCH] Preprocessed texts are IDENTICAL!")
        else:
            print(f"\n[DIFF] Preprocessed texts differ")
        
        print(f"\n{'-'*70}")
        print(f"[SCORE] Your Similarity Score: {evaluation['similarity']:.2%}")
        print(f"[GRADE] Grade: {evaluation['grade']}")
        print(f"[STATUS] Status: {'MASTERED!' if evaluation['mastered'] else 'Keep practicing'}")
        print(f"[ATTEMPTS] Attempts: {evaluation['attempts']}")
        print(f"{'-'*70}")
    
    def get_progress_summary(self):
        """Get progress summary."""
        total_asked = len(set(self.progress['asked_questions']))
        total_mastered = len(set(self.progress['mastered_questions']))
        
        if self.progress['question_scores']:
            all_scores = [
                score 
                for scores in self.progress['question_scores'].values()
                for score in scores
            ]
            avg_score = np.mean(all_scores)
        else:
            avg_score = 0
        
        return {
            'total_available': len(self.all_questions),
            'total_asked': total_asked,
            'total_mastered': total_mastered,
            'mastery_rate': total_mastered / total_asked if total_asked > 0 else 0,
            'average_score': avg_score,
            'total_attempts': len(self.progress['session_history'])
        }
    
    def print_progress(self):
        """Print progress summary."""
        summary = self.get_progress_summary()
        
        print(f"\n{'='*70}")
        print("PROGRESS SUMMARY")
        print(f"{'='*70}")
        
        print(f"\n[TOTAL] Total Questions Available: {summary['total_available']}")
        print(f"[ASKED] Unique Questions Asked: {summary['total_asked']}")
        print(f"[MASTERED] Questions Mastered: {summary['total_mastered']}")
        print(f"[RATE] Mastery Rate: {summary['mastery_rate']:.1%}")
        print(f"[AVG] Average Score: {summary['average_score']:.2%}")
        print(f"[ATTEMPTS] Total Attempts: {summary['total_attempts']}")

# ==================== INTERACTIVE QUIZ ====================
def run_interactive_quiz():
    """Run interactive quiz session."""
    quiz = AdaptiveQuizSystem(
        df_original,
        df_preprocessed,
        answers_collection, 
        questions_collection, 
        get_embedding
    )
    
    print(f"\n{'='*70}")
    print("ADAPTIVE SCIENCE QUIZ SYSTEM (FRESH EMBEDDINGS)")
    print(f"{'='*70}")
    print(f"\nKey Feature: Generate BOTH embeddings fresh for fair comparison!")
    print(f"\nWorkflow:")
    print("  1. Question from science_qa.csv (clean text)")
    print("  2. Get preprocessed question from preprocessed_science_qa.csv")
    print("  3. Student answers")
    print("  4. Preprocess student answer at runtime")
    print("  5. Get preprocessed ideal answer from CSV (same row)")
    print("  6. Generate FRESH embeddings for BOTH answers")
    print("  7. Compare fresh embeddings (apples-to-apples)")
    print("  8. Grade and find similar questions")
    print("  9. Repeat until topic mastered (80%+)")
    print(f"\nCommands: 'quit', 'progress', 'skip'")
    print(f"{'='*70}")
    
    current_question_data = None
    
    while True:
        question_data = quiz.get_next_question(current_question_data)
        
        if not question_data:
            print("\n[COMPLETE] Quiz complete! All questions answered.")
            break
        
        print(f"\n{'='*70}")
        print(f"QUESTION:")
        print(f"{'='*70}")
        print(f"{question_data['question']}")
        print(f"{'='*70}")
        
        if 'similarity' in question_data:
            print(f"(Topic similarity: {question_data['similarity']:.1%})")
        
        student_answer = input("\nYour answer: ").strip()
        
        if student_answer.lower() == 'quit':
            print("\nGoodbye! Progress saved.")
            break
        
        if student_answer.lower() == 'progress':
            quiz.print_progress()
            continue
        
        if student_answer.lower() == 'skip':
            print("Skipped")
            current_question_data = question_data
            continue
        
        if not student_answer:
            print("Please provide an answer")
            continue
        
        evaluation = quiz.evaluate_answer(question_data, student_answer)
        
        if 'error' in evaluation:
            print(f"\nError: {evaluation['error']}")
            continue
        
        quiz.print_evaluation(evaluation)
        current_question_data = question_data
        
        cont = input("\nContinue? (y/n): ").strip().lower()
        if cont != 'y' and cont != '':
            print("\nProgress saved!")
            break
    
    quiz.print_progress()

# ==================== MAIN ====================
if __name__ == "__main__":
    # ==================== TESTING ====================
    import sys
    from test import run_tests
    # Check if user wants to run tests
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        print("Running test suite...")
        
        # Initialize quiz system (without running interactive quiz)
        quiz = AdaptiveQuizSystem(
            df_original,
            df_preprocessed,
            answers_collection, 
            questions_collection, 
            get_embedding
        )
        
        # Run tests
        run_tests(quiz, df_original)
    else:
        # Run normal interactive quiz
        run_interactive_quiz()