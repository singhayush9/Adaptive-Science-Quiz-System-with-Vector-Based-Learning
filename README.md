# Adaptive-Science-Quiz-System-with-Vector-Based-Learning

This project is an AI-powered adaptive learning system that dynamically changes question difficulty based on student performance. The system uses vector embeddings and knowledge modelling to understand learning capabilities and adjust quizzes automatically.

---

## Concept

Traditional quizzes are static. However, learning requires personalization.  
This project builds a system capable of:

✔ Difficulty adjustment  
✔ Topic-level understanding  
✔ Personalized practice  
✔ Intelligent question recommendation  

---

## What Makes It Adaptive?

- If a learner answers correctly → harder questions are shown
- If the learner struggles → easier questions are served
- The system keeps updating learner knowledge state using vector similarity

---

## Technologies Used

- Python
- Machine Learning
- NLP Vector Embeddings
- JSON / CSV question dataset
- Streamlit (optional UI)

---

## System Flow

1️⃣ User selects topic  
2️⃣ Question fetched based on vector similarity  
3️⃣ System evaluates answer  
4️⃣ Difficulty is recalculated  
5️⃣ New suitable question is recommended  

---

## Folder Structure (example)

data/
adaptive_engine/
quiz_app.py
vector_embeddings.py
requirements.txt


---

## Installation


git clone https://github.com/singhayush9/Adaptive-Science-Quiz-System-with-Vector-Based-Learning.git

cd Adaptive-Science-Quiz-System-with-Vector-Based-Learning

pip install -r requirements.txt

## ▶️ Run

python quiz_app.py

streamlit run app.py

## Output

Score tracking
Difficulty curve
Adaptive question patterns
Learning performance analysis

## Use Cases

E-learning platforms
Smart education systems
Personalized tutoring
AI based learning modules

## Future Improvements

Topic recommendation
Voice based quiz
More advanced NLP embeddings
Integration with curriculum systems
