"""
Enhanced Topic Distribution Analysis with Real Topic Names
"""

import pandas as pd
import numpy as np
from collections import Counter
import chromadb
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== LOAD DATA ====================
df_original = pd.read_csv("science_qa.csv")

def find_column(df, name):
    for col in df.columns:
        if col.lower() == name.lower():
            return col
    return None

question_col = find_column(df_original, 'question')
answer_col = find_column(df_original, 'answer')

# ==================== TOPIC DETECTION ====================
print("="*70)
print("INTELLIGENT TOPIC DETECTION")
print("="*70)

# Load embeddings from ChromaDB
client = chromadb.PersistentClient(path="vector_db")
questions_collection = client.get_collection("science_qa_questions")

all_data = questions_collection.get(include=['embeddings', 'metadatas'])
embeddings = np.array(all_data['embeddings'])

print(f"\n✓ Loaded {len(embeddings)} question embeddings")

# Perform K-means clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings)

# ==================== ANALYZE EACH CLUSTER ====================

def extract_keywords(text):
    """Extract meaningful keywords from text."""
    import re
    from nltk.corpus import stopwords
    
    if pd.isna(text):
        return []
    
    # Convert to lowercase and extract words
    text = str(text).lower()
    words = re.findall(r'\b[a-z]{3,}\b', text)  # Words with 3+ letters
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    keywords = [w for w in words if w not in stop_words]
    
    return keywords

# Define topic indicators
TOPIC_KEYWORDS = {
    'Biology': [
        'cell', 'cells', 'dna', 'rna', 'gene', 'genes', 'protein', 'proteins',
        'organism', 'organisms', 'tissue', 'organ', 'photosynthesis', 'respiration',
        'bacteria', 'virus', 'evolution', 'species', 'plant', 'plants', 'animal',
        'animals', 'enzyme', 'mitosis', 'meiosis', 'chromosome', 'genetics',
        'metabolism', 'ecosystem', 'biodiversity', 'taxonomy', 'biology'
    ],
    'Chemistry': [
        'atom', 'atoms', 'molecule', 'molecules', 'element', 'elements', 'compound',
        'compounds', 'reaction', 'reactions', 'chemical', 'acid', 'base', 'bond',
        'bonds', 'electron', 'electrons', 'proton', 'neutron', 'periodic', 'table',
        'oxygen', 'hydrogen', 'carbon', 'solution', 'mixture', 'ionic', 'covalent',
        'oxidation', 'reduction', 'ph', 'chemistry', 'substance', 'catalyst'
    ],
    'Physics': [
        'force', 'forces', 'energy', 'motion', 'velocity', 'acceleration', 'mass',
        'weight', 'gravity', 'friction', 'momentum', 'wave', 'waves', 'light',
        'sound', 'electricity', 'magnetic', 'magnetism', 'heat', 'temperature',
        'pressure', 'density', 'speed', 'distance', 'time', 'power', 'work',
        'newton', 'physics', 'radiation', 'quantum', 'particle', 'nuclear'
    ],
    'Earth Science': [
        'earth', 'planet', 'planets', 'rock', 'rocks', 'mineral', 'minerals',
        'soil', 'weather', 'climate', 'atmosphere', 'ocean', 'water', 'cycle',
        'erosion', 'volcano', 'earthquake', 'plate', 'tectonic', 'fossil',
        'fossils', 'geology', 'geography', 'layer', 'crust', 'mantle', 'core',
        'sediment', 'igneous', 'metamorphic', 'sedimentary', 'mountain'
    ]
}

def detect_topic_from_keywords(keywords):
    """Detect topic based on keyword frequency."""
    topic_scores = {}
    
    for topic, topic_words in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in topic_words)
        topic_scores[topic] = score
    
    # Return topic with highest score
    if max(topic_scores.values()) > 0:
        return max(topic_scores, key=topic_scores.get)
    else:
        return 'General Science'

# Analyze each cluster
cluster_info = {}

for cluster_id in range(n_clusters):
    # Get indices of questions in this cluster
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    
    # Collect all keywords from questions in this cluster
    all_keywords = []
    sample_questions = []
    
    for idx in cluster_indices:
        row_idx = all_data['metadatas'][idx].get('row_index', idx)
        
        if row_idx < len(df_original):
            question_text = str(df_original[question_col].iloc[row_idx])
            answer_text = str(df_original[answer_col].iloc[row_idx])
            
            # Extract keywords
            keywords = extract_keywords(question_text + " " + answer_text)
            all_keywords.extend(keywords)
            
            # Save first 3 questions as samples
            if len(sample_questions) < 3:
                sample_questions.append(question_text)
    
    # Count most common keywords
    keyword_freq = Counter(all_keywords)
    top_keywords = keyword_freq.most_common(10)
    
    # Detect topic
    detected_topic = detect_topic_from_keywords(all_keywords)
    
    cluster_info[cluster_id] = {
        'topic_name': detected_topic,
        'count': len(cluster_indices),
        'top_keywords': top_keywords,
        'sample_questions': sample_questions
    }

# ==================== DISPLAY RESULTS ====================
print("\n" + "="*70)
print("TOPIC ANALYSIS RESULTS")
print("="*70)

for cluster_id, info in cluster_info.items():
    percentage = (info['count'] / len(embeddings)) * 100
    
    print(f"\n{'━'*70}")
    print(f"CLUSTER {cluster_id}: {info['topic_name'].upper()}")
    print(f"{'━'*70}")
    print(f"Questions: {info['count']} ({percentage:.1f}%)")
    
    print(f"\n📌 Top Keywords:")
    for word, count in info['top_keywords'][:5]:
        print(f"   • {word:15s} ({count} occurrences)")
    
    print(f"\n📝 Sample Questions:")
    for i, q in enumerate(info['sample_questions'], 1):
        print(f"   {i}. {q[:80]}...")

# ==================== ENHANCED VISUALIZATION ====================
print("\n" + "="*70)
print("GENERATING ENHANCED VISUALIZATIONS")
print("="*70)

# Prepare data for visualization
topic_names = [cluster_info[i]['topic_name'] for i in range(n_clusters)]
topic_counts = [cluster_info[i]['count'] for i in range(n_clusters)]

# Sort by count (descending)
sorted_data = sorted(zip(topic_names, topic_counts), key=lambda x: x[1], reverse=True)
topic_names_sorted, topic_counts_sorted = zip(*sorted_data)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Pie chart
colors = sns.color_palette('Set2', n_clusters)
axes[0].pie(topic_counts_sorted, 
            labels=topic_names_sorted,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 11, 'weight': 'bold'})
axes[0].set_title('Topic Distribution', fontsize=14, fontweight='bold', pad=20)

# 2. Bar chart
bars = axes[1].bar(range(len(topic_names_sorted)), topic_counts_sorted, color=colors, edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('Topic', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Number of Questions', fontsize=12, fontweight='bold')
axes[1].set_title('Questions per Topic', fontsize=14, fontweight='bold', pad=20)
axes[1].set_xticks(range(len(topic_names_sorted)))
axes[1].set_xticklabels(topic_names_sorted, rotation=30, ha='right')
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('topic_distribution_labeled.png', dpi=300, bbox_inches='tight')
print("\n✓ Enhanced topic distribution saved as 'topic_distribution_labeled.png'")

# ==================== SUMMARY TABLE ====================
print("\n" + "="*70)
print("📊 TOPIC DISTRIBUTION SUMMARY (FOR YOUR REPORT)")
print("="*70)

print("\nTable: Topic Distribution Analysis")
print("─"*70)
print(f"{'Topic':<20} {'Count':>10} {'Percentage':>15} {'Top Keywords':<25}")
print("─"*70)

for name, count in sorted_data:
    percentage = (count / len(embeddings)) * 100
    # Find cluster with this name
    cluster_id = [k for k, v in cluster_info.items() if v['topic_name'] == name][0]
    top_3_keywords = ', '.join([kw for kw, _ in cluster_info[cluster_id]['top_keywords'][:3]])
    print(f"{name:<20} {count:>10} {percentage:>14.1f}% {top_3_keywords:<25}")

print("─"*70)
print(f"{'TOTAL':<20} {len(embeddings):>10} {'100.0%':>15}")
print("─"*70)

# ==================== EXPORT DATA ====================
topic_distribution_data = {
    'clusters': []
}

for cluster_id in range(n_clusters):
    topic_distribution_data['clusters'].append({
        'topic_name': cluster_info[cluster_id]['topic_name'],
        'count': int(cluster_info[cluster_id]['count']),
        'percentage': round((cluster_info[cluster_id]['count'] / len(embeddings)) * 100, 2),
        'top_keywords': [{'word': word, 'count': count} for word, count in cluster_info[cluster_id]['top_keywords']],
        'sample_questions': cluster_info[cluster_id]['sample_questions']
    })

import json
with open('topic_distribution.json', 'w') as f:
    json.dump(topic_distribution_data, f, indent=2)

print("\n✓ Topic distribution data exported to 'topic_distribution.json'")

# Create CSV summary
topic_df = pd.DataFrame([
    {
        'Topic': name,
        'Question Count': count,
        'Percentage': f"{(count / len(embeddings)) * 100:.1f}%"
    }
    for name, count in sorted_data
])

topic_df.to_csv('topic_distribution.csv', index=False)
print("✓ Topic distribution table exported to 'topic_distribution.csv'")

print("\n" + "="*70)
print("✅ TOPIC ANALYSIS COMPLETE!")
print("="*70)
print("\nFor your report, use:")
print("  • Biology: XX questions (XX.X%)")
print("  • Chemistry: XX questions (XX.X%)")
print("  • Physics: XX questions (XX.X%)")
print("  • Earth Science/General: XX questions (XX.X%)")