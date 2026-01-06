# Visual Fashion Recommendation System

This repository implements a **content-based fashion recommendation system**
driven by **deep visual embeddings** and **vector similarity search**.

The system recommends visually similar fashion products by extracting
high-dimensional representations from product images using a pretrained
**ResNet-based CNN** and ranking candidates via **cosine similarity**.

---

## 📌 Problem Statement

In many real-world recommendation scenarios (cold-start users, new catalogs,
limited interaction data), collaborative filtering is insufficient.

This project focuses on **visual similarity–based retrieval** as a strong
alternative, enabling:
- Image-driven product discovery
- Catalog exploration without user history
- Fast, interpretable recommendations based on visual features

---

## 🧠 Methodology

### 1️⃣ Feature Extraction
- Product images are passed through a pretrained **ResNet CNN**
- Deep feature vectors are extracted from the final embedding layer
- These vectors encode semantic and visual attributes of products

### 2️⃣ Embedding Store
- All product embeddings are **precomputed**
- A mapping between embeddings and product metadata is maintained

### 3️⃣ Similarity Search
- **Cosine similarity** is used to compute nearest neighbors
- Top-K visually similar products are retrieved efficiently

### 4️⃣ Filtering & Business Logic
- Gender-based filtering
- Category-level constraints
- Rule-based size recommendation heuristics

### 5️⃣ User Interface
- Interactive **Streamlit** frontend
- Product selection and image-driven recommendation workflow

---

## 🧪 System Capabilities

- Content-based recommendation without collaborative signals
- Image similarity search using deep embeddings
- Lightweight virtual try-on overlay (experimental)
- Size suggestion based on predefined heuristics
- Fast retrieval using precomputed vector representations

---

## 📄 Project Structure

```
fashion-visual-recommender-system/
├── app.py # Streamlit application
├── features_resnet50.pt # Precomputed image embeddings
├── ids_resnet50.txt # Mapping between embeddings and products
├── styles.csv # Product metadata
├── temp.ipynb # Experimentation / analysis notebook
├── requirements.txt # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Run the Application
```bash
streamlit run app.py
```

The application will launch locally in your browser.

---

## 📋 Use Cases

- Visual product discovery
- Cold-start recommendation scenarios
- Fashion catalog exploration
- Similarity-based retrieval systems
- Applied computer vision pipelines

---

## ⚠️ Limitations

- Purely content-based (no collaborative filtering)
- Recommendation quality depends on visual embedding quality
- No online learning or user feedback loop
- Designed as an applied ML system, not a production-scale recommender

---

## 🔮 Future Extensions

- Hybrid recommendation (visual + collaborative filtering)
- FAISS-based approximate nearest neighbor search
- Multimodal embeddings (image + text)
- Model fine-tuning on fashion-specific datasets
- Deployment as a REST inference API

---

## 📃 License

MIT License

---

## ✅ How This Project Reads to a Recruiter

They see:
- ✅ Deep learning (CNNs, embeddings)
- ✅ Recommendation systems
- ✅ Vector similarity search
- ✅ Applied ML (not toy notebooks)
- ✅ Clear system design & limitations

This **complements** StyleScout perfectly:
- **StyleScout** → large, multimodal, system-level BTP
- **This repo** → focused, applied ML fundamentals
