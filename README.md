# Search-Engine-And-Recommendation-system-on-Amazon-Product

🛒 Amazon Product Search Engine & Recommendation System

A simple and smart Search Engine + Recommendation System built using TF-IDF, Cosine Similarity, NLP, and Streamlit.
The goal of this project is to help users search Amazon products and get relevant recommendations using text similarity.

⭐ Project Features

🔍 Product Search — Find products by entering any keyword or sentence.

🧠 TF-IDF Text Vectorization — Converts product titles & descriptions into numerical vectors.

📏 Cosine Similarity Ranking — Shows the most relevant products at the top.

🛍️ Content-Based Recommendations — Suggests similar products based on text similarity.

⚡ Fast & lightweight — Works in real time using optimized preprocessing.

🎨 Clean UI — Built using Streamlit.

🧱 Project Structure
📁 Amazon-Product-Search-Engine
│── app.py                  # Main Streamlit app
│── amazon_product.csv      # Product dataset (or sample file)
│── img.png                 # Amazon logo for UI
│── requirements.txt        # Python dependencies
│── README.md               # This documentation
│── search engine on amazon product dataset.ipynb   # EDA & model development
│── project report.docx     # Detailed analysis document

🧠 Models & Techniques Used
1. Text Preprocessing

Lowercasing

Tokenization (NLTK)

Stopword removal

Stemming / Lemmatization

Cleaning special characters

2. TF-IDF Vectorizer

Used to convert product titles + descriptions into machine-understandable vectors.

TF-IDF(selected corpus) → vector representation of text

3. Cosine Similarity

Used to measure similarity between:

User query → Product descriptions

Product A → Product B (for recommendations)

similarity = cos(theta between two TF-IDF vectors)

4. Content-Based Recommendation

Shows products similar to:

User search query

Any selected product

🚀 How It Works (Simple Explanation)

User enters a keyword like:
“wireless headphones”

TF-IDF converts the keyword & all product texts into vectors.

Cosine similarity compares the query vector with all product vectors.

Products with highest similarity scores are shown at the top.

You also get related product recommendations.

🖥️ How to Run Locally
pip install -r requirements.txt
streamlit run app.py


Make sure the dataset (amazon_product.csv) and img.png are in the same folder as app.py.

🌐 Deployment (Streamlit Cloud)

Push files to GitHub

Go to share.streamlit.io

Select repo → Select app.py

Deploy

Streamlit automatically installs all packages from requirements.txt.

📊 Dataset Info

Contains Amazon product details

Columns: Title, Description, Category, etc.

Used for EDA, text cleaning, and building TF-IDF vectors.

💡 Future Improvements

Use BERT / Sentence Transformers for semantic search

Add Filters (price, category, rating)

Use FAISS for faster similarity search

Add user behavior based recommendations

👨‍💻 Author

Raushan Kumar
B.S (CSDA), IIT Patna
Passionate about Machine Learning, NLP, and Search Systems.
