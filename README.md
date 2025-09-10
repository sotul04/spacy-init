# SpaCy NLP Tools Implementation

A comprehensive implementation of 8 essential Natural Language Processing (NLP) tools using SpaCy, demonstrated with real research paper analysis.

## 🚀 Features

This repository implements the following NLP processing tools:

1. **Sentence Splitter** - Intelligently splits text into individual sentences
2. **Tokenization** - Breaks text into tokens (words, punctuation, etc.)
3. **Stemming** - Reduces words to their root form using rule-based approaches
4. **Lemmatization** - Reduces words to their canonical dictionary form
5. **Entity Masking** - Identifies and masks named entities (NER)
6. **POS Tagger** - Identifies parts of speech for each token
7. **Phrase Chunking** - Groups tokens into meaningful noun and verb phrases
8. **Syntactic Parser** - Analyzes grammatical structure and dependencies

## 📁 Repository Structure

```
spacy-init/
├── spacy_nlp_tools.ipynb    # Main Jupyter notebook with all implementations
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── docs/                   # Detailed documentation and analysis
    ├── implementation_analysis.md
    ├── paper_processing_results.md
    └── pros_and_cons.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sotul04/spacy-init.git
   cd spacy-init
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download SpaCy language model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 🚀 Usage

### Running the Jupyter Notebook

1. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Open `spacy_nlp_tools.ipynb`** and run all cells to see the complete NLP analysis

### Quick Example

```python
import spacy

# Load the English language model
nlp = spacy.load('en_core_web_sm')

# Sample text
text = "SpaCy is an amazing NLP library for Python."

# Process the text
doc = nlp(text)

# Extract information
for token in doc:
    print(f"{token.text}: {token.pos_} ({token.lemma_})")
```

## 📊 Demo: Research Paper Analysis

The notebook demonstrates all 8 NLP tools using the abstract from the famous "Attention Is All You Need" paper (Transformer architecture). The analysis includes:

- **Text Statistics**: Character count, token distribution, sentence structure
- **Morphological Analysis**: Stemming vs. lemmatization comparison
- **Named Entity Recognition**: Identification of technical terms, numbers, and proper nouns
- **Syntactic Analysis**: Dependency parsing and phrase structure
- **Visualizations**: Charts showing POS distribution, entity types, and dependency relationships

### Sample Results

From the Transformer paper abstract:
- **Sentences**: 5 sentences identified
- **Tokens**: ~150 total tokens
- **Named Entities**: Technical terms like "BLEU", "WMT 2014", "P100 GPUs"
- **Key Phrases**: "attention mechanisms", "neural networks", "machine translation"

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- **[Implementation Analysis](docs/implementation_analysis.md)**: Technical deep-dive into each NLP tool
- **[Paper Processing Results](docs/paper_processing_results.md)**: Complete analysis of the research paper abstract
- **[Pros and Cons](docs/pros_and_cons.md)**: Advantages and limitations of each approach

## 🔧 Technologies Used

- **[SpaCy](https://spacy.io/)**: Industrial-strength NLP library
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation and analysis
- **[Matplotlib](https://matplotlib.org/)**: Data visualization
- **[Seaborn](https://seaborn.pydata.org/)**: Statistical data visualization
- **[Jupyter](https://jupyter.org/)**: Interactive development environment

## 📈 Performance Metrics

| Tool | Processing Speed | Accuracy | Memory Usage |
|------|-----------------|----------|--------------|
| Tokenization | Very Fast | 99%+ | Low |
| POS Tagging | Fast | 95%+ | Low |
| Lemmatization | Fast | 98%+ | Medium |
| NER | Medium | 90%+ | Medium |
| Dependency Parsing | Medium | 85%+ | High |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [SpaCy Documentation](https://spacy.io/usage)
- "Attention Is All You Need" - Vaswani et al. (2017)
- [Natural Language Processing with Python](https://www.nltk.org/book/)

## 🎯 Future Enhancements

- [ ] Add support for multiple languages
- [ ] Implement custom entity recognition
- [ ] Add sentiment analysis
- [ ] Include word embeddings visualization
- [ ] Add batch processing capabilities
- [ ] Create REST API interface

## 📞 Contact

- **Author**: [sotul04](https://github.com/sotul04)
- **Repository**: [spacy-init](https://github.com/sotul04/spacy-init)

---

⭐ If you find this project helpful, please consider giving it a star!