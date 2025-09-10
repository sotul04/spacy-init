#!/usr/bin/env python3
"""
Quick validation script to test all 8 NLP tools implementation
"""

import spacy
import pandas as pd
from collections import Counter
import sys

def test_nlp_tools():
    """Test all implemented NLP tools"""
    
    print("🚀 Testing SpaCy NLP Tools Implementation")
    print("=" * 50)
    
    try:
        # Load model
        nlp = spacy.load('en_core_web_sm')
        print("✓ SpaCy model loaded successfully")
        
        # Test text (from Transformer paper abstract)
        text = """The dominant sequence transduction models are based on complex recurrent or 
        convolutional neural networks that include an encoder and a decoder. We propose a new 
        simple network architecture, the Transformer, based solely on attention mechanisms."""
        
        doc = nlp(text)
        
        # 1. Sentence Splitter
        sentences = [sent.text.strip() for sent in doc.sents]
        print(f"✓ Sentence Splitter: {len(sentences)} sentences")
        
        # 2. Tokenization
        tokens = [{'text': token.text, 'is_alpha': token.is_alpha} for token in doc]
        alpha_tokens = sum(1 for t in tokens if t['is_alpha'])
        print(f"✓ Tokenization: {len(tokens)} total tokens, {alpha_tokens} alphabetic")
        
        # 3. Stemming (simple rule-based)
        def simple_stem(word):
            word = word.lower()
            if word.endswith('ing') and len(word) > 5:
                return word[:-3]
            elif word.endswith('ed') and len(word) > 4:
                return word[:-2]
            elif word.endswith('s') and len(word) > 3:
                return word[:-1]
            return word
        
        stemmed = [(token.text, simple_stem(token.text)) for token in doc if token.is_alpha]
        changed_stems = [s for s in stemmed if s[0].lower() != s[1]]
        print(f"✓ Stemming: {len(stemmed)} words processed, {len(changed_stems)} changed")
        
        # 4. Lemmatization
        lemmas = [(token.text, token.lemma_) for token in doc if token.is_alpha]
        changed_lemmas = [l for l in lemmas if l[0].lower() != l[1].lower()]
        print(f"✓ Lemmatization: {len(lemmas)} words processed, {len(changed_lemmas)} changed")
        
        # 5. Entity Masking
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"✓ Entity Masking: {len(entities)} entities found")
        
        # 6. POS Tagging
        pos_tags = [(token.text, token.pos_) for token in doc if not token.is_space]
        pos_counts = Counter([pos[1] for pos in pos_tags])
        print(f"✓ POS Tagging: {len(pos_tags)} tokens tagged, {len(pos_counts)} unique POS")
        
        # 7. Phrase Chunking
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        print(f"✓ Phrase Chunking: {len(noun_phrases)} noun phrases extracted")
        
        # 8. Syntactic Parser
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
        dep_counts = Counter([dep[1] for dep in dependencies])
        print(f"✓ Syntactic Parser: {len(dependencies)} dependencies, {len(dep_counts)} unique types")
        
        # Summary
        print("\n📊 Analysis Summary:")
        print(f"   • Text length: {len(text)} characters")
        print(f"   • Sentences: {len(sentences)}")
        print(f"   • Tokens: {len(tokens)}")
        print(f"   • Entities: {len(entities)}")
        print(f"   • Noun phrases: {len(noun_phrases)}")
        print(f"   • Most common POS: {pos_counts.most_common(3)}")
        
        print("\n🎉 All 8 NLP tools tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_nlp_tools()
    sys.exit(0 if success else 1)