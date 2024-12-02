import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class ResponseAnalyzer:
    def __init__(self):
        self.metrics = {}
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

    def analyze_length(self, response):
        """Analyze response length metrics."""
        return {
            'char_count': len(response),
            'word_count': len(response.split()),
            'average_word_length': np.mean([len(word) for word in response.split()])
        }

    def analyze_quality(self, response, reference=None):
        """Analyze response quality metrics."""
        metrics = {}
        
        # Sentiment analysis
        inputs = self.tokenizer(response, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        sentiment_score = torch.softmax(outputs.logits, dim=1)
        metrics['sentiment_score'] = sentiment_score.detach().numpy()[0]

        # Complexity metrics
        metrics.update({
            'sentence_count': len(response.split('.')),
            'unique_words': len(set(response.lower().split())),
            'vocabulary_richness': len(set(response.lower().split())) / len(response.split())
        })

        if reference:
            # Similarity with reference
            ref_encoding = self.tokenizer(reference, return_tensors="pt")
            resp_encoding = self.tokenizer(response, return_tensors="pt")
            similarity = self._calculate_cosine_similarity(
                self.model(**ref_encoding).logits,
                self.model(**resp_encoding).logits
            )
            metrics['reference_similarity'] = similarity.item()

        return metrics

    def analyze_topic(self, response):
        """Analyze topic and content focus."""
        # Topic analysis using transformers
        inputs = self.tokenizer(response, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        topic_scores = torch.softmax(outputs.logits, dim=1)
        
        return {
            'topic_distribution': topic_scores.detach().numpy()[0],
            'main_topic_confidence': float(topic_scores.max())
        }

    def _calculate_cosine_similarity(self, a, b):
        """Calculate cosine similarity between two tensor embeddings."""
        return torch.nn.functional.cosine_similarity(a, b)