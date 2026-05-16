from transformers import pipeline
import pandas as pd

class SentimentAnalyzer:
    def __init__(self):
        # Using a financial sentiment model from HuggingFace
        import torch
        torch.set_num_threads(1)
        self.analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            framework="pt"
        )

    def analyze_news(self, headlines):
        """
        Analyzes a list of news headlines and returns an average sentiment score.
        """
        if not headlines:
            return {"sentiment": "Neutral", "score": 0.0}
            
        results = self.analyzer(headlines)
        
        # Map results to numerical scores
        score_map = {"positive": 1, "neutral": 0, "negative": -1}
        total_score = sum(score_map[res['label']] * res['score'] for res in results)
        avg_score = total_score / len(headlines)
        
        sentiment = "Neutral"
        if avg_score > 0.2: sentiment = "Positive"
        elif avg_score < -0.2: sentiment = "Negative"
        
        return {"sentiment": sentiment, "score": avg_score}

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    headlines = [
        "Reliance shares surge after strong quarterly results",
        "TCS faces margin pressure amid global tech slowdown",
        "NIFTY touches all-time high as bank stocks rally"
    ]
    print(analyzer.analyze_news(headlines))
