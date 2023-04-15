import nltk

nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    sentiment = 'positive' if scores['compound'] > 0 else 'negative' if scores['compound'] < 0 else 'neutral'
    return sentiment

text = input("Enter a sentence to analyze sentiment: ")

sentiment = analyze_sentiment(text)

print("Sentiment of the input text is: " + sentiment)
