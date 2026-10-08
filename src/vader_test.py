from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

text = input("Enter text: ")

scores = analyzer.polarity_scores(text)

print("\nSentiment Scores:")
print(scores)

compound = scores["compound"]

if compound >= 0.05:
    sentiment = "Positive"
elif compound <= -0.05:
    sentiment = "Negative"
else:
    sentiment = "Neutral"

print("\nFinal Sentiment:", sentiment)