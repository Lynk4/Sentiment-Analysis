import pandas as pd
from transformers import pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
 # type: ignore
df = pd.read_csv('/Users/kant/Desktop/Sentiment_Analysis.csv')

df = df.sample(200)
print(df)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

texts = list(df.content.values)
results = nlp(texts)
print (results)

for text, result, score in zip(texts, results, results):
    print(f"Text: {text}, Sentiment: {result['label']}, score: {result['score']}")


df['sentiment'] = [result['label'] for result in results]

