import torch
from sentiment import model, tfidf



text = "i am happy"


text = tfidf.transform([text])


result = model.predict(text)[0]
print(result)