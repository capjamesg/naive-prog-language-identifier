import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

train = []
classes = []

def load_data(path, ext):
    """Download data from path with extension ext."""
    data = []

    for file in os.listdir(path):
        if file.endswith(ext):
            with open(os.path.join(path, file), "r") as f:
                data.append(f.read())
            
    return data

classes = {
    "python": {
        "dir": "../py",
        "ext": ".py"
    },
    "markdown": {
        "dir": "../docs",
        "ext": ".md"
    },
    "ruby": {
        "dir": ".",
        "ext": ".rb"
    }
}

for key, value in classes.items():
    data = load_data(value["dir"], value["ext"])
    train += data
    classes[key]["data"] = data

train_class = []

for key, value in classes.items():
    for _ in value["data"]:
        train_class.append(key)

model = MultinomialNB()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train)
model.fit(X, train_class)

def predict(text):
    x = vectorizer.transform([text])

    return model.predict(x)[0]

print(predict("::INFO"))
