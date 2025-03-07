import spacy
import json
from spacy.training import Example

#loading dataset
with open("E:/00Projects_new/Conversational bot/Conversational-AI-Bot/intents.json","r")as f:
    intents = json.load(f)
    
#training data
training_data=[]
for intent, data in intents.items():
    for pattern in data["patterns"]:
        training_data.append((pattern,intent))
        
#loading spacy blank model
nlp = spacy.blank("en")
if "textcat" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat",last=True)
    
#adding labels
for _, intent in training_data:
    textcat.add_label(intent)

#train the model
examples=[]
for text, intent in training_data:
    doc=nlp.make_doc(text)
    example = Example.from_dict(doc,{"cats":{intent: 1.0}})
    examples.append(example)
    
#training initialize
nlp.initialize()
for i in range(10):
    nlp.update(examples)
    
#saving model
nlp.to_disk("intent_model")