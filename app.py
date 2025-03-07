from flask import Flask, request, jsonify
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import json

app = Flask(__name__)

# Load NLP and fine-tuned model
nlp = spacy.load("intent_model")
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_gpt")

# ✅ No need to specify `from_safetensors=True` - it automatically detects safetensors
model = AutoModelForCausalLM.from_pretrained("fine_tuned_gpt")

# Load intents
with open("E:/00Projects_new/Conversational bot/Conversational-AI-Bot/intents.json", "r") as f:
    intents = json.load(f)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        return jsonify({"error": "Use POST with a JSON payload"}), 400

    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "Message is required"}), 400
    
    # Predict intent
    doc = nlp(user_input)
    intent = max(doc.cats, key=doc.cats.get)

    # Generate response
    if intent in intents:
        response = random.choice(intents[intent]["responses"])
    else:
        inputs = tokenizer(user_input, return_tensors="pt")
        output = model.generate(**inputs)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
