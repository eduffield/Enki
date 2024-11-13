import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def translate_babylonian(text, model, tokenizer, device):
    inputs = tokenizer(f"translate Babylonian to English: {text}", return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model_name = './results/checkpoint-11500'  # Replace with your trained model path
    tokenizer_name = './results/checkpoint-11500'  # Replace with your tokenizer path if different from model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    babylonian_text = "e2 szu szum2#-ma#" # have fresh water to drink.
    english_translation = translate_babylonian(babylonian_text, model, tokenizer, device)
    print(f"English translation: {english_translation}")

if __name__ == "__main__":
    main()
