import torch
from transformers import RobertaTokenizer

try:
    # Load the saved TorchScript model
    model_path = "model.pt"
    print(f"Loading TorchScript model from {model_path}")
    model = torch.jit.load(model_path)
    
    # Load the tokenizer
    tokenizer_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    
    # Example text samples
    texts = [
        "Operating profit totaled EUR 9.4 mn, down from EUR 11.7 mn in 2004.",
        "The company's revenue increased by 25% compared to last year.",
        "The company reported a significant loss this quarter."
    ]
    
    print("Testing model with example inputs:")
    # Test the model with each text
    for text in texts:
        print(f"\nInput: '{text}'")
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Run the model
        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            
            # Print raw output type for debugging
            print(f"  Raw output type: {type(outputs)}")
            
            # Handle dictionary output
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
                print(f"  Logits shape: {logits.shape}")
                
                # Apply softmax to get probabilities
                predictions = torch.nn.functional.softmax(logits, dim=-1)
                
                # Map indices to labels
                id2label = {0: "negative", 1: "neutral", 2: "positive"}
                
                # Print the predictions
                for idx, score in enumerate(predictions[0]):
                    print(f"  {id2label[idx]}: {score.item():.4f}")
                
                # Get the highest scoring prediction
                predicted_class = torch.argmax(predictions, dim=-1).item()
                print(f"  Predicted sentiment: {id2label[predicted_class]}")
            elif isinstance(outputs, dict):
                print(f"  Dictionary keys: {outputs.keys()}")
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key} shape: {value.shape}")
            else:
                print(f"  Unexpected output format")
    
    print("\nModel testing completed successfully!")
    
except Exception as e:
    import traceback
    print(f"Error testing model: {str(e)}")
    traceback.print_exc() 