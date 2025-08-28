import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import os
import subprocess
import shutil
import tempfile

# Download and convert the distilRoberta model to TorchScript.

## run with:
# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
# python3 convert_to_torchscript.py

try:
    # Model information
    model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    repo_url = "https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    
    print(f"Checking out model repository: {repo_url}")
    
    # Create the model directory name (without creating a subdirectory)
    model_dir = "."
    
    # Clone the repository directly into current directory if requested files don't exist
    model_files = ["config.json", "tokenizer.json", "pytorch_model.bin"]
    files_exist = all(os.path.exists(os.path.join(model_dir, f)) for f in model_files)
    
    if not files_exist:
        print("Model files not found in current directory. Cloning repository...")
        
        # Use a different method to download the files
        # We'll clone into the current directory but only specific model files
        repo_name = repo_url.split("/")[-1]
        
        # Use git to clone the repository directly to current directory
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, repo_name],
            check=True
        )
        
        # Now copy the important files from the repository to the current directory
        source_dir = os.path.join(".", repo_name)
        print(f"Copying model files from {source_dir} to current directory...")
        
        for item in os.listdir(source_dir):
            src_path = os.path.join(source_dir, item)
            dst_path = os.path.join(".", item)
            
            if item != ".git":
                if os.path.isdir(src_path):
                    if os.path.exists(dst_path):
                        shutil.rmtree(dst_path)
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
        
        # Remove the cloned repository directory
        shutil.rmtree(source_dir, ignore_errors=True)
        print("Model files copied successfully")
    else:
        print("Model files already exist in current directory")
    
    print(f"Loading model: {model_name}")
    
    # Use the model_name instead of local files
    # This ensures the model is loaded correctly from Hugging Face Hub
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create an example input
    sample_text = "Operating profit totaled EUR 9.4 mn, down from EUR 11.7 mn in 2004."
    print(f"Creating example input with text: '{sample_text}'")
    
    inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    print("Tracing model - this may take a moment...")
    # Trace the model with strict=False to handle dictionary outputs
    with torch.no_grad():
        traced_model = torch.jit.trace(
            model, 
            (inputs["input_ids"], inputs["attention_mask"]),
            strict=False
        )
    
    # Save the traced model
    output_path = "model.pt"
    traced_model.save(output_path)
    
    print(f"Model successfully converted to TorchScript and saved as {output_path}")
    print(f"Full path: {os.path.abspath(output_path)}")
    
except Exception as e:
    print(f"Error converting model: {str(e)}") 