from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="NousResearch/Hermes-3-Llama-3.1-8B",
    local_dir="./llama-model/hermes-3-llama-3_1-8B",
    local_dir_use_symlinks=False,  
    resume_download=True,         
    token=None                    
)
