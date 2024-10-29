from huggingface_hub import snapshot_download

# Téléchargez le modèle dans le dossier 'llm_model_path'
snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.1", local_dir="mistral_llm_model_path")
