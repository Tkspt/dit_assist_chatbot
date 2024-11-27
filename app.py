import gradio as gr
from rag_system import RAG  # Importer la classe RAG

# Initialiser l'instance de RAG
rag = RAG()

# Fonction utilisée par l'interface Gradio
def generate_response(prompt):
    return rag.retrieve_response(prompt)

# Définir l'interface Gradio
interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Posez une question sur les formations au DIT"),
    outputs=gr.Textbox(label=""),
    title="Agent conversationnel du DIT"
)

# Lancer l'interface
if __name__ == "__main__":
    interface.launch()
