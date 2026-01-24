from huggingface_hub import InferenceClient
from config import Config
from flask import current_app
import time

class AIService:
    @staticmethod
    def get_embeddings(texts):
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else None
        except Exception:
            token = None
        client = InferenceClient(token=token or Config.HUGGINGFACE_API_TOKEN)
        try:
            # feature_extraction returns a numpy array
            try:
                emb_model = current_app.config.get("HF_EMBEDDING_MODEL") if current_app else None
            except Exception:
                emb_model = None
            result = client.feature_extraction(texts, model=emb_model or Config.HF_EMBEDDING_MODEL)
            # Convert to list for compatibility with existing code
            return result.tolist()
        except Exception as e:
            raise Exception(f"HF Embedding Error: {e}")

    @staticmethod
    def generate_answer(question, context):
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else None
        except Exception:
            token = None
        client = InferenceClient(token=token or Config.HUGGINGFACE_API_TOKEN)
        
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant for university students. Answer strictly from the provided context. Reply in a natural mixed style: a brief 1–2 sentence summary, followed by 3–5 short bullet points, and an optional one‑line note if helpful. Keep it concise. When including links, output raw URLs without enclosing symbols or markdown wrappers. If the answer is not in the context, reply exactly: Not available in uploaded documents."
            },
            {
                "role": "user", 
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
        
        try:
            try:
                llm_model = current_app.config.get("HF_LLM_MODEL") if current_app else None
            except Exception:
                llm_model = None
            primary = llm_model or Config.HF_LLM_MODEL
            fallbacks = [
                primary,
                "google/flan-t5-base",
                "mistralai/Mistral-7B-Instruct-v0.2",
            ]
            prompt = (
                "You are a helpful assistant for university students. Answer strictly from the provided context. "
                "Reply in a natural mixed style: a brief 1–2 sentence summary, followed by 3–5 short bullet points, "
                "and an optional one‑line note if helpful. Keep it concise. When including links, output raw URLs "
                "without enclosing symbols or markdown wrappers. If the answer is not in the context, reply exactly: "
                "Not available in uploaded documents.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )
            for mdl in fallbacks:
                if not mdl:
                    continue
                try:
                    response = client.chat_completion(
                        messages,
                        model=mdl,
                        max_tokens=512,
                        temperature=0.1
                    )
                    return response.choices[0].message.content
                except Exception:
                    try:
                        out = client.text_generation(
                            prompt,
                            model=mdl,
                            max_new_tokens=512,
                            temperature=0.1,
                        )
                        return out.strip()
                    except Exception:
                        continue
            return "Not available in uploaded documents."
        except Exception as e:
            return f"Error generating answer: {e}"

    @staticmethod
    def is_smalltalk(text: str) -> bool:
        t = (text or "").strip().lower()
        # Only trigger on explicit greeting phrases, not generic short queries
        greetings = ["hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening", "good afternoon"]
        if any(t == g or t.startswith(g + " ") for g in greetings):
            return True
        # If question mark is present, treat as a query
        if "?" in t:
            return False
        # Very short single-word chats that are greetings
        if len(t.split()) == 1 and t in greetings:
            return True
        return False

    @staticmethod
    def generate_smalltalk(text: str):
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else None
        except Exception:
            token = None
        client = InferenceClient(token=token or Config.HUGGINGFACE_API_TOKEN)
        try:
            try:
                model = current_app.config.get("HF_SMALLTALK_MODEL") if current_app else None
            except Exception:
                model = None
            model = model or Config.HF_SMALLTALK_MODEL
            if 'blenderbot' in (model or '').lower():
                out = client.conversational(text, model=model)
                return (out.get('generated_text') or 'Hello!').strip()
            else:
                prompt = f"Respond politely and briefly: {text}"
                out = client.text_generation(
                    prompt,
                    model=model,
                    max_new_tokens=64,
                    temperature=0.7
                )
                return out.strip()
        except Exception as e:
            return "Hello!"
