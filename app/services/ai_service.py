from huggingface_hub import InferenceClient
from config import Config
from flask import current_app
import time
import logging

class AIService:
    @staticmethod
    def get_embeddings(texts):
        if not texts:
            return []
            
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else None
        except Exception:
            token = None
            
        client = InferenceClient(token=token or Config.HUGGINGFACE_API_TOKEN, timeout=30)  # Reduced timeout
        
        try:
            emb_model = current_app.config.get("HF_EMBEDDING_MODEL") if current_app else None
        except Exception:
            emb_model = None
            
        model = emb_model or Config.HF_EMBEDDING_MODEL
        
        # Optimized batching to prevent huge payloads / timeouts
        BATCH_SIZE = 32  # Increased from 16 for better throughput
        all_embeddings = []
        
        # Process batches with progress tracking
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            
            try:
                # result can be numpy array or list
                result = client.feature_extraction(batch, model=model)
                
                # Normalize to list of lists
                if hasattr(result, 'tolist'):
                    res_list = result.tolist()
                else:
                    res_list = result
                
                # If batch has 1 element, some models return [vector] and some return [float, float...]
                # We need to ensure we return a list of vectors [[float...]]
                if len(batch) == 1:
                    # check if it's a list of floats (single vector) or list of lists
                    if res_list and not isinstance(res_list[0], list):
                        res_list = [res_list]
                
                all_embeddings.extend(res_list)
                
                # Progress logging for large batches
                if len(texts) > 100 and batch_num % 5 == 0:
                    print(f"Embedding progress: {batch_num}/{total_batches} batches completed")
                    
            except Exception as e:
                logging.error(f"Batch embedding failed at index {i}: {e}")
                # If a batch fails, we could either stop or continue with partial results.
                # For consistency, let's raise if the first batch fails, or if it's critical.
                if not all_embeddings:
                    raise e
                # For subsequent batches, continue with what we have
                continue
                    
        return all_embeddings

    @staticmethod
    def generate_answer(question, context):
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else None
        except Exception:
            token = None
        client = InferenceClient(token=token or Config.HUGGINGFACE_API_TOKEN, timeout=45)
        
        # Adaptive prompt that recognizes user instructions and attributes
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a sophisticated AI-powered Intelligence Assistant. Your personality and identity are dynamically defined by the 'Context' provided below.\n\n"
                    "CRITICAL RULES:\n"
                    "1. IDENTITY AWARENESS: Before answering, check the context for 'Software Identity' or 'About this Software' information. If the context contains details about your name, purpose, version, or creators, YOU MUST speak as that specific entity. Use that identity to answer questions like 'Who are you?' or 'What can you do?'.\n"
                    "2. ADAPTIVE ROLE: If the context is purely academic (Syllabus/Courses), act as a precise 'University Academic Advisor'. If the context contains software manuals, act as the 'Official System Interface'.\n"
                    "3. RECOGNIZE INTENT: Match the user's requested depth. If they want a summary, be brief. If they want data (dates, names, fees), be exact and use **bolding**.\n"
                    "4. STRICT GROUNDING: Use ONLY the provided context. Do not invent details. If the context has no identity info and no academic info for the query, say: 'Not available in my current knowledge base'.\n"
                    "5. FORMATTING: Use professional Markdown. Use '###' for headers and bullets for lists. Ensure tone is premium, professional, and helpful."
                )
            },
            {
                "role": "user", 
                "content": f"Context:\n{context}\n\nUser Question/Instruction: {question}\n\nAdaptive Answer:"
            }
        ]
        
        try:
            try:
                llm_model = current_app.config.get("HF_LLM_MODEL") if current_app else None
            except Exception:
                llm_model = None
            primary = llm_model or Config.HF_LLM_MODEL
            fallbacks = []
            if primary:
                fallbacks.append(primary)
            if "HuggingFaceH4/zephyr-7b-beta" not in fallbacks:
                fallbacks.append("HuggingFaceH4/zephyr-7b-beta")
            fallbacks.append("mistralai/Mistral-7B-Instruct-v0.2")
            fallbacks.append("google/flan-t5-small")
            
            for mdl in fallbacks:
                if not mdl:
                    continue
                try:
                    # Try chat completion API (preferred for chat models)
                    response = client.chat_completion(
                        messages=messages,
                        model=mdl,
                        max_tokens=1200,  # Significantly increased to prevent truncation
                        temperature=0.2
                    )
                    # Handle response object or dict
                    if hasattr(response, 'choices'):
                        out = response.choices[0].message.content
                    else:
                        out = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                        
                    if out and len(out.strip()) > 0:
                        return out.strip()
                except Exception as e:
                    logging.warning(f"Chat completion failed with {mdl}: {e}")
                    # Fallback to legacy text generation
                    try:
                        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                        out = client.text_generation(
                            prompt, 
                            model=mdl, 
                            max_new_tokens=1200,
                            temperature=0.2
                        )
                        if out and len(out.strip()) > 0:
                            return out.strip()
                    except Exception as e2:
                        logging.warning(f"Legacy generation failed with {mdl}: {e2}")
                        continue
                        
            logging.error("All fallback models failed to generate an answer.")
            return "Not available in uploaded documents."
        except Exception as e:
            return f"Error generating answer: {e}"

    @staticmethod
    def generate_answer_from_website(question, context, source_url=""):
        """Answer only from the given website page content. Do not use external knowledge."""
        try:
            try:
                token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else None
            except Exception:
                token = None
            client = InferenceClient(token=token or Config.HUGGINGFACE_API_TOKEN, timeout=45)
            
            # Adaptive prompt for website content
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are analyzing webpage content as a university assistant. Use ONLY the provided webpage text.\n\n"
                        "CORE RULES:\n"
                        "1. ADAPTIVE STYLE: Follow the user's lead. If they request a specific format (e.g., 'give me a summary' or 'list the fees'), prioritize that request.\n"
                        "2. DEFAULT FORMAT: Briefly answer in 2-3 sentences, then provide a '### Details' section with bullet points for specific facts.\n"
                        "3. STRICT GROUNDING: Do not use external knowledge. If the info isn't on the page, say: 'This information is not found on the page.'\n"
                        "4. FORMATTING: Use **bold** for dates, fees, numbers, and names.\n"
                        "5. VERIFICATION: Ensure all extracted information is accurate relative to the provided text."
                    )
                },
                {"role": "user", "content": f"Webpage (Source: {source_url}):\n{context}\n\nUser Question/Instruction: {question}\n\nAdaptive Answer:"}
            ]

            try:
                llm_model = current_app.config.get("HF_LLM_MODEL") if current_app else None
            except Exception:
                llm_model = None
            primary = llm_model or Config.HF_LLM_MODEL
            fallbacks = []
            if primary:
                fallbacks.append(primary)
            if "HuggingFaceH4/zephyr-7b-beta" not in fallbacks:
                fallbacks.append("HuggingFaceH4/zephyr-7b-beta")
            fallbacks.append("mistralai/Mistral-7B-Instruct-v0.2")
            fallbacks.append("google/flan-t5-small")
            
            for mdl in fallbacks:
                if not mdl:
                    continue
                try:
                    # Try chat completion API
                    response = client.chat_completion(
                        messages=messages,
                        model=mdl,
                        max_tokens=1300,  # Larger for web content
                        temperature=0.2
                    )
                    
                    if hasattr(response, 'choices'):
                        out = response.choices[0].message.content
                    else:
                        out = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                        
                    if out and len(out.strip()) > 0:
                        return out.strip()
                except Exception as e:
                    logging.warning(f"Website chat completion failed with {mdl}: {e}")
                    # Fallback to legacy text generation
                    try:
                        prompt_legacy = (
                            "Instruction: Analyze the following webpage content and answer the question.\n"
                            f"Webpage Content:\n{context}\n\n"
                            f"Question: {question}\n\n"
                            "Answer:"
                        )
                        out = client.text_generation(
                            prompt_legacy,
                            model=mdl,
                            max_new_tokens=1200,
                            temperature=0.2,
                        )
                        if out and len(out.strip()) > 0:
                            return out.strip()
                    except Exception as e2:
                        continue
                        
            logging.error("All fallback models failed for website content.")
            return "This information is not found on the page."
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
        client = InferenceClient(token=token or Config.HUGGINGFACE_API_TOKEN, timeout=5)
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

    @staticmethod
    def generate_image_caption(image_bytes: bytes):
        """Generate a caption for an image using a VLM via Hugging Face API"""
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else None
        except Exception:
            token = None
            
        # Ensure we have a token
        token = token or Config.HUGGINGFACE_API_TOKEN
        if not token:
            return " [Image: No caption available - API token missing] "
            
        client = InferenceClient(token=token, timeout=10)
        
        try:
            try:
                model = current_app.config.get("HF_IMAGE_CAPTION_MODEL") if current_app else None
            except Exception:
                model = None
            model = model or Config.HF_IMAGE_CAPTION_MODEL
            
            # The client.image_to_text method is the standard for captioning
            # It accepts bytes directly or PIL images
            # Using raw bytes is safer for general transmission
            caption = client.image_to_text(image_bytes, model=model)
            
            # Returns a string directly or an object with 'generated_text'
            if isinstance(caption, dict) and 'generated_text' in caption:
                return f" [Image Description: {caption['generated_text']}] "
            elif isinstance(caption, list) and len(caption) > 0 and 'generated_text' in caption[0]:
                 return f" [Image Description: {caption[0]['generated_text']}] "
            return f" [Image Description: {str(caption)}] "
            
        except Exception as e:
            # Fallback or error logging
            # print(f"Image captioning error: {e}") # specific logging might be noisy
            return " [Image: Caption generation failed] "
