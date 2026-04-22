from huggingface_hub import InferenceClient
from groq import Groq
from config import Config
from flask import current_app
import time
import logging
import json

class AIService:
    @staticmethod
    def rewrite_query(question, history):
        """Rewrite the user's question to be self-contained based on conversation history."""
        if not history:
            return question
            
        # Build history string
        history_snippet = history[-6:]
        history_str = ""
        for m in history_snippet:
            role = "Assistant" if m['role'] == 'assistant' else "User"
            history_str += f"{role}: {m['content'][:250]}...\n" if len(m['content']) > 250 else f"{role}: {m['content']}\n"

        # Try Groq first if key exists
        try:
            groq_key = current_app.config.get("GROQ_API_KEY") if current_app else Config.GROQ_API_KEY
            if groq_key:
                groq_client = Groq(api_key=groq_key)
                rewrite_messages = [
                    {"role": "system", "content": "You are a query refiner. Rewrite the user's latest message to be a STANDALONE search query using the provided history. Return ONLY the rewritten text. DO NOT answer the question."},
                    {"role": "user", "content": f"History:\n{history_str}\n\nLatest Message: {question}\n\nStandalone Query:"}
                ]
                completion = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=rewrite_messages,
                    temperature=0,
                    max_tokens=100
                )
                result = completion.choices[0].message.content
                result = (result or "").strip().strip('"').strip("'").strip()
                if result and len(result) > 2:
                    return result
        except Exception as e:
            logging.warning(f"Groq query rewrite failed: {e}")

        # Fallback to Hugging Face
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else Config.HUGGINGFACE_API_TOKEN
            client = InferenceClient(token=token, timeout=12)
            
            # Use chat_completion instead of text_generation for better compatibility with Instruct models
            rewrite_messages = [
                {"role": "system", "content": "You are a query refiner. Rewrite the user's latest message to be a STANDALONE search query using the provided history. Return ONLY the rewritten text. DO NOT answer the question."},
                {"role": "user", "content": f"History:\n{history_str}\n\nLatest Message: {question}\n\nStandalone Query:"}
            ]
            
            response = client.chat_completion(
                messages=rewrite_messages,
                model="mistralai/Mistral-7B-Instruct-v0.2",
                max_tokens=100,
                temperature=0.0
            )
            
            if hasattr(response, 'choices'):
                result = response.choices[0].message.content
            else:
                result = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            result = (result or "").strip().strip('"').strip("'").strip()
            if result and len(result) > 2:
                return result
        except Exception as e:
            logging.warning(f"Hugging Face query rewrite fallback failed: {e}")
                
        return question

    @staticmethod
    def get_embeddings(texts):
        if not texts:
            return []
            
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else None
        except Exception:
            token = None
            
        client = InferenceClient(token=token or Config.HUGGINGFACE_API_TOKEN, timeout=60)  # Increased timeout
        
        try:
            emb_model = current_app.config.get("HF_EMBEDDING_MODEL") if current_app else None
        except Exception:
            emb_model = None
            
        model = emb_model or Config.HF_EMBEDDING_MODEL
        
        # Optimized batching to prevent huge payloads / timeouts
        BATCH_SIZE = 16  # Reduced from 32 to prevent timeouts
        all_embeddings = []
        
        # Process batches with progress tracking
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            
            for attempt in range(4): # Increased to 4 attempts
                try:
                    # result can be numpy array or list
                    result = client.feature_extraction(batch, model=model)
                    
                    # Normalize to list of lists
                    if hasattr(result, 'tolist'):
                        res_list = result.tolist()
                    else:
                        res_list = result
                    
                    # If batch has 1 element, some models return [vector] and some return [float, float...]
                    if len(batch) == 1:
                        # check if it's a list of floats (single vector) or list of lists
                        if res_list and not isinstance(res_list[0], list):
                            res_list = [res_list]
                    
                    all_embeddings.extend(res_list)
                    break 
                        
                except Exception as e:
                    err_msg = str(e).lower()
                    logging.warning(f"Batch embedding attempt {attempt + 1} failed at index {i}: {e}")
                    
                    # If the model is loading, wait longer. Hugging Face specific error pattern.
                    if "loading" in err_msg or "503" in err_msg:
                        wait_time = (attempt + 1) * 5 # 5s, 10s, 15s...
                        time.sleep(wait_time)
                    elif attempt == 3:
                        logging.error(f"Batch embedding failed permanently at index {i} after 4 attempts.")
                        if not all_embeddings:
                            raise e
                    else:
                        time.sleep(2) 
                    
        return all_embeddings

    @staticmethod
    def generate_answer(question, context, history=None, syllabus_context=None, custom_sys_prompt=None, user_preferred_name=None, course_name=None):
        # 1. Base Identity and User Name
        base_identity = "You are a sophisticated AI-powered Intelligence Assistant. Your name is Unibot."
        
        # 2. System Prompt construction
        if custom_sys_prompt:
            sys_prompt = f"{base_identity}\n\n{custom_sys_prompt}"
        else:
            sys_prompt = (
                f"{base_identity}\n\n"
                "Your personality and identity are dynamically defined by the 'Context' provided below.\n\n"
                "CRITICAL RULES:\n"
                "1. IDENTITY AWARENESS: Use the provided 'Software Identity' or 'About this Software' information to inform your persona ONLY if the user is asking about your identity, purpose, or creators. For general or academic questions, act as a neutral and professional assistant.\n"
                f"2. USER PERSONALIZATION: The user you are helping is named '{user_preferred_name or 'the student'}'. If they ask 'who am I' or 'what is my name', you MUST answer with their name. If no name is provided, politely ask them to set their preferred name in the profile settings.\n"
                "3. ADAPTIVE ROLE: If the context is purely academic (Syllabus/Courses), act as a precise 'University Academic Advisor'. If the context contains software manuals, act as the 'Official System Interface'.\n"
                "4. RECOGNIZE INTENT: Match the user's requested depth. If they want a summary, be brief. If they want data (dates, names, fees), be exact and use **bolding**.\n"
                "5. INTELLIGENT GROUNDING: Use the provided context to answer knowledge-based questions. If the context is empty or irrelevant, you SHOULD use your general knowledge to provide a helpful, polite, and professional response as a university assistant. Do NOT simply say 'I don't know' unless it's a very specific factual question that requires document evidence.\n"
                "6. NO HALLUCINATION: If the user asks a specific factual question about a course, syllabus, or university policy that is definitely NOT in the context AND not common knowledge, explicitly state: 'Not available in my current knowledge base for this category'.\n"
                "7. FORMATTING: Use professional Markdown. Use '###' for headers and bullets for lists."
            )

        if syllabus_context:
            course_label = (course_name or "Academic").upper()
            sys_prompt += (
                f"\n\nSYLLABUS GROUNDING ({course_label}):\n"
                f"{syllabus_context}\n"
                "The above block is the OFFICIAL syllabus for this subject. If the user asks for a unit syllabus, extract it from here. "
                "If the user asks a question, ensure the topic is covered by this syllabus. If it is NOT in the syllabus, "
                "mention that the topic is outside the official course scope but provide a brief answer if found in context."
            )

        # Build messages
        messages = [{"role": "system", "content": sys_prompt}]
        if history: messages.extend(history)
        messages.append({
            "role": "user", 
            "content": f"Context:\n{context}\n\nUser Question/Instruction: {question}\n\nAdaptive Answer:"
        })

        # 1. Try Groq (Primary)
        try:
            groq_key = current_app.config.get("GROQ_API_KEY") if current_app else Config.GROQ_API_KEY
            if groq_key:
                groq_client = Groq(api_key=groq_key)
                model = current_app.config.get("GROQ_LLM_MODEL") if current_app else Config.GROQ_LLM_MODEL
                
                # Try Llama 3 70B for high-quality generation if it's not already the default
                try_models = [model, "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
                for m in try_models:
                    if not m: continue
                    try:
                        completion = groq_client.chat.completions.create(
                            model=m,
                            messages=messages,
                            temperature=0.2,
                            max_tokens=2048
                        )
                        out = completion.choices[0].message.content
                        if out and len(out.strip()) > 0:
                            return out.strip()
                    except Exception as ge:
                        logging.warning(f"Groq generation failed with {m}: {ge}")
        except Exception as e:
            logging.error(f"Groq setup failed: {e}")

        # 2. Try Hugging Face (Fallback)
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else Config.HUGGINGFACE_API_TOKEN
            hf_client = InferenceClient(token=token, timeout=45)
            
            hf_fallbacks = [
                "mistralai/Mistral-7B-Instruct-v0.2",
                "HuggingFaceH4/zephyr-7b-beta",
                "microsoft/Phi-3-mini-4k-instruct"
            ]
            
            for mdl in hf_fallbacks:
                try:
                    response = hf_client.chat_completion(
                        messages=messages,
                        model=mdl,
                        max_tokens=1200,
                        temperature=0.2
                    )
                    if hasattr(response, 'choices'):
                        out = response.choices[0].message.content
                    else:
                        out = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                        
                    if out and len(out.strip()) > 0:
                        return out.strip()
                except Exception:
                    continue
        except Exception as e:
            logging.error(f"Hugging Face fallback failed: {e}")

        return "The AI service is currently experiencing high load or is temporarily unavailable. Please try again in a moment."

    @staticmethod
    def generate_answer_from_website(question, context, source_url="", history=None, user_preferred_name=None):
        """Answer only from the given website page content. Do not use external knowledge."""
        try:
            try:
                token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else None
            except Exception:
                token = None
            client = InferenceClient(token=token or Config.HUGGINGFACE_API_TOKEN, timeout=45)
            
            # 1. System Prompt
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Unibot, a university assistant analyzing webpage content. "
                        + (f"The user you are helping is named '{user_preferred_name}'. Always address them by this name if they ask who they are." if user_preferred_name else "The user has not provided a name yet.")
                        + "\n\nUse ONLY the provided webpage text.\n\n"
                        "CORE RULES:\n"
                        "1. ADAPTIVE STYLE: Follow the user's lead. If they request a specific format (e.g., 'give me a summary' or 'list the fees'), prioritize that request.\n"
                        "2. IDENTITY: If the user asks 'who am I', answer with their name using the information provided above.\n"
                        "3. DEFAULT FORMAT: Briefly answer in 2-3 sentences, then provide a '### Details' section with bullet points for specific facts.\n"
                        "4. STRICT GROUNDING: Do not use external knowledge. If the info isn't on the page, say: 'This information is not found on the page.'\n"
                        "5. FORMATTING: Use **bold** for dates, fees, numbers, and names.\n"
                        "6. VERIFICATION: Ensure all extracted information is accurate relative to the provided text."
                    )
                }
            ]

            # 2. Add History
            if history:
                messages.extend(history)
            
            # 3. Add current question
            messages.append({"role": "user", "content": f"Webpage (Source: {source_url}):\n{context}\n\nUser Question/Instruction: {question}\n\nAdaptive Answer:"})

            try:
                llm_model = current_app.config.get("HF_LLM_MODEL") if current_app else None
            except Exception:
                llm_model = None
            primary = llm_model or Config.HF_LLM_MODEL
            fallbacks = []
            if primary:
                fallbacks.append(primary)
            
            robust_models = [
                "mistralai/Mistral-7B-Instruct-v0.2",
                "HuggingFaceH4/zephyr-7b-beta",
                "microsoft/Phi-3-mini-4k-instruct"
            ]
            for m in robust_models:
                if m not in fallbacks:
                    fallbacks.append(m)
            
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
        t = (text or "").strip().lower().strip('.').strip('!').strip('?').strip()
        if not t: return False
        
        # Expanded greetings and common conversational acknowledgments
        smalltalk_phrases = [
            "hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening", "good afternoon",
            "nice", "okay", "ok", "oka", "cool", "great", "excellent", "awesome", "perfect",
            "wow", "i see", "understood", "got it", "fine", "yes", "no", "bye", "goodbye",
            "hii", "hiii", "hiiii", "heyy", "heyyy", "helloo", "hellooo"
        ]
        
        # Exact matches or matches in our extended list
        if t in smalltalk_phrases:
            return True
            
        # Handle simple greetings with punctuation
        cleaned_t = "".join(filter(str.isalnum, t))
        if cleaned_t in ["hi", "hello", "hey", "hii", "hiii", "heyy", "heyyy"]:
            return True

        # Handle repeated characters (e.g., "heyyyyy")
        import re
        words = t.split()
        if len(words) == 1:
            norm_w = re.sub(r'(.)\1+', r'\1', words[0])
            # Common greet roots
            if norm_w in ["hi", "he", "hey", "helo", "hello", "thank"]:
                return True
        
        # If it's a very short message (1-2 words) that matches any of these, it's smalltalk
        if len(words) <= 2:
            if any(t == p or t.startswith(p + " ") or t.endswith(" " + p) for p in smalltalk_phrases):
                # But only if it doesn't look like a real search (e.g., 'fine art' is NOT smalltalk)
                # Short phrases like "who are you" should NOT be smalltalk (they are identity intent)
                if t in ["how are you", "what's up", "whats up", "how r u"]:
                    return True
                if len(words) == 1 or t in ["i see", "got it", "thank you", "good morning", "good evening", "good afternoon"]:
                    return True
        
        return False

    @staticmethod
    def generate_smalltalk(text: str, user_preferred_name=None):
        # Try Groq first
        try:
            groq_key = current_app.config.get("GROQ_API_KEY") if current_app else Config.GROQ_API_KEY
            if groq_key:
                groq_client = Groq(api_key=groq_key)
                completion = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": f"You are Unibot, a friendly university assistant. Respond briefly to the user's greeting. " + (f"IMPORTANT: The user's name is {user_preferred_name}. You MUST start your response by greeting them by their name (e.g. 'Hello {user_preferred_name}!')" if user_preferred_name else "")},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=64,
                    temperature=0.7
                )
                out = completion.choices[0].message.content
                if out and len(out.strip()) > 0:
                    return out.strip()
        except Exception as e:
            logging.warning(f"Groq smalltalk failed: {e}")

        # Fallback to Hugging Face
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else Config.HUGGINGFACE_API_TOKEN
            hf_client = InferenceClient(token=token, timeout=8)
            
            response = hf_client.chat_completion(
                messages=[
                    {"role": "system", "content": f"You are Unibot, a friendly university assistant. Respond briefly to the user's greeting. " + (f"IMPORTANT: The user's name is {user_preferred_name}. You MUST start your response by greeting them by their name (e.g. 'Hello {user_preferred_name}!')" if user_preferred_name else "")},
                    {"role": "user", "content": text}
                ],
                model="mistralai/Mistral-7B-Instruct-v0.2",
                max_tokens=64,
                temperature=0.7
            )
            
            if hasattr(response, 'choices'):
                out = response.choices[0].message.content
            else:
                out = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            if out and len(out.strip()) > 0:
                return out.strip()
        except Exception:
            pass
        
        # Final hardcoded fallback
        name_part = f" {user_preferred_name}" if user_preferred_name else ""
        fallbacks_dict = {
            "nice": f"Glad you think so{name_part}!",
            "okay": f"I'm ready whenever you are{name_part}! Would you like to know anything more about your courses or subjects?",
            "ok": f"Got it{name_part}! How can I help you further?",
            "thanks": f"You're very welcome{name_part}!",
            "thank you": f"You're very welcome{name_part}!",
            "hi": f"Hello{name_part}! How can I help you today?",
            "hello": f"Hi{name_part}! I'm here to help with your studies or any questions about this system."
        }
        t = (text or "").lower().strip().strip('!').strip('.')
        return fallbacks_dict.get(t, f"I'm here to help{name_part}! Do you have any questions about your documents or subjects?")

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
            return " [Image Description: {str(caption)}] "
            
        except Exception as e:
            # Fallback or error logging
            # print(f"Image captioning error: {e}") # specific logging might be noisy
            return " [Image: Caption generation failed] "

    @staticmethod
    def analyze_syllabus_text(text: str) -> str:
        """Extract a structured JSON of Units and Topics from syllabus text using an LLM."""
        if not text:
            return "{}"
            
        # Try Groq
        try:
            groq_key = current_app.config.get("GROQ_API_KEY") if current_app else Config.GROQ_API_KEY
            if groq_key:
                groq_client = Groq(api_key=groq_key)
                completion = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a Syllabus Parser. Your task is to extract the structural hierarchy of a university course syllabus.\n"
                                "Output ONLY a JSON object with the following structure:\n"
                                "{\n"
                                "  \"units\": [\n"
                                "    {\n"
                                "      \"title\": \"Unit 1: [Title]\",\n"
                                "      \"topics\": [\"Topic 1\", \"Topic 2\", ...]\n"
                                "    },\n"
                                "    ...\n"
                                "  ]\n"
                                "}\n"
                            )
                        },
                        {"role": "user", "content": f"Syllabus Text:\n{text[:15000]}"}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Groq syllabus analysis failed: {e}")

        # Fallback to HF
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else Config.HUGGINGFACE_API_TOKEN
            hf_client = InferenceClient(token=token, timeout=60)
            
            response = hf_client.chat_completion(
                messages=[
                    {"role": "system", "content": "Extract syllabus JSON structure from the following text. Output ONLY valid JSON."},
                    {"role": "user", "content": f"Text: {text[:10000]}"}
                ],
                model="mistralai/Mistral-7B-Instruct-v0.2",
                max_tokens=2000
            )
            
            if hasattr(response, 'choices'):
                out = response.choices[0].message.content
            else:
                out = response.get('choices', [{}])[0].get('message', {}).get('content', '{}')
            
            if "```json" in out:
                out = out.split("```json")[1].split("```")[0].strip()
            elif "```" in out:
                out = out.split("```")[1].split("```")[0].strip()
            return out
        except Exception as e:
            logging.error(f"HF syllabus analysis fallback failed: {e}")
            return "{}"
