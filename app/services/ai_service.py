from huggingface_hub import InferenceClient
from config import Config
from flask import current_app
import time
import logging
import json

def approx_tokens(text: str) -> int:
    """Rough estimate of tokens from text (1 word ~= 1.3 tokens)"""
    if not text: return 0
    return int(len(text.split()) * 1.35)

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

        # Try Hugging Face first (Primary)
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else Config.HUGGINGFACE_API_TOKEN
            client = InferenceClient(token=token, timeout=12)
            
            rewrite_messages = [
                {"role": "system", "content": "You are a query refiner. Rewrite the user's latest message to be a STANDALONE search query using the provided history. Return ONLY the rewritten text. DO NOT answer the question."},
                {"role": "user", "content": f"History:\n{history_str}\n\nLatest Message: {question}\n\nStandalone Query:"}
            ]
            
            hf_model = current_app.config.get("HF_LLM_MODEL") if current_app else Config.HF_LLM_MODEL
            response = client.chat_completion(
                messages=rewrite_messages,
                model=hf_model or "mistralai/Mistral-7B-Instruct-v0.2",
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
            logging.warning(f"Hugging Face query rewrite failed: {e}")
                
        return question

    @staticmethod
    def get_embeddings(texts):
        if not texts:
            return []
            
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else None
        except Exception:
            token = None
            
        client = InferenceClient(token=token or Config.HUGGINGFACE_API_TOKEN, timeout=60)
        
        try:
            emb_model = current_app.config.get("HF_EMBEDDING_MODEL") if current_app else None
        except Exception:
            emb_model = None
            
        model = emb_model or Config.HF_EMBEDDING_MODEL
        
        BATCH_SIZE = 16
        all_embeddings = []
        
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            
            success = False
            for attempt in range(5):
                try:
                    result = client.feature_extraction(batch, model=model)
                    
                    if hasattr(result, 'tolist'):
                        res_list = result.tolist()
                    else:
                        res_list = result
                    
                    if len(batch) == 1:
                        if res_list and not isinstance(res_list[0], list):
                            res_list = [res_list]
                    
                    if len(res_list) != len(batch):
                        raise ValueError(f"Expected {len(batch)} embeddings, got {len(res_list)}")
                        
                    all_embeddings.extend(res_list)
                    success = True
                    break 
                        
                except Exception as e:
                    err_msg = str(e).lower()
                    logging.warning(f"Batch embedding attempt {attempt + 1} failed: {e}")
                    
                    if "loading" in err_msg or "503" in err_msg:
                        time.sleep((attempt + 1) * 6)
                    else:
                        time.sleep(3) 
            
            if not success:
                logging.error(f"Failed to generate embeddings for batch starting at index {i}")
                raise RuntimeError(f"Embedding generation failed for a batch of text. Indexing aborted to prevent data corruption.")
                    
        return all_embeddings

    @staticmethod
    def generate_answer(question, context, mode='syllabus', history=None, syllabus_context=None, custom_sys_prompt=None, user_preferred_name=None, course=None, semester=None, subject=None):
        # 1. Base Identity
        base_identity = "You are a sophisticated AI-powered Intelligence Assistant. Your name is Unibot."
        
        # 2. System Prompt construction
        if custom_sys_prompt:
            sys_prompt = f"{base_identity}\n\n{custom_sys_prompt}"
        else:
            role_desc = "University Academic Advisor" if mode == 'syllabus' else "Official Institutional Interface"
            sys_prompt = (
                f"{base_identity}\n\n"
                f"You are currently in **{mode.upper()} MODE**. Your role is: **{role_desc}**.\n\n"
                "Your personality and identity are primarily defined by your helpful and academic nature.\n\n"
                "CRITICAL RULES:\n"
                "1. IDENTITY AWARENESS: You are Unibot. If asked about your creators or identity, be professional. You were created to assist students.\n"
                f"2. USER PERSONALIZATION: The user's name is '{user_preferred_name or 'the student'}'. " +
                (f"They are studying {course}, in Semester {semester}" + (f" (Subject: {subject})." if subject else ".") if course and semester else "") +
                " Use this to be friendly, but don't overdo it.\n"
                "3. ADAPTIVE ROLE: In STUDIES mode, prioritize accuracy based on syllabus and documents. In GENERAL mode, be a helpful explorer of university life.\n"
                "4. NATURAL SPEECH: Answer directly. NEVER mention 'provided context' or 'the text'. Avoid phrases like 'Based on the information provided...'.\n"
                "5. INTELLIGENT GROUNDING: Use the provided context to answer. If the context doesn't contain the answer, use your general knowledge to provide a helpful response, but clearly distinguish it if it's not from official sources.\n"
                "6. HELPFULNESS: Never be dismissive. If you don't know something, suggest where the user might find it or offer related helpful information.\n"
                "7. FORMATTING: Use professional Markdown. Use bold for key terms and bullet points for lists."
            )

        if syllabus_context:
            course_label = (course or "Academic").upper()
            sys_prompt += (
                f"\n\nSYLLABUS GROUNDING (Subject: {subject or course_label}):\n"
                f"{syllabus_context}\n"
                "The above block is the structured JSON syllabus for this subject. Use it to verify if a topic is officially part of the curriculum. "
                "If the user asks about a specific UNIT or MODULE, refer to this structure. "
                "If a user's question relates to a topic NOT in this structure, provide a helpful answer from the provided context but clarify it may be outside the core syllabus."
            )

        # Build messages
        messages = [{"role": "system", "content": sys_prompt}]
        if history: messages.extend(history)
        messages.append({
            "role": "user", 
            "content": f"Context:\n{context}\n\nUser Question/Instruction: {question}\n\nAdaptive Answer:"
        })

        # 1. Try Hugging Face (Primary)
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else Config.HUGGINGFACE_API_TOKEN
            hf_client = InferenceClient(token=token, timeout=45)
            
            hf_model = current_app.config.get("HF_LLM_MODEL") if current_app else Config.HF_LLM_MODEL
            hf_fallbacks = [
                hf_model,
                "meta-llama/Llama-3.2-3B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3"
            ]

            for mdl in hf_fallbacks:
                if not mdl: continue
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
            logging.error(f"Hugging Face generation failed: {e}")

        return "The AI service is currently experiencing high load or is temporarily unavailable. Please try again in a moment."

    @staticmethod
    def generate_answer_from_website(question, context, source_url="", history=None, user_preferred_name=None, course=None, semester=None, subject=None):
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
                        + (f"The user you are helping is named '{user_preferred_name}'. " if user_preferred_name else "The user has not provided a name yet. ")
                        + (f"They are studying {course} (Semester {semester})" + (f", specifically {subject}." if subject else ".") if course and semester else "")
                        + "\n\nUse ONLY the provided webpage text.\n\n"
                        "CORE RULES:\n"
                        "1. NATURAL RESPONSES: Speak naturally and directly. NEVER mention 'the provided webpage', 'the text', or 'based on the content'. Answer as if you simply know the facts.\n"
                        "2. ADAPTIVE STYLE: Follow the user's lead. If they request a specific format (e.g., 'give me a summary' or 'list the fees'), prioritize that request.\n"
                        "3. IDENTITY: If the user asks 'who am I', answer with their name using the information provided above.\n"
                        "4. DEFAULT FORMAT: Briefly answer in 2-3 sentences, then provide a '### Details' section with bullet points for specific facts.\n"
                        "5. STRICT GROUNDING: Do not use external knowledge. If the info isn't on the page, say: 'This information is not found on the page.'\n"
                        "6. FORMATTING: Use **bold** for dates, fees, numbers, and names.\n"
                        "7. VERIFICATION: Ensure all extracted information is accurate relative to the provided text."
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
                "meta-llama/Llama-3.2-3B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3"
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
                        
            logging.error("All Hugging Face fallback models failed for website content.")
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
        if cleaned_t in ["hi", "hello", "hey", "hii", "hiii", "heyy", "heyyy", "yo", "sup", "greetings"]:
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
    def generate_smalltalk(text: str, user_preferred_name=None, course=None, semester=None, subject=None):
        # Try Hugging Face first (Primary)
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else Config.HUGGINGFACE_API_TOKEN
            hf_client = InferenceClient(token=token, timeout=8)
            
            hf_model = current_app.config.get("HF_SMALLTALK_MODEL") if current_app else Config.HF_SMALLTALK_MODEL
            response = hf_client.chat_completion(
                messages=[
                    {"role": "system", "content": f"You are Unibot, a friendly university assistant. Respond briefly to the user's greeting. " + 
                     (f"The user is {user_preferred_name}, studying {course} (Semester {semester})" + (f", specifically {subject}." if subject else ".") if user_preferred_name and course and semester else "") +
                     (f" IMPORTANT: You MUST start your response by greeting the user by their name '{user_preferred_name}' (e.g. 'Hello {user_preferred_name}!')" if user_preferred_name else "")},
                    {"role": "user", "content": text}
                ],
                model=hf_model or "google/gemma-2-2b-it",
                max_tokens=64,
                temperature=0.7
            )
            
            if hasattr(response, 'choices'):
                out = response.choices[0].message.content
            else:
                out = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            if out and len(out.strip()) > 0:
                return out.strip()
        except Exception as e:
            logging.warning(f"Hugging Face smalltalk failed: {e}")
        
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
    def analyze_syllabus_text(text):
        """Extract a structured JSON of Units and Topics from syllabus text using an LLM."""
        if not text:
            return "{}"
            
        try:
            # Increase limit to 60,000 characters
            processed_text = text[:60000]
            
            system_prompt = """You are a curriculum analysis expert. 
            Extract the units and their topics from this syllabus text. 
            
            Return ONLY a valid JSON object with the following structure:
            {"units": [{"title": "Unit X: Name", "topics": ["Topic 1", "Topic 2"]}]}
            
            Rules:
            1. ONLY return the JSON. No conversational filler.
            2. If no units are found, return {"units": []}.
            3. Ensure all units are captured.
            """
            
            model = Config.AI_MODEL # Should be Gemini or high-quality LLM
            token = Config.HUGGINGFACE_API_TOKEN
            
            client = InferenceClient(token=token, timeout=90)
            
            response = client.text_generation(
                f"{system_prompt}\n\nSyllabus Text:\n{processed_text}",
                max_new_tokens=4000,
                model=model
            )
            
            out = response.strip()
            
            # More robust JSON extraction using regex
            import re
            json_match = re.search(r'(\{.*\}|\[.*\])', out, re.DOTALL)
            if json_match:
                out = json_match.group(1)
            else:
                # Fallback to backtick cleaning if regex fails
                if "```json" in out:
                    out = out.split("```json")[1].split("```")[0].strip()
                elif "```" in out:
                    out = out.split("```")[1].split("```")[0].strip()
            
            # Validate JSON before returning
            try:
                import json
                json.loads(out)
                return out
            except Exception as e:
                logging.error(f"Syllabus analysis returned invalid JSON: {e}\nOutput snippet: {out[:200]}...")
                return '{"units": []}'
                
        except Exception as e:
            logging.error(f"Syllabus analysis failed: {e}")
            return '{"units": []}'
