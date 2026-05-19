from huggingface_hub import InferenceClient
from config import Config
from flask import current_app
import time
import logging
import json
__all__ = ['AIService', 'approx_tokens']

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

        # Try Hugging Face first (Primary with robust fallbacks)
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else Config.HUGGINGFACE_API_TOKEN
            client = InferenceClient(token=token, timeout=12)
            
            rewrite_messages = [
                {"role": "system", "content": "You are a query refiner. Rewrite the user's latest message to be a STANDALONE search query using the provided history. Return ONLY the rewritten text. DO NOT answer the question."},
                {"role": "user", "content": f"History:\n{history_str}\n\nLatest Message: {question}\n\nStandalone Query:"}
            ]
            
            hf_model = current_app.config.get("HF_LLM_MODEL") if current_app else Config.HF_LLM_MODEL
            fallbacks = [
                hf_model,
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-72B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct"
            ]
            
            for mdl in fallbacks:
                if not mdl: continue
                try:
                    response = client.chat_completion(
                        messages=rewrite_messages,
                        model=mdl,
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
                except Exception as inner_ex:
                    logging.warning(f"Hugging Face query rewrite fallback {mdl} failed: {inner_ex}")
                    continue
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
                "3. ADAPTIVE ROLE: In STUDIES mode, rely entirely on syllabus and academic documents. In GENERAL mode, rely entirely on university general documents. Do not answer outside of this scope.\n"
                "4. NATURAL SPEECH: Answer directly. NEVER mention 'provided context' or 'the text'. Avoid phrases like 'Based on the information provided...'.\n"
                "5. STRICT GROUNDING: You are a strict RAG chatbot. You MUST answer strictly using ONLY the provided context. If the provided context is empty or does not contain the answer, you MUST politely state that you do not have the information in your knowledge base. NEVER use your general pre-trained knowledge to answer questions.\n"
                "6. SYLLABUS PRIORITY: For questions about curriculum structure, Units, Modules, or specific topics, you MUST prioritize the **SYLLABUS GROUNDING** section. Provide the topics exactly as listed in the official curriculum.\n"
                "7. GROUNDING SAFEGUARD: If you are in STUDIES (SYLLABUS) mode and the SYLLABUS GROUNDING section is missing or empty, and the user asks for topics/curriculum, you MUST politely explain that you don't have their specific subject's syllabus yet. Ask them to ensure their **Course, Semester, and Subject** are correctly set in their profile or the sidebar.\n"
                "8. HELPFULNESS: Never be dismissive. If you don't know something, suggest where the user might find it or offer related helpful information.\n"
                "9. FORMATTING: Use professional Markdown. Use bold for key terms and bullet points for lists."
            )

        if syllabus_context:
            course_label = (course or "Academic").upper()
            sys_prompt += (
                f"\n\n### SYLLABUS GROUNDING (Subject: {subject or course_label})\n"
                f"{syllabus_context}\n"
                "The above JSON block is the **OFFICIAL STRUCTURE** for this subject. \n"
                " - If the user asks 'what are the topics', 'give me the syllabus', or 'what is in Module/Unit X', you MUST use the titles and topics from this JSON.\n"
                " - Maintain the exact terminology used in the JSON.\n"
                " - If the JSON topics are detailed, include that detail in your answer."
            )

        # Build messages
        messages = [{"role": "system", "content": sys_prompt}]
        if history: messages.extend(history)
        context_str = context if context.strip() else "[NO CONTEXT FOUND IN KNOWLEDGE BASE. STRICT RULE: YOU MUST DECLINE TO ANSWER THIS QUESTION AS NO DATA WAS RETRIEVED.]"
        messages.append({
            "role": "user", 
            "content": f"Context Information:\n<context>\n{context_str}\n</context>\n\nBased STRICTLY on the context above, answer the following question. If the context does not contain the answer, you MUST output exactly 'I do not have the information.'\n\nQuestion: {question}\n\nAnswer:"
        })

        # 1. Try Hugging Face (Primary)
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else Config.HUGGINGFACE_API_TOKEN
            hf_client = InferenceClient(token=token, timeout=45)
            
            hf_model = current_app.config.get("HF_LLM_MODEL") if current_app else Config.HF_LLM_MODEL
            hf_fallbacks = [
                hf_model,
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-72B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct"
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
            context_str = context if context.strip() else "[NO WEBPAGE CONTENT FOUND. YOU MUST STATE THE INFORMATION IS NOT ON THE PAGE.]"
            messages.append({"role": "user", "content": f"Webpage (Source: {source_url}):\n{context_str}\n\nUser Question/Instruction: {question}\n\nAnswer:"})

            try:
                llm_model = current_app.config.get("HF_LLM_MODEL") if current_app else None
            except Exception:
                llm_model = None
            primary = llm_model or Config.HF_LLM_MODEL
            fallbacks = []
            if primary:
                fallbacks.append(primary)
            
            robust_models = [
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-72B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct"
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
        # Try Hugging Face first (Primary with robust fallbacks)
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else Config.HUGGINGFACE_API_TOKEN
            hf_client = InferenceClient(token=token, timeout=8)
            
            hf_model = current_app.config.get("HF_SMALLTALK_MODEL") if current_app else Config.HF_SMALLTALK_MODEL
            fallbacks = [
                hf_model,
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "Qwen/Qwen2.5-7B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct"
            ]
            
            for mdl in fallbacks:
                if not mdl: continue
                try:
                    response = hf_client.chat_completion(
                        messages=[
                            {"role": "system", "content": f"You are Unibot, a friendly university assistant. Respond briefly to the user's greeting. " + 
                             (f"The user is {user_preferred_name}, studying {course} (Semester {semester})" + (f", specifically {subject}." if subject else ".") if user_preferred_name and course and semester else "") +
                             (f" IMPORTANT: You MUST start your response by greeting the user by their name '{user_preferred_name}' (e.g. 'Hello {user_preferred_name}!')" if user_preferred_name else "")},
                            {"role": "user", "content": text}
                        ],
                        model=mdl,
                        max_tokens=64,
                        temperature=0.7
                    )
                    
                    if hasattr(response, 'choices'):
                        out = response.choices[0].message.content
                    else:
                        out = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                    
                    if out and len(out.strip()) > 0:
                        return out.strip()
                except Exception as inner_ex:
                    logging.warning(f"Hugging Face smalltalk fallback {mdl} failed: {inner_ex}")
                    continue
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
            return f" [Image Description: {str(caption)}] "
            
        except Exception as e:
            # Fallback or error logging
            # print(f"Image captioning error: {e}") # specific logging might be noisy
            return " [Image: Caption generation failed] "

    @staticmethod
    def analyze_syllabus_text(text):
        """Extract a structured JSON of Units and Topics from syllabus text using an LLM."""
        if not text or len(text.strip()) < 50:
            logging.warning("Syllabus text too short for analysis.")
            return '{"units": []}'
            
        try:
            # Use up to 45,000 characters to stay within context limits of most free-tier models
            processed_text = text[:45000]
            
            system_prompt = (
                "You are an expert Academic Content Architect. Your mission is to decompose the provided university syllabus into a precise, hierarchical Knowledge Map.\n\n"
                "INSTRUCTIONS:\n"
                "1. SCAN: Thoroughly scan the text for 'Unit', 'Module', 'Chapter', or 'Section' headings.\n"
                "2. EXTRACT: For each major division, capture its full title and ALL specific topics, sub-topics, or keywords mentioned under it.\n"
                "3. PRECISION: Maintain the **EXACT wording and terminology** of the topics as found in the syllabus. Do not summarize, paraphrase, or use generic textbook names for topics.\n"
                "4. OBJECTIVES: Include 'Learning Objectives' or 'Expected Outcomes' as topics if they are listed specifically for a unit.\n"
                "5. LOGIC: If no clear unit headings exist, logically group the curriculum content into coherent modules based on subject matter.\n"
                "6. OUTPUT: You MUST return ONLY a valid JSON object. No prose, no code blocks, no preamble.\n\n"
                "JSON FORMAT:\n"
                "{\n"
                '  "units": [\n'
                '    {\n'
                '      "title": "Unit [Number]: [Topic Name]",\n'
                '      "topics": ["Sub-topic 1", "Sub-topic 2", "Key Concept X"]\n'
                '    }\n'
                '  ]\n'
                "}\n\n"
                "If the text is not a syllabus or contains no curriculum data, return {\"units\": []}."
            )
            
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else Config.HUGGINGFACE_API_TOKEN
            client = InferenceClient(token=token, timeout=90)
            
            primary_model = current_app.config.get("HF_SYLLABUS_MODEL") if current_app else Config.HF_SYLLABUS_MODEL
            fallbacks = [
                primary_model,
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-72B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct"
            ]

            out = ""
            for mdl in fallbacks:
                if not mdl: continue
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Syllabus Text:\n{processed_text}\n\nStrict JSON Knowledge Map:"}
                    ]
                    response = client.chat_completion(
                        messages=messages,
                        model=mdl,
                        max_tokens=2500,
                        temperature=0.1
                    )
                    
                    if hasattr(response, 'choices'):
                        out = response.choices[0].message.content
                    else:
                        out = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                    
                    if out and len(out.strip()) > 5:
                        break # Success
                except Exception as e:
                    logging.warning(f"Syllabus extraction attempt with {mdl} failed: {e}")
                    continue

            if not out:
                return '{"units": []}'

            out = out.strip()
            
            # Extract JSON from potential markdown blocks or noise
            import re
            # Try to find the first { and last }
            json_match = re.search(r'(\{.*\})', out, re.DOTALL)
            if json_match:
                out = json_match.group(1)
            else:
                # Fallback cleaning
                out = re.sub(r'```json\s*|\s*```', '', out).strip()

            # Final validation
            try:
                json_data = json.loads(out)
                if 'units' not in json_data:
                    json_data = {"units": []}
                return json.dumps(json_data)
            except Exception as e:
                logging.error(f"Syllabus analysis returned invalid JSON content: {e}")
                return '{"units": []}'
                
        except Exception as e:
            logging.error(f"Syllabus analysis critical failure: {e}")
            return '{"units": []}'
