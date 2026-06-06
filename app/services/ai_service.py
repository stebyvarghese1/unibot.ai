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
    def _chat_completion_with_fallback(messages, model, token, max_tokens=1200, temperature=0.2, timeout=45):
        """
        Run chat completion with a model.
        First tries the metered Inference Providers router.
        If it fails due to credit exhaustion (402 Payment Required),
        it falls back to the free Serverless Inference Hub.
        """
        # 1. Try metered Inference Providers router
        try:
            client = InferenceClient(token=token, timeout=timeout)
            response = client.chat_completion(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            if hasattr(response, 'choices'):
                out = response.choices[0].message.content
            else:
                out = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            if out and len(out.strip()) > 0:
                return out.strip()
        except Exception as e:
            err_str = str(e).lower()
            if "402" not in err_str and "payment required" not in err_str and "credits" not in err_str:
                raise e
            logging.warning(f"Metered router failed with 402 for {model}. Falling back to free serverless endpoint...")
            
        # 2. Fallback to free Serverless Inference Hub
        base_url = f"https://api-inference.huggingface.co/models/{model}"
        client_free = InferenceClient(token=token, base_url=base_url, timeout=timeout)
        
        for attempt in range(5):
            try:
                response = client_free.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                if hasattr(response, 'choices'):
                    out = response.choices[0].message.content
                else:
                    out = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                if out and len(out.strip()) > 0:
                    return out.strip()
            except Exception as free_ex:
                free_err = str(free_ex).lower()
                if ("loading" in free_err or "503" in free_err or "currently loading" in free_err) and attempt < 4:
                    time.sleep(6 * (attempt + 1))
                    continue
                raise free_ex
                
        raise RuntimeError(f"Model {model} failed on both metered router and free serverless endpoint.")

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
            
            import datetime
            current_time_str = datetime.datetime.now().strftime("%B %d, %Y (YYYY-MM-DD: %Y-%m-%d)")
            
            rewrite_messages = [
                {"role": "system", "content": f"You are a query refiner. The current date is {current_time_str}. Rewrite the user's latest message to be a STANDALONE search query using the provided history. Return ONLY the rewritten text. DO NOT answer the question."},
                {"role": "user", "content": f"History:\n{history_str}\n\nLatest Message: {question}\n\nStandalone Query:"}
            ]
            
            hf_model = current_app.config.get("HF_LLM_MODEL") if current_app else Config.HF_LLM_MODEL
            fallbacks = [
                hf_model,
                "Qwen/Qwen2.5-7B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3",
                "Qwen/Qwen3-8B"
            ]
            
            for mdl in fallbacks:
                if not mdl: continue
                try:
                    result = AIService._chat_completion_with_fallback(
                        messages=rewrite_messages,
                        model=mdl,
                        token=token,
                        max_tokens=100,
                        temperature=0.0,
                        timeout=12
                    )
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
    def normalize_syllabus_question(question, syllabus_json_str):
        if not syllabus_json_str or not question:
            return question
            
        try:
            import json
            import re
            
            data = json.loads(syllabus_json_str)
            units = data.get("units", [])
            if not units:
                return question
                
            arabic_to_roman = {
                1: "I", 2: "II", 3: "III", 4: "IV", 5: "V",
                6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"
            }
            
            words_to_num = {
                "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
            }
            
            q_lower = question.lower().strip()
            
            for word, num in words_to_num.items():
                q_lower = re.sub(rf"\b{word}\b", str(num), q_lower)
                
            for idx, unit in enumerate(units):
                title = unit.get("title", "")
                if not title:
                    continue
                    
                unit_num = idx + 1
                roman_num = arabic_to_roman.get(unit_num, "")
                
                patterns = [
                    rf"\b(unit|module|chapter|section)\s+{unit_num}\b",
                    rf"\b{unit_num}\s+(unit|module|chapter|section)\b",
                    rf"\b(unit|module|chapter|section)\s+{roman_num.lower()}\b",
                    rf"\b{roman_num.lower()}\s+(unit|module|chapter|section)\b"
                ]
                
                for pattern in patterns:
                    if re.search(pattern, q_lower):
                        return f"Provide the topics for '{title}' as listed in the syllabus grounding."
                        
        except Exception:
            pass
            
        return question

    @staticmethod
    def clean_response(text: str) -> str:
        if not text:
            return ""
            
        import re
        
        # 1. Strip leading labels/prefixes (e.g. "Answer:", "Response:", "Assistant:", etc.)
        prefix_pattern = r'^\s*(?:answer|response|assistant|ai|unibot|output|unibot\s*answer)\s*:\s*'
        text = re.sub(prefix_pattern, '', text, flags=re.IGNORECASE)
        
        # 2. Strip common introductory RAG phrases
        prohibited_phrases = [
            r'^\s*based\s+on\s+the\s+(?:provided\s+)?(?:context|text|information|syllabus\s+grounding|document|webpage|page)\s*,?\s*',
            r'^\s*according\s+to\s+the\s+(?:provided\s+)?(?:context|text|information|syllabus\s+grounding|document|webpage|page)\s*,?\s*',
            r'^\s*in\s+the\s+provided\s+(?:context|text|document|webpage)\s*,?\s*',
            r'^\s*based\s+strictly\s+on\s+the\s+(?:provided\s+)?(?:context|text|information|syllabus\s+grounding)\s*,?\s*',
            r'^\s*from\s+the\s+provided\s+(?:context|text|document|webpage)\s*,?\s*',
        ]
        
        # Apply the regex replacements iteratively until no more matches at the start of the text
        changed = True
        while changed:
            original = text
            for pattern in prohibited_phrases:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            text = text.strip()
            if text == original:
                changed = False
                
        # 3. Capitalize first letter if it got lowercased by the replacement
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
            
        return text

    @staticmethod
    def generate_answer(question, context, mode='syllabus', history=None, syllabus_context=None, custom_sys_prompt=None, user_preferred_name=None, course=None, semester=None, subject=None):
        # Clean/normalize question terms deterministically for syllabus queries
        if mode == 'syllabus' and syllabus_context:
            question = AIService.normalize_syllabus_question(question, syllabus_context)
            
            try:
                import json
                import re
                syllabus_data = json.loads(syllabus_context)
                units = syllabus_data.get("units", [])
                
                # Check for full syllabus request
                q_clean = question.lower().strip().strip('?').strip('.')
                if q_clean in [
                    "give me the syllabus", "what is the syllabus", "show syllabus", "syllabus", 
                    "view syllabus", "what are the units", "get syllabus", "provide syllabus", 
                    "explain syllabus", "tell me the syllabus", "show the units"
                ]:
                    out_parts = []
                    out_parts.append("Here is the syllabus structure for this course:\n")
                    for unit in units:
                        title = unit.get("title", "Unknown Unit")
                        topics = unit.get("topics", [])
                        out_parts.append(f"### {title}")
                        if topics:
                            topics_list_str = "\n".join([f"* {t}" for t in topics])
                            out_parts.append(topics_list_str)
                        else:
                            out_parts.append("*No topics listed.*")
                        out_parts.append("")
                    return "\n".join(out_parts)
                
                # Check for unit-specific request (e.g. from normalize_syllabus_question)
                match = re.search(r"^Provide the topics for '(.*)' as listed in the syllabus grounding\.$", question)
                if match:
                    target_title = match.group(1).strip()
                    for unit in units:
                        if unit.get("title", "").strip().lower() == target_title.lower():
                            topics = unit.get("topics", [])
                            if topics:
                                topics_list_str = "\n".join([f"* {t}" for t in topics])
                                return f"Here are the topics listed under **{unit.get('title')}**:\n\n{topics_list_str}"
                            else:
                                return f"There are no specific topics listed under **{unit.get('title')}**."
            except Exception as e:
                logging.warning(f"Deterministic local syllabus parsing failed: {e}")

        # 1. Base Identity
        import datetime
        current_time_str = datetime.datetime.now().strftime("%B %d, %Y (YYYY-MM-DD: %Y-%m-%d)")
        base_identity = f"You are a sophisticated AI-powered Intelligence Assistant. Your name is Unibot. The current date is {current_time_str}."
        
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
                "1. IDENTITY AWARENESS: You are Unibot. You were developed and are maintained by Steby Varghese (King's Guard). If asked about your creator, developer, or who made you, always state clearly that you were created and are maintained by Steby Varghese (King's Guard).\n"
                f"2. USER PERSONALIZATION: The user's name is '{user_preferred_name or 'the student'}'. " +
                (f"They are studying {course}, in Semester {semester}" + (f" (Subject: {subject})." if subject else ".") if (mode == 'syllabus' and course and semester) else "") +
                " Use this to be friendly, but don't overdo it.\n"
                "3. ADAPTIVE ROLE: In STUDIES mode, rely entirely on syllabus and academic documents. In GENERAL mode, rely entirely on university general documents. Do not answer outside of this scope.\n"
                "4. NATURAL SPEECH: Answer directly. NEVER mention 'provided context', 'context', 'the text', 'knowledge base', 'the database', or 'the files'. Avoid phrases like 'Based on the information provided...'. Speak as if you simply know the facts.\n"
                "5. STRICT GROUNDING: You are a strict RAG chatbot. You MUST answer strictly using ONLY the provided context and the SYLLABUS GROUNDING information (if provided). If both are empty or do not contain the answer, you MUST politely state that you do not have this information in your records. NEVER use your general pre-trained knowledge to answer questions.\n"
                "6. SYLLABUS PRIORITY: For questions about curriculum structure, Units, Modules, or specific topics, you MUST prioritize the **SYLLABUS GROUNDING** section. Provide the topics exactly as listed in the official curriculum.\n"
                "7. GROUNDING SAFEGUARD: If you are in STUDIES (SYLLABUS) mode and the SYLLABUS GROUNDING section is missing or empty, and the user asks for topics/curriculum, you MUST politely explain that you don't have their specific subject's syllabus yet. Ask them to ensure their **Course, Semester, and Subject** are correctly set in their profile or the sidebar.\n"
                "8. HELPFULNESS: Never be dismissive. If you don't know something, suggest where the user might find it or offer related helpful information.\n"
                "9. FORMATTING: Use professional Markdown. Use bold for key terms and bullet points for lists."
            )

        if syllabus_context:
            course_label = (course or "Academic").upper()
            sys_prompt += (
                f"\n\n### SYLLABUS GROUNDING ROLE\n"
                "You are provided with a `<syllabus_grounding>` JSON block in the user message containing the official course structure. \n"
                " - If the user asks 'what are the topics', 'give me the syllabus', or 'what is in Module/Unit X', you MUST use the titles and topics from that JSON.\n"
                " - Maintain the exact terminology of the topics as listed in the JSON (do not paraphrase or summarize the topic names).\n"
                " - Note: 'Unit', 'Module', 'Chapter', and 'Section' are equivalent terms. The user may use them interchangeably and use digits (e.g., 'Unit 4', 'Module 4') or Roman numerals (e.g., 'Unit IV', 'Module IV'). Map them correctly to the corresponding division in the JSON structure (e.g., 'Module 4' maps to 'Unit IV' or 'Unit 4').\n"
                " - If the JSON topics are detailed, include that detail in your answer."
            )

        # Build messages
        messages = [{"role": "system", "content": sys_prompt}]
        if history: messages.extend(history)
        context_str = context if context.strip() else "[NO CONTEXT FOUND IN KNOWLEDGE BASE. STRICT RULE: YOU MUST DECLINE TO ANSWER THIS QUESTION AS NO DATA WAS RETRIEVED.]"
        
        if syllabus_context:
            # Normalize terms to satisfy strict RAG constraints (e.g. Unit/Module)
            enriched_syllabus = syllabus_context
            enriched_syllabus = enriched_syllabus.replace('"Unit ', '"Unit/Module ').replace('"unit ', '"unit/module ')
            enriched_syllabus = enriched_syllabus.replace('"Module ', '"Unit/Module ').replace('"module ', '"unit/module ')
            
            user_content = (
                f"Syllabus Grounding Information (Subject: {subject or 'Academic'}):\n"
                f"<syllabus_grounding>\n{enriched_syllabus}\n</syllabus_grounding>\n\n"
                f"Reference Information:\n<context>\n{context_str}\n</context>\n\n"
                "Ground your answer strictly in the syllabus grounding and reference information provided. "
                "If the information is not present, politely state that you do not have this information in your records, and suggest that the user verify their Course/Semester/Subject settings in their profile or ask an administrator to update the system.\n\n"
                f"Question: {question}"
            )
        else:
            user_content = (
                f"Reference Information:\n<context>\n{context_str}\n</context>\n\n"
                "Ground your answer strictly in the reference information provided. "
                "If the information is not present, politely state that you do not have this information in your records, and suggest that they double-check their course settings or ask an administrator to update the system.\n\n"
                f"Question: {question}"
            )
            
        messages.append({
            "role": "user", 
            "content": user_content
        })

        # 1. Try Hugging Face (Primary)
        credits_depleted = False
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else Config.HUGGINGFACE_API_TOKEN
            
            hf_model = current_app.config.get("HF_LLM_MODEL") if current_app else Config.HF_LLM_MODEL
            hf_fallbacks = [
                hf_model,
                "Qwen/Qwen2.5-7B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3",
                "Qwen/Qwen3-8B"
            ]

            for mdl in hf_fallbacks:
                if not mdl: continue
                try:
                    out = AIService._chat_completion_with_fallback(
                        messages=messages,
                        model=mdl,
                        token=token,
                        max_tokens=1200,
                        temperature=0.2,
                        timeout=45
                    )
                    if out and len(out.strip()) > 0:
                        return AIService.clean_response(out.strip())
                except Exception as inner_ex:
                    err_str = str(inner_ex).lower()
                    if "402" in err_str or "payment required" in err_str or "credits" in err_str:
                        credits_depleted = True
                    logging.warning(f"Hugging Face generation fallback {mdl} failed: {inner_ex}")
                    continue
        except Exception as e:
            err_str = str(e).lower()
            if "402" in err_str or "payment required" in err_str or "credits" in err_str:
                credits_depleted = True
            logging.error(f"Hugging Face generation failed: {e}")

        if credits_depleted:
            return "Your Hugging Face API monthly included credits are depleted. Please purchase pre-paid credits, upgrade your Hugging Face account to Pro, or configure another API token in your settings."
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
            import datetime
            current_time_str = datetime.datetime.now().strftime("%B %d, %Y (YYYY-MM-DD: %Y-%m-%d)")
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Unibot, a university assistant analyzing webpage content. "
                        f"The current date is {current_time_str}. "
                        "You were developed and are maintained by Steby Varghese (King's Guard). If asked about your creator, developer, or who made you, always state clearly that you were created and are maintained by Steby Varghese (King's Guard).\n\n"
                        + (f"The user you are helping is named '{user_preferred_name}'. " if user_preferred_name else "The user has not provided a name yet. ")
                        + "\n\nUse ONLY the provided webpage text.\n\n"
                        "CORE RULES:\n"
                        "1. NATURAL RESPONSES: Speak naturally and directly. NEVER mention 'the provided webpage', 'the text', 'the database', or 'based on the content'. Answer as if you simply know the facts.\n"
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
            messages.append({"role": "user", "content": f"Webpage (Source: {source_url}):\n{context_str}\n\nUser Question/Instruction: {question}"})

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
                "meta-llama/Llama-3.2-3B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3",
                "Qwen/Qwen3-8B"
            ]
            for m in robust_models:
                if m not in fallbacks:
                    fallbacks.append(m)
            
            credits_depleted = False
            for mdl in fallbacks:
                if not mdl:
                    continue
                try:
                    out = AIService._chat_completion_with_fallback(
                        messages=messages,
                        model=mdl,
                        token=token or Config.HUGGINGFACE_API_TOKEN,
                        max_tokens=1300,
                        temperature=0.2,
                        timeout=45
                    )
                    if out and len(out.strip()) > 0:
                        return AIService.clean_response(out.strip())
                except Exception as e:
                    err_str = str(e).lower()
                    if "402" in err_str or "payment required" in err_str or "credits" in err_str:
                        credits_depleted = True
                    logging.warning(f"Website chat completion failed with {mdl}: {e}")
                    # Fallback to legacy text generation
                    try:
                        client_legacy = InferenceClient(token=token or Config.HUGGINGFACE_API_TOKEN, timeout=45)
                        prompt_legacy = (
                            "Instruction: Analyze the following webpage content and answer the question.\n"
                            f"Webpage Content:\n{context}\n\n"
                            f"Question: {question}\n\n"
                            "Answer:"
                        )
                        out = client_legacy.text_generation(
                            prompt_legacy,
                            model=mdl,
                            max_new_tokens=1200,
                            temperature=0.2,
                        )
                        if out and len(out.strip()) > 0:
                            return AIService.clean_response(out.strip())
                    except Exception as e2:
                        err_str2 = str(e2).lower()
                        if "402" in err_str2 or "payment required" in err_str2 or "credits" in err_str2:
                            credits_depleted = True
                        continue
                        
            if credits_depleted:
                return "Your Hugging Face API monthly included credits are depleted. Please purchase pre-paid credits, upgrade your Hugging Face account to Pro, or configure another API token in your settings."
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
    def generate_smalltalk(text: str, mode='syllabus', user_preferred_name=None, course=None, semester=None, subject=None):
        credits_depleted = False
        # Try Hugging Face first (Primary with robust fallbacks)
        try:
            token = current_app.config.get("HUGGINGFACE_API_TOKEN") if current_app else Config.HUGGINGFACE_API_TOKEN
            
            hf_model = current_app.config.get("HF_SMALLTALK_MODEL") if current_app else Config.HF_SMALLTALK_MODEL
            fallbacks = [
                hf_model,
                "Qwen/Qwen3-4B-Instruct-2507",
                "google/gemma-2-2b-it",
                "Qwen/Qwen2.5-7B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct"
            ]
            
            # If in general mode, we do NOT include coursework details
            has_coursework = (mode == 'syllabus' or mode == 'studies') and course and semester
            coursework_str = f" studying {course} (Semester {semester})" + (f", specifically {subject}." if subject else ".") if has_coursework else ""
            
            import datetime
            current_time_str = datetime.datetime.now().strftime("%B %d, %Y")
            system_content = (
                f"You are Unibot, a friendly university assistant developed and maintained by Steby Varghese (King's Guard). Respond briefly to the user's greeting. "
                f"The current date is {current_time_str}. "
                f"The user is {user_preferred_name or 'a student'}{coursework_str}. "
                f"IMPORTANT: You MUST start your response by greeting the user by their name '{user_preferred_name}' (e.g. 'Hello {user_preferred_name}!')" if user_preferred_name else ""
            )

            for mdl in fallbacks:
                if not mdl: continue
                try:
                    out = AIService._chat_completion_with_fallback(
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": text}
                        ],
                        model=mdl,
                        token=token,
                        max_tokens=64,
                        temperature=0.7,
                        timeout=8
                    )
                    if out and len(out.strip()) > 0:
                        return AIService.clean_response(out.strip())
                except Exception as inner_ex:
                    err_str = str(inner_ex).lower()
                    if "402" in err_str or "payment required" in err_str or "credits" in err_str:
                        credits_depleted = True
                    logging.warning(f"Hugging Face smalltalk fallback {mdl} failed: {inner_ex}")
                    continue
        except Exception as e:
            err_str = str(e).lower()
            if "402" in err_str or "payment required" in err_str or "credits" in err_str:
                credits_depleted = True
            logging.warning(f"Hugging Face smalltalk failed: {e}")
        
        if credits_depleted:
            return "Your Hugging Face API monthly included credits are depleted. Please purchase pre-paid credits, upgrade your Hugging Face account to Pro, or configure another API token in your settings."
        
        # Final hardcoded fallback
        name_part = f" {user_preferred_name}" if user_preferred_name else ""
        has_coursework = (mode == 'syllabus' or mode == 'studies')
        course_part = " about your courses or subjects" if has_coursework else ""
        doc_part = "documents or subjects" if has_coursework else "questions or system navigation"
        fallbacks_dict = {
            "nice": f"Glad you think so{name_part}!",
            "okay": f"I'm ready whenever you are{name_part}! Would you like to know anything more{course_part}?",
            "ok": f"Got it{name_part}! How can I help you further?",
            "thanks": f"You're very welcome{name_part}!",
            "thank you": f"You're very welcome{name_part}!",
            "hi": f"Hello{name_part}! How can I help you today?",
            "hello": f"Hi{name_part}! I'm here to help with your studies or any questions about this system."
        }
        t = (text or "").lower().strip().strip('!').strip('.')
        res = fallbacks_dict.get(t, f"I'm here to help{name_part}! Do you have any questions about your {doc_part}?")
        return AIService.clean_response(res)

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
    def fallback_parse_syllabus(text: str) -> str:
        """
        Locally parse syllabus text using regex to extract units and topics
        when Hugging Face APIs are unavailable or fail.
        """
        if not text:
            return '{"units": []}'
            
        import re
        import json
        
        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        # Pass 1: Look for Unit / Module / Chapter / Section headings
        unit_pattern = re.compile(
            r'^\s*(Unit|Module|Chapter|Section|Part|Paper)\s*(?:No\.?\s*|[-–—:]\s*)?([0-9]+|[ivxlcdm]+|[A-Z])\b(?:\s*[-–—:]\s*|\s+)?(.*)$',
            re.IGNORECASE
        )
        
        bullet_pattern = re.compile(
            r'^\s*[-*+•]?\s*(?:\d+\.|\b[a-zA-Z]\.|\b[ivxlcdm]+\.|\b\d+\b|\(\d+\)|\([a-zA-Z]\))?\s*(.*)$',
            re.IGNORECASE
        )
        
        # Exclude headings like References, Text Books, Page numbers, etc.
        exclude_pattern = re.compile(
            r'^\s*(?:References?|Text\s*Books?|Reference\s*Books?|Suggested\s*(?:Readings?|Read|Books?)|Bibliography|Prescribed\s*Books?|Course\s*(?:Outcomes?|Objectives?)|Evaluation\s*(?:Scheme|Criteria)|Assessment|Grading\s*Policy|Prerequisites?|Page\s*\d+)\b',
            re.IGNORECASE
        )
        
        connecting_words = {
            'and', 'or', 'of', 'for', 'with', 'the', 'a', 'an', 'to', 'in', 'at', 'by', 'from',
            'under', 'over', 'on', 'into', 'through', 'during', 'including', 'such', 'as'
        }
        
        def split_topic_line(text_line):
            # 1. Split by colon first (if not part of http/https)
            if ':' in text_line and not any(text_line.startswith(proto) for proto in ['http:', 'https:']):
                parts = []
                for p in text_line.split(':'):
                    parts.extend(split_topic_line(p))
                return parts
                
            # 2. Split by semicolon
            if ';' in text_line:
                parts = []
                for p in text_line.split(';'):
                    parts.extend(split_topic_line(p))
                return parts
                
            # 3. Split by period (sentence boundary)
            if '.' in text_line:
                parts = []
                for p in text_line.split('.'):
                    p_clean = p.strip()
                    if p_clean and not p_clean.isdigit():
                        parts.extend(split_topic_line(p_clean))
                return parts
                
            # 4. Split by comma (no length constraint!)
            if ',' in text_line:
                raw_parts = text_line.split(',')
                cleaned_parts = []
                for p in raw_parts:
                    p_clean = p.strip()
                    if p_clean.lower().startswith('and '):
                        p_clean = p_clean[4:].strip()
                    elif p_clean.lower().startswith('or '):
                        p_clean = p_clean[3:].strip()
                    if p_clean:
                        cleaned_parts.append(p_clean)
                if cleaned_parts:
                    return cleaned_parts
                    
            return [text_line]

        def merge_continuation_lines(lines_list):
            merged = []
            for line in lines_list:
                line_clean = line.strip()
                if not line_clean:
                    continue
                if not merged:
                    merged.append(line_clean)
                    continue
                
                prev = merged[-1]
                should_merge = False
                
                if prev.endswith('-'):
                    should_merge = True
                elif line_clean[0].islower():
                    should_merge = True
                else:
                    prev_words = prev.split()
                    if prev_words:
                        last_word = prev_words[-1].lower().strip(',.;:')
                        if last_word in connecting_words:
                            should_merge = True
                
                if should_merge:
                    if prev.endswith('-'):
                        merged[-1] = prev[:-1] + line_clean
                    else:
                        merged[-1] = prev + ' ' + line_clean
                else:
                    merged.append(line_clean)
            return merged

        units = []
        
        # First pass: Group raw lines by unit
        unit_groups = []
        current_group = None
        
        for line in raw_lines:
            match = unit_pattern.match(line)
            if match:
                unit_word = match.group(1).strip()
                unit_num = match.group(2).strip()
                unit_title = match.group(3).strip()
                
                full_title = f"{unit_word} {unit_num}"
                if unit_title:
                    full_title += f": {unit_title}"
                    
                current_group = {
                    "title": full_title,
                    "lines": []
                }
                unit_groups.append(current_group)
            elif current_group is not None:
                if exclude_pattern.match(line):
                    current_group = None # Stop collecting lines for this unit
                else:
                    current_group["lines"].append(line)
                
        # Process Pass 1 groups
        for group in unit_groups:
            processed_topics = []
            merged_lines = merge_continuation_lines(group["lines"])
            for line in merged_lines:
                topic_match = bullet_pattern.match(line)
                if topic_match:
                    raw_topic = topic_match.group(1).strip()
                    if len(raw_topic) >= 3 and not raw_topic.isdigit():
                        for topic in split_topic_line(raw_topic):
                            topic_clean = topic.strip().rstrip(',;: ')
                            if len(topic_clean) >= 3 and not topic_clean.isdigit() and topic_clean not in processed_topics:
                                processed_topics.append(topic_clean)
            units.append({
                "title": group["title"],
                "topics": processed_topics
            })
            
        # Pass 2: If no units found, look for numerical/alphabetical section headers (e.g. "1. Introduction")
        if not units:
            section_pattern = re.compile(
                r'^\s*(?:[0-9]+|[IVXLCDM]+|[A-Za-z])\s*[-–—.:]\s*(.+)$'
            )
            unit_groups = []
            current_group = None
            for line in raw_lines:
                match = section_pattern.match(line)
                if match:
                    current_group = {
                        "title": line,
                        "lines": []
                    }
                    unit_groups.append(current_group)
                elif current_group is not None:
                    if exclude_pattern.match(line):
                        current_group = None
                    else:
                        current_group["lines"].append(line)
                    
            for group in unit_groups:
                processed_topics = []
                merged_lines = merge_continuation_lines(group["lines"])
                for line in merged_lines:
                    topic_match = bullet_pattern.match(line)
                    if topic_match:
                        raw_topic = topic_match.group(1).strip()
                        if len(raw_topic) >= 3 and not raw_topic.isdigit():
                            for topic in split_topic_line(raw_topic):
                                topic_clean = topic.strip().rstrip(',;: ')
                                if len(topic_clean) >= 3 and not topic_clean.isdigit() and topic_clean not in processed_topics:
                                    processed_topics.append(topic_clean)
                units.append({
                    "title": group["title"],
                    "topics": processed_topics
                })
                
        # Pass 3: If still no units found, group everything under one default unit
        if not units:
            processed_topics = []
            filtered_lines = []
            for line in raw_lines:
                if exclude_pattern.match(line):
                    break
                filtered_lines.append(line)
                
            merged_lines = merge_continuation_lines(filtered_lines)
            for line in merged_lines:
                topic_match = bullet_pattern.match(line)
                if topic_match:
                    raw_topic = topic_match.group(1).strip()
                    if len(raw_topic) >= 3 and not raw_topic.isdigit():
                        for topic in split_topic_line(raw_topic):
                            topic_clean = topic.strip().rstrip(',;: ')
                            if len(topic_clean) >= 3 and not topic_clean.isdigit() and topic_clean not in processed_topics:
                                processed_topics.append(topic_clean)
            if processed_topics:
                units.append({
                    "title": "Syllabus Core Topics",
                    "topics": processed_topics
                })
                
        return json.dumps({"units": units})

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
                "Qwen/Qwen3-8B"
            ]

            credits_depleted = False
            out = ""
            for mdl in fallbacks:
                if not mdl: continue
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Syllabus Text:\n{processed_text}\n\nStrict JSON Knowledge Map:"}
                    ]
                    out = AIService._chat_completion_with_fallback(
                        messages=messages,
                        model=mdl,
                        token=token,
                        max_tokens=2500,
                        temperature=0.1,
                        timeout=90
                    )
                    if out and len(out.strip()) > 5:
                        break # Success
                except Exception as e:
                    err_str = str(e).lower()
                    if "402" in err_str or "payment required" in err_str or "credits" in err_str:
                        credits_depleted = True
                    logging.warning(f"Syllabus extraction attempt with {mdl} failed: {e}")
                    continue

            if not out:
                logging.warning("All LLM models failed to analyze syllabus. Invoking local fallback parser.")
                return AIService.fallback_parse_syllabus(processed_text)

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
                if 'units' not in json_data or not json_data['units']:
                    logging.warning("LLM returned JSON with no units. Invoking local fallback parser.")
                    return AIService.fallback_parse_syllabus(processed_text)
                return json.dumps(json_data)
            except Exception as e:
                logging.error(f"Syllabus analysis returned invalid JSON content: {e}. Invoking local fallback parser.")
                return AIService.fallback_parse_syllabus(processed_text)
                
        except Exception as e:
            logging.error(f"Syllabus analysis critical failure: {e}. Invoking local fallback parser.")
            return AIService.fallback_parse_syllabus(text)
