import time
import sys
import os
import random

# ANSI Colors for Terminal
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
BOLD = '\033[1m'

def slow_print(text, delay=0.03, color=RESET):
    print(color, end="")
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print(RESET)

def step_header(step_num, title):
    print(f"\n{MAGENTA}{BOLD}{'='*60}{RESET}")
    print(f"{MAGENTA}{BOLD} STEP {step_num}: {title} {RESET}")
    print(f"{MAGENTA}{BOLD}{'='*60}{RESET}\n")
    time.sleep(1)

def run_demo():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{CYAN}{BOLD}🏥 UPKSK SWASTHYA MITRA - RAG PIPELINE DEMONSTRATION 🏥{RESET}")
    print(f"{CYAN}Visualizing the AI Backend Architecture...{RESET}\n")
    time.sleep(1)

    # User Input
    user_query = "I fell from my bike and my leg is swelling very badly. Where should I go?"
    slow_print(f"👤 USER INPUT: \"{user_query}\"", delay=0.05, color=YELLOW)
    time.sleep(1)

    # STEP 1: NLP Text Processing & Embedding
    step_header(1, "NLP Tokenization & Vector Embedding (all-MiniLM-L6-v2)")
    slow_print("The AI does not understand English. It only understands numbers.", color=CYAN)
    slow_print("Converting user query into a 384-dimensional mathematical vector...", color=CYAN)
    time.sleep(1)
    
    # Simulate vector representation
    vector_preview = [round(random.uniform(-0.1, 0.1), 4) for _ in range(8)]
    print(f"{GREEN}Vector Array: [{vector_preview[0]}, {vector_preview[1]}, {vector_preview[2]}, ..., {vector_preview[7]}] (Length: 384){RESET}")
    time.sleep(1.5)

    # STEP 2: Vector Database Search
    step_header(2, "Similarity Search in ChromaDB (Vector Database)")
    slow_print("Searching through 'health_insurance_data.txt' for relevant medical knowledge...", color=CYAN)
    slow_print("Calculating Cosine Similarity between the User's Vector and Hospital Data Vectors...", color=CYAN)
    time.sleep(1.5)

    print(f"{GREEN}>> Found Match 1 (Score: 0.89):{RESET} 'Orthopedics treats fractures, bone injuries, road accidents...'")
    time.sleep(0.5)
    print(f"{GREEN}>> Found Match 2 (Score: 0.82):{RESET} 'King George Medical University (KGMU), Lucknow has excellent Orthopedics...'")
    time.sleep(0.5)
    print(f"{GREEN}>> Found Match 3 (Score: 0.76):{RESET} 'Emergency Ambulance Service: Dial 108'")
    time.sleep(1.5)

    # STEP 3: Prompt Engineering (Augmentation)
    step_header(3, "Prompt Augmentation (The 'A' in RAG)")
    slow_print("Combining the User Query + Retrieved Database Information into a strict Prompt...", color=CYAN)
    time.sleep(1)
    
    prompt = f"""{YELLOW}SYSTEM PROMPT CONSTRUCTED:{RESET}
You are Swasthya Mitra. Answer using ONLY the following context:
[Context: Orthopedics handles road accidents. KGMU has Orthopedics. Emergency is 108.]
User Question: {user_query}"""
    print(prompt)
    time.sleep(2)

    # STEP 4: LLM Generation
    step_header(4, "LLM Generation (Llama 3 / Gemini)")
    slow_print("Sending Augmented Prompt to the Large Language Model...", color=CYAN)
    time.sleep(1)
    
    slow_print(f"🤖 AI RESPONSE GENERATED:", color=GREEN, delay=0.01)
    response = """🚨 This sounds like an emergency! Since you fell from a bike and have severe swelling, you might have a fracture or bone injury.
    
Based on UPKSK data, you need to see an **Orthopedics** specialist. 
If you are in Lucknow, please visit **King George Medical University (KGMU)**.
    
For an immediate ambulance, please dial **108**."""
    
    slow_print(response, delay=0.03, color=GREEN)
    
    print(f"\n{CYAN}{BOLD}✅ RAG PIPELINE COMPLETE.{RESET}")

if __name__ == "__main__":
    run_demo()
