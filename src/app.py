import gradio as gr
from src.retrieval.vector_store import PharmaVectorStore
from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

# Initialize components with error handling
try:
    vector_store = PharmaVectorStore()
    client = OpenAI()
    logger.info("Components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    raise

def rag_pipeline(message: str, history: List[Tuple[str, str]]) -> str:
    """
    RAG pipeline with semantic chunking and summary search.
    
    Args:
        message: Current user question
        history: List of (user_msg, bot_msg) tuples
    
    Returns:
        Bot response with cited sources
    """
    
    # Input validation
    if not message or not message.strip():
        return "⚠️ Please ask a question about drug labels."
    
    try:
        # 1. RETRIEVE - Search for relevant documents
        logger.info(f"Processing query: {message[:100]}...")
        results = vector_store.search(message, k=3)
        
        # Handle empty results
        if not results or not results.get('ids') or not results['ids'][0]:
            return ("❌ I couldn't find relevant information in the drug labels. "
                   "Please try rephrasing your question or ask about a different topic.")
        
        # Extract results
        ids = results['ids'][0]
        metadatas = results['metadatas'][0]
        docs = results['documents'][0]
        
        # Build context and track sources
        context_parts = []
        sources_set = set()
        
        for i, meta in enumerate(metadatas):
            # Get raw content with proper fallback chain
            raw_text = meta.get("raw_content", "")
            
            if not raw_text or raw_text == "No content":
                raw_text = docs[i] if i < len(docs) else ""
    
            if not raw_text:
                continue
            
            section = meta.get("section", "Unknown Section")
            drug = meta.get("drug_name", "Unknown Drug")
            
            # Add to context with clear source markers
            context_parts.append(
                f"--- SOURCE {i+1}: {drug} - {section} ---\n{raw_text}"
            )
            
            # Track unique sources
            sources_set.add(f"{drug} ({section})")
        
        # Validate context
        if not context_parts:
            return ("⚠️ I found some results but couldn't extract meaningful content. "
                   "Please try a different question.")
        
        context_str = "\n\n".join(context_parts)
        logger.info(f"Retrieved {len(context_parts)} relevant sources")
        
        # 2. BUILD MESSAGES
        system_prompt = """You are a Clinical Pharmacist Assistant specializing in drug label interpretation.

INSTRUCTIONS:
- Answer questions STRICTLY based on the provided context from drug labels
- If the answer isn't in the context, say "I cannot find that information in the provided labels"
- Always cite the drug name and section when providing information
- Be precise with dosages, contraindications, and warnings
- If dosing differs by patient population (pediatric, geriatric, renal impairment), specify clearly
- Use clear, professional language appropriate for healthcare settings

IMPORTANT: Do not make assumptions or use external knowledge. Only use the context provided."""

        user_prompt = f"""Context from drug labels:

{context_str}

Question: {message}"""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (limit to last 5 exchanges)
        recent_history = history[-5:] if len(history) > 5 else history
        for user_msg, bot_msg in recent_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
        
        # Add current question with context
        messages.append({"role": "user", "content": user_prompt})
        
        # 3. GENERATE - Get LLM response
        logger.info("Generating response with GPT-4o-mini")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,  # Deterministic for medical info
            max_tokens=800
        )
        
        answer = response.choices[0].message.content
        
        # 4. FORMAT OUTPUT
        sources_list = sorted(list(sources_set))
        sources_str = ", ".join(sources_list) if sources_list else "No sources"
        
        final_output = f"{answer}\n\n---\n📚 **Sources:** {sources_str}"
        logger.info("Response generated successfully")
        return final_output
    
    except OpenAI.APIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return ("❌ **API Error:** Unable to generate response. Please check your OpenAI API key "
               "and try again.\n\nError details: " + str(e))
    
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}", exc_info=True)
        return (f"❌ **Error:** An unexpected error occurred while processing your question.\n\n"
               f"Please try again or rephrase your query.\n\n"
               f"Technical details: {str(e)}")

# Custom CSS for better styling and increased height
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
#component-0 {
    max-width: 900px;
    margin: auto;
}
/* Increase chatbot height */
.chatbot {
    height: 600px !important;
}
"""

# Launch UI
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 💊 Pharma-Aware RAG Agent
        
        Ask questions about drug labels using intelligent semantic search powered by RAG.
        
        
        """
    )
    
    chatbot = gr.ChatInterface(
        fn=rag_pipeline,
        chatbot=gr.Chatbot(height=600),  # Set explicit height
        examples=[
            "What is the recommended dosage for Mekinist?",
            "What are the contraindications?",
            "What were the most common adverse reactions in clinical trials?",
            "Is dose adjustment needed for renal impairment?",
            "What are the drug interactions I should be aware of?",
            "What are the warnings and precautions?"
        ],
    )
    
    gr.Markdown(
        """
        ---
        **Note:** This tool provides information from drug labels only. Always consult current 
        prescribing information and healthcare professionals for clinical decisions.
        """
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",  # Allow external connections
        server_port=7861,
        show_error=True
    )