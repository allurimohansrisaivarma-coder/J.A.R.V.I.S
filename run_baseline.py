import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from jarvis.core.session import SessionManager
from jarvis.utils.metrics import metrics

async def run_baseline():
    print("Starting baseline tests...")
    session = SessionManager(config_path=Path(".env"))
    await session.initialize()
    
    queries = [
        "What time is it?",
        "Say hello",
        "Explain quantum computing in one sentence",
        "What is on my screen?",
        "Search the web for the current stock price of Apple",
        "Summarize my recent emails",
        "What's 2+2?",
        "Check my calendar",
        "Who is the president of the US?",
        "Goodbye"
    ]
    
    for i, q in enumerate(queries):
        print(f"\nQuery {i+1}: {q}")
        try:
            # We use process_input_stream to simulate live text/voice processing
            async for chunk in session.process_input_stream(q):
                pass
        except Exception as e:
            print(f"Error: {e}")
            
    await session.shutdown()
    print("\nBaseline tests completed. Check logs/performance.jsonl")

if __name__ == "__main__":
    asyncio.run(run_baseline())
