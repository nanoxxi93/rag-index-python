"""
Main application entry point for the RAG pipeline.

Supports two execution modes:
- router: Uses RouterQueryEngine for structured routing
- agent: Uses AgentWorkflow / FunctionAgent for conversational AI

Configure via .env file:
    EXECUTION_MODE=router | agent
"""

import asyncio
from config.settings import settings
from core.orchestrator import RAGOrchestrator


async def main():
    """
    Main function to run the RAG pipeline.
    """
    settings.validate()

    print("=" * 60)
    print(f"RAG Pipeline - Execution Mode: {settings.EXECUTION_MODE.upper()}")
    print(f"Debug Mode: {settings.DEBUG}")
    print("=" * 60)

    orchestrator = RAGOrchestrator("data")
    await orchestrator.initialize()

    if settings.is_agent_mode():
        print("\n" + "=" * 50)
        print("AGENT MODE - Multiple queries with conversation")
        print("=" * 50)

        questions = [
            "Tell me about your latest work experience",
            "What technical skills do you have?",
            "Summarize the professional profile",
        ]

        for i, question in enumerate(questions):
            print(f"\nQuery {i + 1}: {question}")
            print("-" * 40)
            response = await orchestrator.query(
                question,
                reset_history=(i == 0),
            )
            print(f"Response: {response}")

        print("\n" + "=" * 50)
        print("CHAT EXAMPLE - Continuing conversation")
        print("=" * 50)

        follow_up = "Can you provide more details about the latest experience?"
        print(f"\nFollow-up: {follow_up}")
        print("-" * 40)
        response = await orchestrator.chat(follow_up)
        print(f"Response: {response}")

    else:
        print("\n" + "=" * 50)
        print("ROUTER MODE - Direct queries")
        print("=" * 50)

        question = "Tell me about your latest work experience"
        print(f"\nQuestion: {question}")
        print("-" * 40)
        response = await orchestrator.query(question)
        print(f"Response: {response}")

        print("\n" + "=" * 50)
        print("Additional Router Queries")
        print("=" * 50)

        more_questions = [
            "What are my technical skills?",
            "Summarize my professional profile",
        ]

        for q in more_questions:
            print(f"\nQuestion: {q}")
            print("-" * 40)
            response = await orchestrator.query(q)
            print(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
