from budget_manager import ContextBudget
from compressor import ConversationManager
from tool_selector import select_tools, classify_intent
import json
from lost_in_middle_reordering import score_relevance, reorder_lost_in_middle

TOOL_REGISTRY = {
    "read_file": {
        "description": "Read contents of a file from disk",
        "tokens": 120,
        "categories": ["code", "files"],
    },
    "write_file": {
        "description": "Write content to a file on disk",
        "tokens": 150,
        "categories": ["code", "files"],
    },
    "search_code": {
        "description": "Search for patterns across the codebase",
        "tokens": 130,
        "categories": ["code"],
    },
    "run_command": {
        "description": "Execute a shell command and return output",
        "tokens": 140,
        "categories": ["code", "system"],
    },
    "create_calendar_event": {
        "description": "Create a new event on the calendar",
        "tokens": 180,
        "categories": ["calendar"],
    },
    "list_emails": {
        "description": "List recent emails from inbox",
        "tokens": 160,
        "categories": ["email"],
    },
    "send_email": {
        "description": "Compose and send an email message",
        "tokens": 200,
        "categories": ["email"],
    },
    "web_search": {
        "description": "Search the web for information",
        "tokens": 140,
        "categories": ["research"],
    },
    "query_database": {
        "description": "Run a SQL query against the database",
        "tokens": 170,
        "categories": ["code", "data"],
    },
    "generate_chart": {
        "description": "Generate a visualization from data",
        "tokens": 190,
        "categories": ["data", "visualization"],
    },
}

class ContextEngine:
    def __init__(self, max_tokens=128000, generation_reserve=4000):
        self.budget = ContextBudget(max_tokens, generation_reserve)
        self.conversation = ConversationManager(max_history_tokens=5000)
        self.system_prompt = (
            "You are a helpful AI assistant. You have access to tools for "
            "code editing, file management, web search, and data analysis. "
            "Use the appropriate tools for each task. Be concise and accurate."
        )
        self.knowledge_base = [
            "Python 3.12 introduced type parameter syntax for generic classes using bracket notation.",
            "The project uses PostgreSQL 16 with pgvector for embedding storage.",
            "Authentication is handled by Supabase Auth with JWT tokens.",
            "The frontend is built with Next.js 15 using the App Router.",
            "API rate limits are set to 100 requests per minute per user.",
            "The deployment pipeline uses GitHub Actions with Docker multi-stage builds.",
            "Test coverage must be above 80% for all new modules.",
            "The codebase follows the repository pattern for data access.",
        ]

    def assemble(self, query):
        self.budget = ContextBudget(self.budget.max_tokens, self.budget.generation_reserve)

        system_content, _ = self.budget.allocate("system_prompt", self.system_prompt, max_tokens=1000)

        tools, tool_tokens = select_tools(query, token_budget=2000)
        tool_text = json.dumps(list(tools.keys()))
        tool_content, _ = self.budget.allocate("tools", tool_text, max_tokens=2000)

        relevance = score_relevance(query, self.knowledge_base)
        threshold = 0.1
        relevant_docs = [
            doc for doc, score in zip(self.knowledge_base, relevance)
            if score >= threshold
        ]

        if relevant_docs:
            doc_scores = [s for s in relevance if s >= threshold]
            reordered = reorder_lost_in_middle(relevant_docs, doc_scores)
            doc_text = "\n".join(reordered)
            doc_content, _ = self.budget.allocate("retrieved_context", doc_text, max_tokens=3000)

        history_text = self.conversation.get_context()
        if history_text.strip():
            history_content, _ = self.budget.allocate("conversation_history", history_text, max_tokens=5000)

        query_content, _ = self.budget.allocate("user_query", query, max_tokens=500)

        return self.budget

    def chat(self, query):
        self.conversation.add_turn("user", query)
        budget = self.assemble(query)
        response = f"[Response to: {query[:50]}...]"
        self.conversation.add_turn("assistant", response)
        return budget


def run_budget_demo():
    print("=" * 60)
    print("  STEP 1: Context Budget Manager")
    print("=" * 60)

    budget = ContextBudget(max_tokens=128000, generation_reserve=4000)
    budget.allocate("system_prompt", "You are a helpful assistant. " * 20, max_tokens=500)
    budget.allocate("tools", json.dumps(list(TOOL_REGISTRY.keys())), max_tokens=2000)
    budget.allocate("retrieved_docs", "The project uses PostgreSQL. " * 50, max_tokens=3000)
    budget.allocate("history", "user: How do I fix this?\nassistant: Check the logs." * 10, max_tokens=5000)
    budget.allocate("query", "Fix the authentication bug in the JWT validation module", max_tokens=500)
    print(budget.report())


def run_reorder_demo():
    print(f"\n{'=' * 60}")
    print("  STEP 2: Lost-in-the-Middle Reordering")
    print("=" * 60)

    docs = [
        "Doc A: PostgreSQL connection pooling (most relevant)",
        "Doc B: Redis caching layer (somewhat relevant)",
        "Doc C: CSS styling guide (not relevant)",
        "Doc D: Database migration scripts (relevant)",
        "Doc E: CI/CD pipeline config (slightly relevant)",
        "Doc F: API authentication flow (relevant)",
        "Doc G: Frontend routing (not relevant)",
    ]
    scores = [0.95, 0.60, 0.05, 0.80, 0.30, 0.75, 0.10]

    reordered = reorder_lost_in_middle(docs, scores)

    print(f"\n  Original order (by insertion):")
    for doc, score in zip(docs, scores):
        print(f"    {score:.2f}  {doc}")

    print(f"\n  Reordered (high relevance at start + end, low in middle):")
    for i, doc in enumerate(reordered):
        position = "START" if i < 2 else "END" if i >= len(reordered) - 2 else "middle"
        print(f"    [{position:>6}]  {doc}")


def run_conversation_demo():
    print(f"\n{'=' * 60}")
    print("  STEP 3: Conversation History Compression")
    print("=" * 60)

    conv = ConversationManager(max_history_tokens=200)

    exchanges = [
        ("How do I set up the database?", "Run docker-compose up to start PostgreSQL. Then run the migrations with npm run migrate."),
        ("What about the environment variables?", "Copy .env.example to .env and fill in DATABASE_URL and JWT_SECRET."),
        ("The migrations are failing with a connection error.", "Check that PostgreSQL is running on port 5432 and the DATABASE_URL matches."),
        ("Fixed it. Now how do I seed test data?", "Run npm run seed which loads fixtures from the test/fixtures directory."),
        ("Can I run the tests now?", "Yes, run npm test. Make sure the test database is separate from development."),
    ]

    for i, (user_msg, assistant_msg) in enumerate(exchanges):
        conv.add_turn("user", user_msg)
        conv.add_turn("assistant", assistant_msg)
        stats = conv.stats()
        print(f"\n  After turn {i + 1}:")
        print(f"    Live turns: {stats['live_turns']}, Summaries: {stats['summaries']}, Tokens: {stats['tokens']}")

    print(f"\n  Final context:")
    for line in conv.get_context().split("\n"):
        print(f"    {line}")


def run_tool_selection_demo():
    print(f"\n{'=' * 60}")
    print("  STEP 4: Dynamic Tool Selection")
    print("=" * 60)

    test_queries = [
        "Fix the bug in auth.py where JWT tokens expire too early",
        "Schedule a meeting with the design team for next Tuesday at 2pm",
        "Show me the database query performance stats and generate a chart",
        "Search for best practices on error handling in Python",
        "Send an email to the team about the deployment schedule",
        "Read the config file and check for database connection settings",
    ]

    print(f"\n  All tools: {list(TOOL_REGISTRY.keys())} ({sum(t['tokens'] for t in TOOL_REGISTRY.values())} total tokens)")

    for q in test_queries:
        tools, tokens = select_tools(q)
        intents = classify_intent(q)
        all_tokens = sum(t["tokens"] for t in TOOL_REGISTRY.values())
        savings = all_tokens - tokens
        print(f"\n  Query: {q[:60]}...")
        print(f"    Intents: {intents}")
        print(f"    Selected: {list(tools.keys())}")
        print(f"    Tokens: {tokens} (saved {savings} by pruning)")


def run_full_pipeline_demo():
    print(f"\n{'=' * 60}")
    print("  STEP 5: Full Context Assembly Pipeline")
    print("=" * 60)

    engine = ContextEngine(max_tokens=128000, generation_reserve=4000)

    queries = [
        "Fix the bug in the authentication module where JWT tokens expire too early",
        "What is the best approach for implementing vector search with PostgreSQL?",
        "Schedule a team standup meeting for tomorrow morning",
    ]

    for q in queries:
        print(f"\n  Query: {q}")
        budget = engine.chat(q)
        print(budget.report())

    print(f"\n  --- After building up conversation history ---")
    for i in range(6):
        engine.conversation.add_turn("user", f"Follow-up question {i+1} about the database migration and authentication setup")
        engine.conversation.add_turn("assistant", f"Detailed response {i+1} covering the technical architecture and implementation steps")

    budget = engine.chat("Now implement all the changes we discussed in the previous turns")
    print(budget.report())
    conv_stats = engine.conversation.stats()
    print(f"\n  Conversation state: {conv_stats['live_turns']} live turns, {conv_stats['summaries']} summaries, {conv_stats['tokens']} tokens")


def run_relevance_demo():
    print(f"\n{'=' * 60}")
    print("  STEP 6: Relevance Scoring + Filtering")
    print("=" * 60)

    knowledge = [
        "Python 3.12 introduced type parameter syntax for generic classes.",
        "The project uses PostgreSQL 16 with pgvector for embedding storage.",
        "Authentication is handled by Supabase Auth with JWT tokens.",
        "The frontend is built with Next.js 15 using the App Router.",
        "API rate limits are set to 100 requests per minute per user.",
        "The deployment pipeline uses GitHub Actions with Docker builds.",
        "Test coverage must be above 80% for all new modules.",
        "Error logging uses structured JSON format with correlation IDs.",
    ]

    query = "How do I fix the JWT authentication token expiry bug?"
    scores = score_relevance(query, knowledge)

    print(f"\n  Query: {query}")
    print(f"\n  Relevance scores:")
    for doc, score in sorted(zip(knowledge, scores), key=lambda x: -x[1]):
        marker = "*" if score >= 0.05 else " "
        print(f"    {marker} {score:.3f}  {doc[:70]}...")

    threshold = 0.05
    included = sum(1 for s in scores if s >= threshold)
    excluded = len(scores) - included
    print(f"\n  Threshold {threshold}: {included} included, {excluded} excluded")
    print(f"  Token savings: ~{excluded * 20} tokens from excluded docs")


if __name__ == "__main__":
    run_budget_demo()
    run_reorder_demo()
    run_conversation_demo()
    run_tool_selection_demo()
    run_full_pipeline_demo()
    run_relevance_demo()
