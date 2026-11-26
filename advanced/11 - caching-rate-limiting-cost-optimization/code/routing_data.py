"""Labeled queries for the embedding-based routing classifier.

Label definitions:
  simple  - greeting, thanks, or a single fact lookup with a one-line answer.
  medium  - explain, summarize, describe or compare something well known; a paragraph answer.
  complex - write code, design, plan, analyze trade-offs, or reason through several
            conditions at once; a long, careful answer.

TRAIN_SET is what the classifier searches (the "nearest labeled example").
TEST_SET is held out: none of its queries are in TRAIN_SET, and none are near-copies.
"""

TRAIN_SET = [
    # --- simple (17) ---
    ("Hi there", "simple"),
    ("Thanks, that's all I needed", "simple"),
    ("What time does the store close today?", "simple"),
    ("Are you open on Sunday?", "simple"),
    ("What's your customer service phone number?", "simple"),
    ("Do you ship to Canada?", "simple"),
    ("How much is express shipping?", "simple"),
    ("Is standard shipping free?", "simple"),
    ("Can I pay with PayPal?", "simple"),
    ("Where is your corporate office?", "simple"),
    ("How many points do I need for a $10 reward?", "simple"),
    ("Do you sell garden hoses?", "simple"),
    ("What's the email address for support?", "simple"),
    ("Is the loyalty program free?", "simple"),
    ("Can I cancel my order?", "simple"),
    ("How long do refunds take?", "simple"),
    ("What is the capital of France?", "simple"),
    # --- medium (17) ---
    ("Explain how your return policy works for furniture", "medium"),
    ("What's the difference between standard and express shipping?", "medium"),
    ("Summarize the benefits of joining Northwind Rewards", "medium"),
    ("How does the price adjustment policy work?", "medium"),
    ("Explain what a restocking fee is and when it applies", "medium"),
    ("Compare returning an item online versus in a store", "medium"),
    ("Describe the steps to return an online order by mail", "medium"),
    ("What should I do if my item arrived damaged?", "medium"),
    ("Explain the difference between RAM and storage", "medium"),
    ("What is the difference between HTTP and HTTPS?", "medium"),
    ("Summarize how photosynthesis works", "medium"),
    ("Explain what an API is in simple terms", "medium"),
    ("Translate 'where is the checkout' into Spanish and French", "medium"),
    ("Describe how compound interest works", "medium"),
    ("What are the main differences between Python lists and tuples?", "medium"),
    ("Explain how DNS resolves a domain name", "medium"),
    ("List the pros and cons of buy-now-pay-later for a $300 order", "medium"),
    # --- complex (16) ---
    ("Write a Python script that parses a CSV of orders and computes the refunds owed under a 30-day return policy", "complex"),
    ("Design a database schema for an e-commerce returns system", "complex"),
    ("Analyze the trade-offs between microservices and a monolith for a mid-size retailer", "complex"),
    ("My recursive Fibonacci function is extremely slow for n=50. Why, and how do I fix it?", "complex"),
    ("Plan a 12-month strategy to reduce our product return rate by 20%", "complex"),
    ("Implement a thread-safe LRU cache in Java", "complex"),
    ("Compare three pricing strategies for a loyalty program and recommend one with justification", "complex"),
    ("Prove that the square root of 2 is irrational", "complex"),
    ("Write a SQL query that finds customers who returned more than 3 items in 6 months, with their total refund amount", "complex"),
    ("Evaluate whether we should move our inventory system to event sourcing, considering cost, risk and team skills", "complex"),
    ("I bought a sofa 10 days ago, opened it, found a small scratch today, and want a replacement shipped express to a different address. What are all my options and costs?", "complex"),
    ("Optimize an O(n^2) algorithm that finds pairs summing to a target into O(n log n) or better", "complex"),
    ("Write a unit test suite for a shopping cart class with discounts and sales tax", "complex"),
    ("Explain and implement the Raft consensus algorithm in Go", "complex"),
    ("Draft a zero-downtime migration plan from a REST API to GraphQL", "complex"),
    ("Build a regression model to forecast weekly garden-tool sales and justify the feature choices", "complex"),
]

TEST_SET = [
    # --- simple (7) ---
    ("Hello!", "simple"),
    ("What are your store hours on Saturday?", "simple"),
    ("Do you accept Apple Pay?", "simple"),
    ("How much does next-day delivery cost?", "simple"),
    ("Can I return a gift card?", "simple"),
    ("How do I reach a human agent?", "simple"),
    ("Who wrote Romeo and Juliet?", "simple"),
    # --- medium (7) ---
    ("Explain the difference between TCP and UDP", "medium"),
    ("How does white-glove furniture delivery work?", "medium"),
    ("Summarize your policy for damaged or defective items", "medium"),
    ("What's the difference between store credit and a refund?", "medium"),
    ("Explain what machine learning is to a beginner", "medium"),
    ("Describe how a credit card payment is processed", "medium"),
    ("How do loyalty points and free shipping work together for members?", "medium"),
    # --- complex (6) ---
    ("Write a Python function that checks whether an order is still eligible for return given the purchase date, item category and condition", "complex"),
    ("Analyze the pros and cons of serverless architecture for our online store", "complex"),
    ("Design a recommendation system for garden products based on purchase history", "complex"),
    ("My order shipped late, one item was damaged and another was final sale. Work out exactly which refunds and credits I'm entitled to", "complex"),
    ("Refactor a 2000-line legacy checkout module into testable components and outline the steps and risks", "complex"),
    ("Write a Dockerfile and CI pipeline for deploying a Flask app with a Postgres database", "complex"),
]

assert len(TRAIN_SET) == 50, len(TRAIN_SET)
assert len(TEST_SET) == 20, len(TEST_SET)
assert not {q for q, _ in TRAIN_SET} & {q for q, _ in TEST_SET}, "test queries leaked into training set"
