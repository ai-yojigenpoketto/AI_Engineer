# AI Engineer Interview Preparation Guide 2025 - Part 2
## LLMs, AI Agents, RAG, System Design & More

---

# 3. Large Language Models (LLMs) & Generative AI

## 3.1 LLM Fundamentals & APIs

### Concept Definition
Large Language Models (LLMs) are transformer-based neural networks trained on massive text corpora to understand and generate human-like text. Production AI engineers must have 1+ years of hands-on experience with LLM APIs (OpenAI GPT-4, Anthropic Claude, Google Gemini, Azure OpenAI) and understand prompt engineering, context management, and LLM evaluation.

### Rationale & Industry Relevance
- **Foundational for 2025**: LLMs aren't experimental—they're the foundation of modern AI applications
- **Universal Requirement**: 90%+ of GenAI Engineer positions require production LLM experience
- **Revenue Impact**: LLM applications drive billions in revenue (ChatGPT, GitHub Copilot, etc.)
- **Career Critical**: Understanding LLMs is non-negotiable for AI Engineers in 2025

### Pros & Cons

**Advantages:**
- **Zero-shot capability**: Solve tasks without training data
- **Few-shot learning**: Adapt to new tasks with minimal examples
- **Generalization**: Handle diverse tasks with single model
- **Rapid iteration**: Change behavior through prompts, not retraining

**Limitations:**
- **Hallucinations**: Generate plausible but incorrect information
- **Context limits**: Finite context windows (4K-200K tokens depending on model)
- **Latency**: API calls add 200ms-2s latency
- **Cost**: $0.01-$0.10 per 1K tokens (expensive at scale)
- **Inconsistency**: Non-deterministic outputs (even with temperature=0)

**When to Use LLMs vs Traditional ML:**
- **LLMs**: Unstructured text, reasoning, generation, few-shot tasks
- **Traditional ML**: Structured data, real-time <10ms, high-volume low-cost, deterministic outputs

### Interview Questions

#### Question 10: Foundational - Prompt Engineering Best Practices
**Question:** "You're building a customer support chatbot. Write a prompt that reliably extracts structured information (customer issue, urgency, sentiment) from messages. Explain your prompt engineering strategy."

**Comprehensive Answer:**

```python
from typing import Literal, Optional
from pydantic import BaseModel, Field
import openai
import json

# ===== STRUCTURED OUTPUT WITH PYDANTIC =====
class CustomerIssue(BaseModel):
    """Structured customer support issue"""
    category: Literal['billing', 'technical', 'account', 'product', 'other']
    urgency: Literal['low', 'medium', 'high', 'critical']
    sentiment: Literal['positive', 'neutral', 'negative', 'angry']
    summary: str = Field(..., description="One sentence summary of the issue")
    requires_human: bool = Field(..., description="Whether issue requires human escalation")
    suggested_solution: Optional[str] = Field(None, description="Possible automated solution")

# ===== PROMPT ENGINEERING STRATEGIES =====

class PromptEngineering:
    """
    Production-grade prompt engineering

    Strategies:
    1. Few-shot examples
    2. Structured output (JSON)
    3. Chain of thought
    4. Role definition
    5. Constraints and guidelines
    """

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def basic_prompt(self, customer_message: str) -> str:
        """
        ❌ BAD: Vague, unstructured prompt

        Problems:
        - No format specification
        - No examples
        - No constraints
        - Unpredictable output
        """
        prompt = f"Analyze this customer message: {customer_message}"
        # Don't use this in production!
        return prompt

    def engineered_prompt(self, customer_message: str) -> str:
        """
        ✅ GOOD: Well-engineered prompt

        Techniques:
        - Clear role definition
        - Specific instructions
        - Structured output format
        - Few-shot examples
        - Constraints
        """
        prompt = f"""You are an expert customer support analyst for a SaaS company.

Your task: Analyze the customer message and extract key information in JSON format.

Output Format (strict JSON):
{{
  "category": "billing|technical|account|product|other",
  "urgency": "low|medium|high|critical",
  "sentiment": "positive|neutral|negative|angry",
  "summary": "one sentence summary",
  "requires_human": true|false,
  "suggested_solution": "optional automated solution"
}}

Classification Guidelines:
- Urgency "critical": Service completely down, revenue impact, security issue
- Urgency "high": Major functionality broken, multiple users affected
- Urgency "medium": Feature not working, workaround available
- Urgency "low": Enhancement request, minor issue

- Requires Human: true if angry, critical urgency, complex/ambiguous, or explicitly requesting human
- Suggested Solution: Only provide if issue is common and solution is straightforward

Examples:

Input: "I can't log in to my account! This is urgent, I have a presentation in 30 minutes!"
Output: {{
  "category": "account",
  "urgency": "high",
  "sentiment": "negative",
  "summary": "User unable to log in with time-sensitive need",
  "requires_human": true,
  "suggested_solution": "Send password reset link and escalate to support"
}}

Input: "Hey, I noticed a small typo on the pricing page. Just FYI!"
Output: {{
  "category": "other",
  "urgency": "low",
  "sentiment": "positive",
  "summary": "User reported typo on pricing page",
  "requires_human": false,
  "suggested_solution": "Log issue for content team review"
}}

Input: "I was charged twice this month! I want a refund NOW or I'm canceling my subscription!"
Output: {{
  "category": "billing",
  "urgency": "high",
  "sentiment": "angry",
  "summary": "Double billing issue with cancellation threat",
  "requires_human": true,
  "suggested_solution": null
}}

Now analyze this customer message:
"{customer_message}"

Output (JSON only, no additional text):"""

        return prompt

    def extract_issue(self, customer_message: str) -> CustomerIssue:
        """
        Extract structured issue using OpenAI function calling

        Benefits:
        - Guaranteed valid JSON
        - Type safety with Pydantic
        - No parsing errors
        - Supports complex schemas
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert customer support analyst.
Analyze messages and extract key information accurately."""
                },
                {
                    "role": "user",
                    "content": f"Analyze this customer message: {customer_message}"
                }
            ],
            functions=[{
                "name": "extract_customer_issue",
                "description": "Extract structured information from customer support message",
                "parameters": CustomerIssue.schema()
            }],
            function_call={"name": "extract_customer_issue"},
            temperature=0.3  # Lower temperature for consistency
        )

        # Parse function call arguments
        function_call = response.choices[0].message.function_call
        issue_data = json.loads(function_call.arguments)

        # Validate with Pydantic
        issue = CustomerIssue(**issue_data)
        return issue

    def extract_with_chain_of_thought(self, customer_message: str) -> CustomerIssue:
        """
        Chain of Thought prompting for complex reasoning

        Technique: Ask model to "think step by step"
        Benefits: Better accuracy on complex/ambiguous cases
        """
        prompt = f"""Analyze this customer support message step by step.

Message: "{customer_message}"

Step 1: What is the primary issue? (billing, technical, account, product, other)
Step 2: How urgent is this? Consider impact and time sensitivity.
Step 3: What is the customer's emotional state?
Step 4: Can this be automated or does it need human attention?
Step 5: Is there a standard solution?

After your analysis, provide final JSON output:
{{
  "category": "...",
  "urgency": "...",
  "sentiment": "...",
  "summary": "...",
  "requires_human": ...,
  "suggested_solution": "..."
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        # Extract JSON from response (might be embedded in text)
        content = response.choices[0].message.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        json_str = content[json_start:json_end]

        issue_data = json.loads(json_str)
        return CustomerIssue(**issue_data)

# ===== ADVANCED: PROMPT TEMPLATING =====

class PromptTemplate:
    """
    Reusable prompt templates with variable substitution

    Benefits:
    - Consistency across codebase
    - Easy testing and iteration
    - Version control for prompts
    """

    CUSTOMER_SUPPORT_TEMPLATE = """You are a {role} for {company}.

Context:
{context}

Task: {task}

Guidelines:
{guidelines}

Examples:
{examples}

User Input: {user_input}

Output Format: {output_format}"""

    @classmethod
    def render(cls, template: str, **kwargs) -> str:
        """Render template with variables"""
        return template.format(**kwargs)

    @classmethod
    def customer_support_prompt(
        cls,
        customer_message: str,
        company: str = "SaaS Company",
        include_examples: bool = True
    ) -> str:
        """Generate customer support analysis prompt"""

        examples = """
Example 1:
Input: "Can't log in!"
Output: {"category": "account", "urgency": "high", ...}

Example 2:
Input: "Love the new feature!"
Output: {"category": "product", "urgency": "low", "sentiment": "positive", ...}
""" if include_examples else "N/A"

        return cls.render(
            cls.CUSTOMER_SUPPORT_TEMPLATE,
            role="expert customer support analyst",
            company=company,
            context="You analyze customer messages to extract structured information.",
            task="Extract category, urgency, sentiment, and suggested actions.",
            guidelines="- Be accurate\n- Consider context\n- Escalate when needed",
            examples=examples,
            user_input=customer_message,
            output_format="Valid JSON matching CustomerIssue schema"
        )

# ===== PRODUCTION BEST PRACTICES =====

class ProductionPromptManager:
    """
    Production-grade prompt management

    Features:
    - Prompt versioning
    - A/B testing
    - Performance monitoring
    - Fallback strategies
    """

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.prompt_versions = {
            'v1': 'basic prompt...',
            'v2': 'improved prompt...',
            'v3': 'current best prompt...'
        }
        self.performance_metrics = {}

    def call_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        timeout: float = 10.0
    ) -> str:
        """
        Robust API calls with retry logic

        Production requirements:
        - Retry on rate limits (exponential backoff)
        - Timeout handling
        - Error logging
        - Fallback responses
        """
        import time

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    timeout=timeout
                )
                return response.choices[0].message.content

            except openai.RateLimitError:
                wait_time = (2 ** attempt) + (random.random() * 0.1)
                print(f"Rate limited, waiting {wait_time}s")
                time.sleep(wait_time)

            except openai.APITimeoutError:
                print(f"Timeout on attempt {attempt + 1}")
                continue

            except Exception as e:
                print(f"Error: {e}")
                if attempt == max_retries - 1:
                    # Final attempt failed, use fallback
                    return self.fallback_response()

        return self.fallback_response()

    def fallback_response(self) -> str:
        """Fallback when LLM unavailable"""
        return json.dumps({
            "category": "other",
            "urgency": "medium",
            "sentiment": "neutral",
            "summary": "Unable to analyze message (service unavailable)",
            "requires_human": True,
            "suggested_solution": None
        })

# ===== USAGE EXAMPLES =====
if __name__ == "__main__":
    # Initialize
    engineer = PromptEngineering(api_key="your-api-key")

    # Test messages
    messages = [
        "I can't access my account and I have an important meeting in 10 minutes!",
        "Just wanted to say the new dashboard looks great!",
        "You charged me $500 but I only signed up for the $50 plan. This is UNACCEPTABLE!",
    ]

    for msg in messages:
        print(f"\nMessage: {msg}")

        # Method 1: Function calling (recommended)
        issue = engineer.extract_issue(msg)
        print(f"Extracted: {issue.dict()}")

        # Method 2: Chain of thought (for complex cases)
        issue_cot = engineer.extract_with_chain_of_thought(msg)
        print(f"CoT Result: {issue_cot.dict()}")
```

**Prompt Engineering Strategy - Key Principles:**

1. **Role Definition**: Clear persona ("expert customer support analyst")
2. **Task Specification**: Explicit instructions on what to do
3. **Output Format**: Structured (JSON), with schema
4. **Examples**: 2-5 diverse examples (few-shot learning)
5. **Constraints**: Explicit guidelines and edge cases
6. **Temperature**: Lower (0.1-0.3) for consistency, higher (0.7-0.9) for creativity

**Production Best Practices:**

1. **Use Function Calling**: Guarantees valid JSON, type-safe
2. **Prompt Versioning**: Track prompts in version control
3. **A/B Testing**: Compare prompt variants with metrics
4. **Retry Logic**: Handle rate limits and timeouts
5. **Fallbacks**: Graceful degradation when LLM fails
6. **Monitoring**: Track latency, cost, quality metrics
7. **Caching**: Cache identical prompts (90% hit rate common)

**Common Mistakes:**
- Vague instructions ("analyze this")
- No output format specification
- No examples (especially for edge cases)
- Temperature too high (inconsistent outputs)
- Not handling API errors
- No validation of outputs

**Excellence Indicators:**
- Uses function calling or structured outputs
- Provides specific few-shot examples
- Discusses prompt versioning and testing
- Mentions cost and latency optimization
- Knows when to use chain of thought
- Can explain temperature effects

---

#### Question 11: Intermediate - Context Window Management
**Question:** "You're building a RAG system but documents exceed GPT-4's 128K context window. How do you handle this? Implement a solution."

**Comprehensive Answer:**

```python
from typing import List, Dict, Optional, Tuple
import tiktoken
import numpy as np
from dataclasses import dataclass

@dataclass
class Chunk:
    """Document chunk with metadata"""
    text: str
    chunk_id: int
    doc_id: str
    tokens: int
    embedding: Optional[np.ndarray] = None
    score: float = 0.0

class ContextWindowManager:
    """
    Manage context windows for LLMs

    Strategies:
    1. Chunking with overlap
    2. Semantic retrieval (only relevant chunks)
    3. Token counting and budget management
    4. Hierarchical summarization
    5. Map-reduce for long documents
    """

    def __init__(
        self,
        model: str = "gpt-4",
        max_context_tokens: int = 128000,
        response_tokens: int = 4096,
        system_tokens: int = 1000
    ):
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.response_tokens = response_tokens
        self.system_tokens = system_tokens

        # Available tokens for user content
        self.available_tokens = (
            max_context_tokens - response_tokens - system_tokens
        )

        # Initialize tokenizer
        self.tokenizer = tiktoken.encoding_for_model(model)

    def count_tokens(self, text: str) -> int:
        """Accurately count tokens for given model"""
        return len(self.tokenizer.encode(text))

    # ===== STRATEGY 1: INTELLIGENT CHUNKING =====

    def chunk_document(
        self,
        document: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        doc_id: str = "doc_0"
    ) -> List[Chunk]:
        """
        Chunk document with overlap

        Overlap is critical:
        - Prevents information loss at boundaries
        - Maintains context across chunks
        - Standard: 10-20% overlap
        """
        tokens = self.tokenizer.encode(document)
        chunks = []

        start = 0
        chunk_id = 0

        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                doc_id=doc_id,
                tokens=len(chunk_tokens)
            ))

            # Move start with overlap
            start += (chunk_size - overlap)
            chunk_id += 1

        return chunks

    # ===== STRATEGY 2: SEMANTIC CHUNKING =====

    def semantic_chunk(
        self,
        document: str,
        max_chunk_tokens: int = 1000
    ) -> List[Chunk]:
        """
        Chunk by semantic boundaries (paragraphs, sections)

        Better than fixed-size:
        - Respects natural boundaries
        - Maintains coherent context
        - Better for retrieval
        """
        # Split by paragraphs
        paragraphs = document.split('\n\n')

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_id = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If paragraph itself exceeds limit, split it
            if para_tokens > max_chunk_tokens:
                # Handle very long paragraph
                if current_chunk:
                    # Save current chunk
                    chunks.append(Chunk(
                        text='\n\n'.join(current_chunk),
                        chunk_id=chunk_id,
                        doc_id="semantic",
                        tokens=current_tokens
                    ))
                    chunk_id += 1
                    current_chunk = []
                    current_tokens = 0

                # Split long paragraph by sentences
                sentences = para.split('. ')
                for sent in sentences:
                    sent_tokens = self.count_tokens(sent)
                    if current_tokens + sent_tokens > max_chunk_tokens:
                        if current_chunk:
                            chunks.append(Chunk(
                                text='. '.join(current_chunk),
                                chunk_id=chunk_id,
                                doc_id="semantic",
                                tokens=current_tokens
                            ))
                            chunk_id += 1
                        current_chunk = [sent]
                        current_tokens = sent_tokens
                    else:
                        current_chunk.append(sent)
                        current_tokens += sent_tokens

            # Normal paragraph
            elif current_tokens + para_tokens > max_chunk_tokens:
                # Save current chunk and start new one
                chunks.append(Chunk(
                    text='\n\n'.join(current_chunk),
                    chunk_id=chunk_id,
                    doc_id="semantic",
                    tokens=current_tokens
                ))
                chunk_id += 1
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Don't forget last chunk
        if current_chunk:
            chunks.append(Chunk(
                text='\n\n'.join(current_chunk),
                chunk_id=chunk_id,
                doc_id="semantic",
                tokens=current_tokens
            ))

        return chunks

    # ===== STRATEGY 3: RETRIEVAL + RANKING =====

    def select_top_chunks(
        self,
        chunks: List[Chunk],
        query: str,
        token_budget: int = None
    ) -> List[Chunk]:
        """
        Select most relevant chunks within token budget

        Assumes chunks have embeddings and scores
        """
        if token_budget is None:
            token_budget = self.available_tokens

        # Sort by relevance score
        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)

        selected = []
        total_tokens = 0

        for chunk in sorted_chunks:
            if total_tokens + chunk.tokens <= token_budget:
                selected.append(chunk)
                total_tokens += chunk.tokens
            else:
                break

        return selected

    # ===== STRATEGY 4: HIERARCHICAL SUMMARIZATION =====

    def hierarchical_summarize(
        self,
        document: str,
        openai_client
    ) -> str:
        """
        Summarize very long documents hierarchically

        Process:
        1. Chunk document
        2. Summarize each chunk
        3. Combine summaries
        4. If still too long, recurse

        Use case: 500-page documents
        """
        # Base case: document fits in context
        tokens = self.count_tokens(document)
        if tokens <= self.available_tokens:
            return document

        # Chunk document
        chunks = self.chunk_document(document, chunk_size=50000)

        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            response = openai_client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": f"Summarize this text concisely:\n\n{chunk.text}"
                }],
                max_tokens=2000
            )
            summaries.append(response.choices[0].message.content)

        # Combine summaries
        combined = '\n\n'.join(summaries)

        # Recurse if still too long
        if self.count_tokens(combined) > self.available_tokens:
            return self.hierarchical_summarize(combined, openai_client)

        return combined

    # ===== STRATEGY 5: MAP-REDUCE PATTERN =====

    def map_reduce_query(
        self,
        document: str,
        query: str,
        openai_client
    ) -> str:
        """
        Map-reduce for answering queries over long documents

        Map phase: Query each chunk independently
        Reduce phase: Synthesize answers

        Benefits:
        - Handles documents of any length
        - Parallelizable
        - Comprehensive (all chunks considered)
        """
        # Chunk document
        chunks = self.chunk_document(document, chunk_size=10000)

        # MAP: Query each chunk
        chunk_answers = []
        for chunk in chunks:
            prompt = f"""Based on this text excerpt, answer the query.
If the excerpt doesn't contain relevant information, say "Not relevant".

Excerpt:
{chunk.text}

Query: {query}

Answer:"""

            response = openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            answer = response.choices[0].message.content

            if "not relevant" not in answer.lower():
                chunk_answers.append(answer)

        # REDUCE: Synthesize answers
        if not chunk_answers:
            return "No relevant information found in document."

        synthesis_prompt = f"""Given these partial answers from different parts of a document,
synthesize a comprehensive answer to the query.

Query: {query}

Partial Answers:
{chr(10).join(f"{i+1}. {ans}" for i, ans in enumerate(chunk_answers))}

Synthesized Answer:"""

        response = openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": synthesis_prompt}],
            max_tokens=1000
        )

        return response.choices[0].message.content

    # ===== STRATEGY 6: SLIDING WINDOW =====

    def sliding_window_search(
        self,
        document: str,
        query: str,
        openai_client,
        window_size: int = 8000,
        stride: int = 6000
    ) -> Tuple[str, int]:
        """
        Sliding window to find relevant section in long document

        Returns: (best_window_text, start_position)
        """
        tokens = self.tokenizer.encode(document)
        best_score = -1
        best_window = ""
        best_pos = 0

        # Slide window across document
        for start in range(0, len(tokens), stride):
            end = start + window_size
            window_tokens = tokens[start:end]
            window_text = self.tokenizer.decode(window_tokens)

            # Score window relevance to query
            prompt = f"""Rate how relevant this text is to the query on a scale of 0-10.
Output only the number.

Query: {query}

Text: {window_text[:500]}...

Relevance (0-10):"""

            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Faster for scoring
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10
            )

            try:
                score = int(response.choices[0].message.content.strip())
                if score > best_score:
                    best_score = score
                    best_window = window_text
                    best_pos = start
            except:
                continue

            if end >= len(tokens):
                break

        return best_window, best_pos

# ===== COMPLETE PRODUCTION EXAMPLE =====

class ProductionRAGSystem:
    """
    Production RAG system with context management

    Handles:
    - Documents exceeding context limits
    - Optimal chunk selection
    - Token budget management
    - Fallback strategies
    """

    def __init__(self, openai_client, embedding_model):
        self.client = openai_client
        self.embedding_model = embedding_model
        self.context_manager = ContextWindowManager()

    def query_long_document(
        self,
        document: str,
        query: str,
        strategy: str = 'retrieval'
    ) -> str:
        """
        Query document using specified strategy

        Strategies:
        - 'retrieval': Semantic retrieval of top chunks
        - 'map_reduce': Map-reduce pattern
        - 'hierarchical': Hierarchical summarization
        - 'sliding_window': Sliding window search
        """
        if strategy == 'retrieval':
            # Chunk and embed
            chunks = self.context_manager.semantic_chunk(document)

            # Embed query
            query_embedding = self.embedding_model.encode(query)

            # Embed chunks and compute scores
            for chunk in chunks:
                chunk.embedding = self.embedding_model.encode(chunk.text)
                chunk.score = np.dot(query_embedding, chunk.embedding)

            # Select top chunks within budget
            selected_chunks = self.context_manager.select_top_chunks(
                chunks,
                query,
                token_budget=100000  # Leave room for query + response
            )

            # Build context
            context = '\n\n'.join(chunk.text for chunk in selected_chunks)

            # Query with context
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "Answer questions based on the provided context."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}"
                    }
                ]
            )
            return response.choices[0].message.content

        elif strategy == 'map_reduce':
            return self.context_manager.map_reduce_query(
                document, query, self.client
            )

        elif strategy == 'hierarchical':
            summary = self.context_manager.hierarchical_summarize(
                document, self.client
            )
            # Then query the summary
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": f"Based on this summary:\n{summary}\n\nQuestion: {query}"
                }]
            )
            return response.choices[0].message.content

        elif strategy == 'sliding_window':
            window, _ = self.context_manager.sliding_window_search(
                document, query, self.client
            )
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": f"Context:\n{window}\n\nQuestion: {query}"
                }]
            )
            return response.choices[0].message.content

# Usage
if __name__ == "__main__":
    # Very long document (500K tokens)
    document = "..." * 100000  # Hypothetical long doc

    manager = ContextWindowManager(model="gpt-4", max_context_tokens=128000)

    # Strategy 1: Semantic chunking
    chunks = manager.semantic_chunk(document, max_chunk_tokens=1000)
    print(f"Document split into {len(chunks)} semantic chunks")

    # Strategy 2: Map-reduce
    answer = manager.map_reduce_query(
        document,
        "What are the main findings?",
        openai_client
    )
    print(f"Map-reduce answer: {answer}")
```

**Key Strategies for Context Overflow:**

1. **Retrieval (RAG)**: Only include relevant chunks (90% of use cases)
2. **Map-Reduce**: Process chunks independently, synthesize
3. **Hierarchical Summarization**: Iteratively summarize until fits
4. **Sliding Window**: Find most relevant section
5. **Hybrid**: Combine strategies based on query type

**Production Considerations:**

- **Token Counting**: Use tiktoken for accurate counts (not len(text.split()))
- **Chunking**: Semantic > fixed-size (respects boundaries)
- **Overlap**: 10-20% prevents information loss
- **Budget Management**: Reserve tokens for response (4K+)
- **Caching**: Cache chunked/embedded documents

**Common Mistakes:**
- Naive character-based chunking (doesn't align with tokens)
- No overlap between chunks (loses context)
- Not considering system message tokens
- Exceeding context limit (API error)
- Not handling edge cases (very long sentences)

**Excellence Indicators:**
- Uses tiktoken for accurate token counting
- Explains trade-offs of different strategies
- Mentions embedding-based retrieval
- Discusses cost optimization
- Knows when to use each strategy

---

#### Question 12: Advanced - LLM Evaluation & Performance Tuning
**Question:** "Your LLM-based application has inconsistent quality. Design a comprehensive evaluation framework. How do you measure and improve performance?"

**Comprehensive Answer:**

```python
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import json

@dataclass
class EvaluationResult:
    """Single evaluation result"""
    input: str
    output: str
    expected: Optional[str]
    metrics: Dict[str, float]
    latency_ms: float
    cost_usd: float
    timestamp: str

class LLMEvaluationFramework:
    """
    Comprehensive LLM evaluation framework

    Evaluation types:
    1. Automated metrics (BLEU, ROUGE, exact match)
    2. LLM-as-judge (GPT-4 evaluates outputs)
    3. Human evaluation (gold standard)
    4. Task-specific metrics
    5. Operational metrics (latency, cost)
    """

    def __init__(self, openai_client):
        self.client = openai_client
        self.results: List[EvaluationResult] = []

    # ===== AUTOMATED METRICS =====

    def exact_match(self, prediction: str, ground_truth: str) -> float:
        """Exact string match (strict)"""
        return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0

    def contains_match(self, prediction: str, ground_truth: str) -> float:
        """Ground truth appears in prediction"""
        return 1.0 if ground_truth.lower() in prediction.lower() else 0.0

    def semantic_similarity(
        self,
        prediction: str,
        ground_truth: str,
        embedding_model
    ) -> float:
        """
        Cosine similarity between embeddings

        Better than exact match:
        - Captures semantic meaning
        - Robust to paraphrasing
        """
        pred_emb = embedding_model.encode(prediction)
        truth_emb = embedding_model.encode(ground_truth)

        similarity = np.dot(pred_emb, truth_emb) / (
            np.linalg.norm(pred_emb) * np.linalg.norm(truth_emb)
        )
        return float(similarity)

    def token_overlap_f1(self, prediction: str, ground_truth: str) -> float:
        """
        F1 score based on token overlap

        Good for:
        - Extractive QA
        - Named entity extraction
        - Fact recall
        """
        pred_tokens = set(prediction.lower().split())
        truth_tokens = set(ground_truth.lower().split())

        if not truth_tokens:
            return 0.0

        overlap = pred_tokens & truth_tokens

        precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
        recall = len(overlap) / len(truth_tokens) if truth_tokens else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    # ===== LLM-AS-JUDGE =====

    def llm_judge_quality(
        self,
        input_text: str,
        output_text: str,
        criteria: str = "helpfulness, accuracy, coherence"
    ) -> Dict[str, float]:
        """
        Use GPT-4 to evaluate output quality

        Benefits:
        - Nuanced evaluation
        - Adapts to task
        - Correlates well with human judgment

        Limitations:
        - Expensive
        - Latency
        - Potential bias
        """
        judge_prompt = f"""You are an expert evaluator of AI assistant responses.

Evaluate the following response on these criteria: {criteria}

Input: "{input_text}"

Response: "{output_text}"

For each criterion, provide:
1. Score (1-5, where 5 is best)
2. Brief justification

Output format (JSON):
{{
  "helpfulness": {{"score": <1-5>, "reason": "..."}},
  "accuracy": {{"score": <1-5>, "reason": "..."}},
  "coherence": {{"score": <1-5>, "reason": "..."}},
  "overall": {{"score": <1-5>, "reason": "..."}}
}}"""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.3
        )

        # Parse JSON
        content = response.choices[0].message.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        scores_dict = json.loads(content[json_start:json_end])

        # Extract scores
        scores = {
            criterion: scores_dict[criterion]['score']
            for criterion in ['helpfulness', 'accuracy', 'coherence', 'overall']
        }

        return scores

    def llm_judge_pairwise(
        self,
        input_text: str,
        output_a: str,
        output_b: str
    ) -> str:
        """
        Pairwise comparison (A vs B)

        More reliable than absolute scoring:
        - Easier for model to compare
        - Reduces calibration issues
        - Used in Chatbot Arena
        """
        judge_prompt = f"""Compare these two AI assistant responses.
Which is better overall? Consider helpfulness, accuracy, and clarity.

Input: "{input_text}"

Response A: "{output_a}"

Response B: "{output_b}"

Output one of: "A", "B", or "TIE"
Provide brief reasoning.

Format:
Winner: <A|B|TIE>
Reasoning: <explanation>"""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.3
        )

        content = response.choices[0].message.content
        if 'Winner: A' in content:
            return 'A'
        elif 'Winner: B' in content:
            return 'B'
        else:
            return 'TIE'

    # ===== TASK-SPECIFIC METRICS =====

    def evaluate_rag_response(
        self,
        query: str,
        response: str,
        retrieved_docs: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        RAG-specific evaluation

        Metrics:
        - Faithfulness: Response grounded in retrieved docs?
        - Answer relevance: Response addresses query?
        - Context relevance: Retrieved docs are relevant?
        """
        metrics = {}

        # 1. Faithfulness (groundedness)
        faithfulness_prompt = f"""Does the response only contain information present in the provided context?

Context:
{chr(10).join(retrieved_docs)}

Response: "{response}"

Answer "yes" or "no" and explain briefly."""

        faith_response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": faithfulness_prompt}],
            temperature=0.0
        )
        faithfulness = 1.0 if 'yes' in faith_response.choices[0].message.content.lower() else 0.0
        metrics['faithfulness'] = faithfulness

        # 2. Answer relevance
        relevance_prompt = f"""Does this response adequately answer the question?

Question: "{query}"

Response: "{response}"

Rate 0-1 (0=not relevant, 1=highly relevant). Output only the number."""

        rel_response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": relevance_prompt}],
            temperature=0.0
        )
        try:
            relevance = float(rel_response.choices[0].message.content.strip())
            metrics['answer_relevance'] = relevance
        except:
            metrics['answer_relevance'] = 0.5

        # 3. Context relevance (if ground truth available)
        if ground_truth:
            context_contains_answer = any(
                ground_truth.lower() in doc.lower()
                for doc in retrieved_docs
            )
            metrics['context_recall'] = 1.0 if context_contains_answer else 0.0

        return metrics

    def evaluate_classification(
        self,
        prediction: str,
        ground_truth: str,
        all_predictions: List[str] = None,
        all_ground_truths: List[str] = None
    ) -> Dict[str, float]:
        """
        Classification task metrics

        Returns:
        - Accuracy
        - Precision, Recall, F1 (if multi-example)
        - Confusion matrix
        """
        metrics = {}

        # Single example accuracy
        metrics['accuracy'] = 1.0 if prediction == ground_truth else 0.0

        # If batch provided, compute aggregate metrics
        if all_predictions and all_ground_truths:
            from sklearn.metrics import precision_recall_fscore_support, accuracy_score

            # Overall accuracy
            metrics['batch_accuracy'] = accuracy_score(all_ground_truths, all_predictions)

            # Per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                all_ground_truths,
                all_predictions,
                average='weighted'
            )

            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1

        return metrics

    # ===== OPERATIONAL METRICS =====

    def measure_latency(self, func: Callable, *args, **kwargs) -> tuple:
        """Measure function latency"""
        import time
        start = time.time()
        result = func(*args, **kwargs)
        latency_ms = (time.time() - start) * 1000
        return result, latency_ms

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-4"
    ) -> float:
        """
        Estimate API cost

        Pricing (as of 2025):
        - GPT-4: $0.03/1K input, $0.06/1K output
        - GPT-3.5: $0.0015/1K input, $0.002/1K output
        - Claude-3-Opus: $0.015/1K input, $0.075/1K output
        """
        pricing = {
            'gpt-4': (0.03, 0.06),
            'gpt-3.5-turbo': (0.0015, 0.002),
            'claude-3-opus-20240229': (0.015, 0.075),
        }

        input_price, output_price = pricing.get(model, (0.03, 0.06))

        cost = (input_tokens / 1000 * input_price +
                output_tokens / 1000 * output_price)

        return cost

    # ===== COMPREHENSIVE EVALUATION =====

    def evaluate_model_on_benchmark(
        self,
        test_cases: List[Dict],
        model_func: Callable,
        embedding_model=None
    ) -> pd.DataFrame:
        """
        Evaluate model on benchmark dataset

        test_cases format:
        [
          {"input": "...", "expected": "...", "type": "qa"},
          ...
        ]

        Returns comprehensive metrics DataFrame
        """
        results = []

        for case in test_cases:
            input_text = case['input']
            expected = case.get('expected')
            task_type = case.get('type', 'general')

            # Run model
            output, latency = self.measure_latency(model_func, input_text)

            # Compute metrics
            metrics = {}

            # Automated metrics
            if expected:
                metrics['exact_match'] = self.exact_match(output, expected)
                metrics['contains'] = self.contains_match(output, expected)
                metrics['token_f1'] = self.token_overlap_f1(output, expected)

                if embedding_model:
                    metrics['semantic_sim'] = self.semantic_similarity(
                        output, expected, embedding_model
                    )

            # LLM-as-judge
            judge_scores = self.llm_judge_quality(input_text, output)
            metrics.update({
                'judge_' + k: v for k, v in judge_scores.items()
            })

            # Task-specific
            if task_type == 'rag' and 'retrieved_docs' in case:
                rag_metrics = self.evaluate_rag_response(
                    input_text, output,
                    case['retrieved_docs'],
                    expected
                )
                metrics.update(rag_metrics)

            # Operational
            metrics['latency_ms'] = latency

            # Estimate cost (simplified)
            metrics['cost_usd'] = self.estimate_cost(
                len(input_text.split()) * 1.3,  # Rough token estimate
                len(output.split()) * 1.3,
                model='gpt-4'
            )

            results.append({
                'input': input_text,
                'output': output,
                'expected': expected,
                **metrics
            })

        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)

        # Summary statistics
        print("\n=== EVALUATION SUMMARY ===")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(df[numeric_cols].describe())

        print(f"\nTotal cost: ${df['cost_usd'].sum():.4f}")
        print(f"Avg latency: {df['latency_ms'].mean():.2f}ms")

        return df

    # ===== A/B TESTING =====

    def ab_test(
        self,
        test_cases: List[Dict],
        model_a: Callable,
        model_b: Callable,
        metric: str = 'judge_overall'
    ) -> Dict:
        """
        A/B test two models/prompts

        Returns statistical significance and winner
        """
        from scipy.stats import ttest_ind

        # Evaluate both models
        results_a = self.evaluate_model_on_benchmark(test_cases, model_a)
        results_b = self.evaluate_model_on_benchmark(test_cases, model_b)

        # Compare on metric
        scores_a = results_a[metric].values
        scores_b = results_b[metric].values

        # Statistical test
        t_stat, p_value = ttest_ind(scores_a, scores_b)

        winner = 'A' if scores_a.mean() > scores_b.mean() else 'B'
        significant = p_value < 0.05

        return {
            'winner': winner,
            'mean_a': scores_a.mean(),
            'mean_b': scores_b.mean(),
            'difference': abs(scores_a.mean() - scores_b.mean()),
            'p_value': p_value,
            'significant': significant,
            't_statistic': t_stat
        }

# ===== PERFORMANCE IMPROVEMENT STRATEGIES =====

class LLMPerformanceOptimizer:
    """
    Systematic LLM performance improvement

    Strategies:
    1. Prompt optimization
    2. Few-shot example selection
    3. Model selection
    4. Hyperparameter tuning
    5. Fine-tuning (if needed)
    """

    def __init__(self, openai_client):
        self.client = openai_client

    def optimize_prompt(
        self,
        base_prompt: str,
        test_cases: List[Dict],
        iterations: int = 5
    ) -> str:
        """
        Iteratively improve prompt using feedback

        Process:
        1. Evaluate current prompt
        2. Identify failure cases
        3. Generate improved prompt
        4. Test and repeat
        """
        current_prompt = base_prompt
        evaluator = LLMEvaluationFramework(self.client)

        for i in range(iterations):
            print(f"\n=== Iteration {i+1} ===")

            # Evaluate current prompt
            def model_with_prompt(input_text):
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": current_prompt.format(input=input_text)}]
                )
                return response.choices[0].message.content

            results = evaluator.evaluate_model_on_benchmark(
                test_cases,
                model_with_prompt
            )

            # Find failure cases
            failures = results[results['judge_overall'] < 4.0]

            if len(failures) == 0:
                print("Perfect performance! Done.")
                break

            # Generate improved prompt
            improvement_prompt = f"""Current prompt:
{current_prompt}

This prompt fails on these cases:
{failures[['input', 'output', 'expected']].to_string()}

Suggest an improved version of the prompt that addresses these failures.
Output only the improved prompt."""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": improvement_prompt}],
                temperature=0.7
            )

            current_prompt = response.choices[0].message.content
            print(f"New prompt: {current_prompt[:200]}...")

        return current_prompt

    def select_best_examples(
        self,
        available_examples: List[Dict],
        query: str,
        k: int = 3,
        embedding_model=None
    ) -> List[Dict]:
        """
        Dynamic few-shot example selection

        Choose most relevant examples for each query
        Better than static examples
        """
        if not embedding_model:
            # Random selection fallback
            import random
            return random.sample(available_examples, min(k, len(available_examples)))

        # Embed query
        query_emb = embedding_model.encode(query)

        # Score examples by relevance
        scored_examples = []
        for ex in available_examples:
            ex_emb = embedding_model.encode(ex['input'])
            score = np.dot(query_emb, ex_emb)
            scored_examples.append((score, ex))

        # Sort and select top k
        scored_examples.sort(reverse=True, key=lambda x: x[0])
        selected = [ex for _, ex in scored_examples[:k]]

        return selected

# Usage
if __name__ == "__main__":
    import openai
    client = openai.OpenAI()

    # Create evaluation framework
    evaluator = LLMEvaluationFramework(client)

    # Test cases
    test_cases = [
        {
            "input": "What is the capital of France?",
            "expected": "Paris",
            "type": "qa"
        },
        {
            "input": "Explain quantum computing",
            "expected": "Quantum computing uses quantum bits (qubits) that can exist in superposition...",
            "type": "explanation"
        }
    ]

    # Define model function
    def my_model(input_text):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": input_text}]
        )
        return response.choices[0].message.content

    # Evaluate
    results = evaluator.evaluate_model_on_benchmark(test_cases, my_model)

    # A/B test two prompts
    def model_a(input_text):
        return client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Answer this: {input_text}"}]
        ).choices[0].message.content

    def model_b(input_text):
        return client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Provide a detailed answer: {input_text}"}]
        ).choices[0].message.content

    ab_results = evaluator.ab_test(test_cases, model_a, model_b)
    print(f"\nA/B Test Winner: {ab_results['winner']}")
    print(f"Statistically significant: {ab_results['significant']}")
```

**Comprehensive Evaluation Strategy:**

1. **Automated Metrics** (fast, cheap):
   - Exact match
   - Token overlap F1
   - Semantic similarity (embeddings)

2. **LLM-as-Judge** (expensive, accurate):
   - Absolute scoring
   - Pairwise comparison
   - Correlates 0.8+ with human judgment

3. **Human Evaluation** (slow, gold standard):
   - Crowdsourcing (Scale AI, Surge AI)
   - Expert review
   - User feedback

4. **Task-Specific**:
   - RAG: Faithfulness, answer relevance, context precision
   - Classification: Accuracy, F1, confusion matrix
   - Generation: Coherence, creativity, factuality

5. **Operational**:
   - Latency (P50, P95, P99)
   - Cost per query
   - Throughput (QPS)

**Improvement Strategies:**

1. **Prompt Engineering**: 80% of quality issues
2. **Example Selection**: Dynamic few-shot
3. **Model Choice**: GPT-4 vs Claude vs Gemini
4. **Temperature**: Lower for consistency
5. **Fine-tuning**: Last resort (expensive, time-consuming)

**Common Mistakes:**
- Evaluating on training data
- Using only automated metrics
- Not measuring operational metrics
- No statistical significance testing
- Optimizing for wrong metric

**Excellence Indicators:**
- Uses multiple evaluation methods
- Discusses metric trade-offs
- Mentions statistical significance
- Tracks operational metrics
- Knows LLM-as-judge limitations

---

### Summary: LLMs & Generative AI

**Critical Takeaways:**
- LLMs are foundational (not experimental) in 2025
- Prompt engineering: role, task, format, examples, constraints
- Context management: chunking, retrieval, map-reduce
- Evaluation: automated + LLM-judge + human + operational

**Study Priority:**
- **High**: Prompt engineering, OpenAI/Claude API, RAG basics
- **Medium**: Evaluation frameworks, context management
- **Lower**: Fine-tuning (unless specialist role)

**Preparation Time:**
- LLM API basics: 1 week
- Prompt engineering: 2 weeks
- RAG & context management: 2 weeks
- Evaluation: 1 week

---

# 4. AI Agents & Agentic AI

## 4.1 AI Agent Fundamentals

### Concept Definition
AI Agents are autonomous systems that use LLMs combined with tools, memory, and reasoning capabilities to accomplish complex, multi-step tasks with minimal human intervention. Unlike simple chatbots, agents can plan, execute actions, observe results, and adapt their approach. Agentic AI is the **dominant trend of 2025**, with frameworks like LangGraph, AutoGen, and CrewAI becoming essential skills.

### Rationale & Industry Relevance
- **Biggest Development of 2025**: "Agentic" was the word of the year
- **Market Shift**: 75%+ of listings seek domain specialists with deep agent knowledge
- **Production Deployment**: Enterprises moving from single LLM calls to multi-agent workflows
- **Career Critical**: Agent framework proficiency (LangGraph, AutoGen, CrewAI) is now a **differentiator**
- **Revenue Impact**: Agentic systems enable automation of complex workflows (customer support, data analysis, software development)

### Pros & Cons

**Advantages:**
- **Autonomy**: Can accomplish multi-step tasks without constant human input
- **Tool Use**: Access external APIs, databases, search engines
- **Reasoning**: Plan and adjust strategies based on feedback
- **Scalability**: Handle complex workflows that would require teams of humans
- **Adaptability**: Generalize to new tasks with proper framework

**Limitations:**
- **Reliability**: Non-deterministic, can fail in unpredictable ways
- **Cost**: Multiple LLM calls per task (expensive at scale)
- **Latency**: Multi-step reasoning adds 5-30s latency
- **Control**: Difficult to constrain behavior completely
- **Safety**: Can take unintended actions if not properly controlled

**When to Use Agents vs Deterministic Systems:**
- **Agents**: Complex reasoning, ambiguous tasks, multi-step workflows
- **Deterministic**: Real-time <1s, safety-critical, well-defined rules

### Interview Questions

#### Question 13: Foundational - Function Calling & Tool Use
**Question:** "Explain function calling in LLMs. Implement an agent that can check weather, search Wikipedia, and send emails based on user requests."

**Comprehensive Answer:**

```python
from typing import List, Dict, Callable, Optional, Any
from dataclasses import dataclass
import json
import openai
from enum import Enum

# ===== TOOL DEFINITIONS =====

@dataclass
class Tool:
    """Tool that an agent can use"""
    name: str
    description: str
    parameters: Dict
    function: Callable

class ToolLibrary:
    """Collection of tools available to agent"""

    @staticmethod
    def get_weather(location: str, unit: str = "celsius") -> str:
        """
        Get current weather for a location

        Args:
            location: City name (e.g., "San Francisco, CA")
            unit: Temperature unit ("celsius" or "fahrenheit")

        Returns:
            Weather description
        """
        # In production: call actual weather API
        return f"The weather in {location} is 72°{unit[0].upper()}, partly cloudy."

    @staticmethod
    def search_wikipedia(query: str, sentences: int = 3) -> str:
        """
        Search Wikipedia and return summary

        Args:
            query: Search query
            sentences: Number of sentences to return

        Returns:
            Wikipedia summary
        """
        # In production: use wikipedia library or API
        import wikipedia
        try:
            result = wikipedia.summary(query, sentences=sentences)
            return result
        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"

    @staticmethod
    def send_email(to: str, subject: str, body: str) -> str:
        """
        Send an email

        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body

        Returns:
            Confirmation message
        """
        # In production: use SMTP or email service API
        print(f"[EMAIL] To: {to}, Subject: {subject}, Body: {body}")
        return f"Email sent to {to} with subject '{subject}'"

    @staticmethod
    def calculate(expression: str) -> str:
        """
        Evaluate mathematical expression

        Args:
            expression: Math expression (e.g., "2 + 2 * 3")

        Returns:
            Result
        """
        try:
            # Safe eval (in production: use safer alternative)
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

# ===== TOOL SCHEMA FOR OPENAI FUNCTION CALLING =====

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for information on a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "sentences": {
                        "type": "integer",
                        "description": "Number of sentences to return",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line"
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content"
                    }
                },
                "required": ["to", "subject", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# ===== AGENT IMPLEMENTATION =====

class FunctionCallingAgent:
    """
    Agent with function calling capabilities

    Architecture:
    1. User provides request
    2. LLM decides which tool(s) to call
    3. Agent executes tool calls
    4. LLM processes results
    5. Agent returns response or continues reasoning
    """

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.tools = ToolLibrary()

        # Map function names to actual functions
        self.function_map = {
            "get_weather": self.tools.get_weather,
            "search_wikipedia": self.tools.search_wikipedia,
            "send_email": self.tools.send_email,
            "calculate": self.tools.calculate
        }

        # Conversation history
        self.messages = []

    def run(self, user_request: str, max_iterations: int = 5) -> str:
        """
        Run agent with function calling

        Args:
            user_request: User's request
            max_iterations: Max number of LLM calls to prevent infinite loops

        Returns:
            Final response
        """
        # Initialize conversation
        self.messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant with access to various tools.
Use the available functions to answer user requests accurately.
Always use tools when needed rather than making up information."""
            },
            {
                "role": "user",
                "content": user_request
            }
        ]

        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")

            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto"  # Let model decide when to use tools
            )

            assistant_message = response.choices[0].message

            # Check if model wants to call function(s)
            if assistant_message.tool_calls:
                # Add assistant's function call to history
                self.messages.append(assistant_message)

                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    print(f"Calling: {function_name}({function_args})")

                    # Execute function
                    function_to_call = self.function_map[function_name]
                    function_result = function_to_call(**function_args)

                    print(f"Result: {function_result}")

                    # Add function result to messages
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": function_result
                    })

                # Continue loop to let model process results
                continue

            else:
                # No function call, model has final answer
                final_response = assistant_message.content
                print(f"\nFinal Response: {final_response}")
                return final_response

        # Max iterations reached
        return "I apologize, but I wasn't able to complete your request. Please try rephrasing."

    def run_streaming(self, user_request: str) -> None:
        """
        Streaming version for real-time feedback

        Shows thinking process to user
        """
        # Similar to run() but with streaming enabled
        pass

# ===== ADVANCED: PARALLEL FUNCTION CALLING =====

class ParallelFunctionAgent(FunctionCallingAgent):
    """
    Agent that can call multiple functions in parallel

    Example: "What's the weather in NYC and LA, and who invented the internet?"
    Calls: get_weather("NYC"), get_weather("LA"), search_wikipedia("internet inventor")
    """

    def run_parallel(self, user_request: str) -> str:
        """Execute multiple tool calls in parallel"""
        import concurrent.futures

        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_request}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto"
        )

        assistant_message = response.choices[0].message

        if assistant_message.tool_calls:
            self.messages.append(assistant_message)

            # Execute tool calls in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []

                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    function_to_call = self.function_map[function_name]

                    # Submit to thread pool
                    future = executor.submit(function_to_call, **function_args)
                    futures.append((tool_call, future))

                # Collect results
                for tool_call, future in futures:
                    result = future.result()

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": result
                    })

            # Get final response with all tool results
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages
            )

            return final_response.choices[0].message.content

        else:
            return assistant_message.content

# ===== PRODUCTION CONSIDERATIONS =====

class ProductionAgent(FunctionCallingAgent):
    """
    Production-ready agent with safety and monitoring

    Features:
    - Tool call approval (for sensitive operations)
    - Rate limiting
    - Logging and monitoring
    - Error handling
    - Cost tracking
    """

    def __init__(self, api_key: str, model: str = "gpt-4", require_approval: bool = True):
        super().__init__(api_key, model)
        self.require_approval = require_approval
        self.sensitive_tools = {"send_email"}  # Tools requiring approval

    def needs_approval(self, function_name: str) -> bool:
        """Check if tool call needs human approval"""
        return self.require_approval and function_name in self.sensitive_tools

    def get_approval(self, function_name: str, args: Dict) -> bool:
        """Get human approval for sensitive tool calls"""
        print(f"\n⚠️  APPROVAL REQUIRED")
        print(f"Function: {function_name}")
        print(f"Arguments: {json.dumps(args, indent=2)}")
        response = input("Approve this action? (yes/no): ")
        return response.lower() == 'yes'

    def run(self, user_request: str, max_iterations: int = 5) -> str:
        """Run agent with approval checks"""
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_request}
        ]

        for iteration in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto"
            )

            assistant_message = response.choices[0].message

            if assistant_message.tool_calls:
                self.messages.append(assistant_message)

                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    # Check if approval needed
                    if self.needs_approval(function_name):
                        if not self.get_approval(function_name, function_args):
                            # User rejected, inform model
                            self.messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": "Action rejected by user."
                            })
                            continue

                    # Execute function
                    try:
                        function_to_call = self.function_map[function_name]
                        result = function_to_call(**function_args)

                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": result
                        })

                    except Exception as e:
                        # Handle errors gracefully
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": f"Error: {str(e)}"
                        })

                continue

            else:
                return assistant_message.content

        return "Unable to complete request within iteration limit."

# ===== USAGE EXAMPLES =====
if __name__ == "__main__":
    # Initialize agent
    agent = FunctionCallingAgent(api_key="your-key")

    # Example 1: Weather query
    response = agent.run("What's the weather in San Francisco?")
    # Agent will call get_weather("San Francisco, CA")

    # Example 2: Multi-step reasoning
    response = agent.run(
        "Look up who invented the iPhone, then email me@example.com "
        "a summary of what you find."
    )
    # Agent will:
    # 1. Call search_wikipedia("iPhone inventor")
    # 2. Call send_email(to="me@example.com", subject="...", body="...")

    # Example 3: Parallel execution
    parallel_agent = ParallelFunctionAgent(api_key="your-key")
    response = parallel_agent.run_parallel(
        "What's the weather in NYC and LA? Also, who is the current president?"
    )
    # Calls get_weather("NYC"), get_weather("LA"), search_wikipedia("current president")
    # All in parallel

    # Example 4: Production agent with approval
    prod_agent = ProductionAgent(api_key="your-key", require_approval=True)
    response = prod_agent.run(
        "Send an email to boss@company.com saying I'll be late tomorrow"
    )
    # Will ask for approval before sending email
```

**Function Calling - Key Concepts:**

1. **Tool Definition**: Describe function with name, description, parameters (JSON Schema)
2. **LLM Decision**: Model decides when and which tool to call
3. **Argument Generation**: Model generates valid JSON arguments
4. **Execution**: Agent executes function with generated arguments
5. **Result Processing**: Model uses function result to continue reasoning

**Why Function Calling Matters:**
- **Extends LLM capabilities**: Access real-time data, APIs, databases
- **Grounding**: Reduces hallucinations by using real data
- **Actions**: Enables agents to take actions (send emails, book flights)
- **Reliability**: Structured JSON outputs (more reliable than text parsing)

**Production Best Practices:**
1. **Approval for Sensitive Tools**: Email, payments, deletions
2. **Rate Limiting**: Prevent runaway tool calls
3. **Error Handling**: Gracefully handle tool failures
4. **Logging**: Track all tool calls for debugging
5. **Timeout**: Limit execution time per tool
6. **Validation**: Validate tool arguments before execution

**Common Mistakes:**
- Not describing tools clearly (model won't call them)
- Too many tools (model gets confused, use <20)
- No error handling (tool failure crashes agent)
- Not validating tool outputs (garbage in, garbage out)
- Allowing sensitive operations without approval

**Excellence Indicators:**
- Explains parallel function calling
- Discusses tool call approval workflow
- Mentions error handling strategies
- Knows when to use function calling vs fine-tuning
- Can debug failed tool calls

---

[Due to length constraints, I'll create additional files for the remaining sections]
