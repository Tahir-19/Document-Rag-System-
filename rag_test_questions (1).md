# RAG Test Suite — Chatbot Pricing/Architecture Knowledge Base

Use these against the chatbot with `rag_knowledge_base.md` loaded as its knowledge source. Each question includes the expected answer and what it's designed to catch.

---

## A. Single-Fact Lookup (easy — tests basic retrieval)

**Q1.** What models are allowed on the $10 Small Business plan?
**Expected:** DeepSeek Chat (V3/V4) or Gemini Flash-Lite only.

**Q2.** What is the max output token limit for the $25 plan?
**Expected:** 600 tokens (~450 words).

**Q3.** How many rounds of chat history does the $50 plan retain?
**Expected:** Last 25 rounds of chat (or the full session).

**Q4.** What embedding models are recommended for vectorization?
**Expected:** text-embedding-3-small (Azure/OpenAI) or text-embedding-004 (Gemini).

**Q5.** What is the data ingestion limit for the $25 tier's knowledge base?
**Expected:** 10MB of documents (~5 million words), supporting basic PDFs and clean website scrapes.

---

## B. Numeric / Calculation Recall (medium — tests precision on numbers)

**Q6.** What is the profit margin range on the $10 plan?
**Expected:** 91%–94%.

**Q7.** Roughly how much does DeepSeek charge per 1M input tokens after the 90% cache discount is applied?
**Expected:** $0.014 per 1M input tokens (down from $0.14).

**Q8.** What is the Top-K chunk limit for the $50 tier, and roughly how many tokens does that inject?
**Expected:** Top-K = 8 chunks, ~4,000 tokens.

**Q9.** What character limit should the frontend enforce on user input for the $10 tier?
**Expected:** Max 2,000 characters.

**Q10.** At what point in a conversation should the backend start truncating the oldest messages?
**Expected:** Once a chat session reaches 15 back-and-forth messages.

---

## C. Multi-Hop / Reasoning (harder — requires combining 2+ facts)

**Q11.** If a client on the $25 plan switches from DeepSeek to Azure OpenAI mid-chat, how does that affect their credit consumption?
**Expected:** Each message costs more credits — DeepSeek messages cost 1 credit, while Azure OpenAI messages cost 5 credits, so switching burns through their credit wallet faster.

**Q12.** Why is Azure OpenAI or flagship Gemini Pro banned from the $10 tier, based on the cost math given?
**Expected:** Those models are far more expensive per token; given the $10 tier's thin absolute margin (~$0.60–$0.90 wholesale cost), a single long conversation on a premium model could wipe out the entire month's revenue from that client.

**Q13.** A $10-tier account is misconfigured with Top-K = 20 instead of Top-K = 2. What is the approximate consequence?
**Expected:** Each user message would inject ~10,000 tokens of company documents (10x the intended ~1,000 tokens), which the document says would instantly destroy the plan's margin.

**Q14.** Explain the difference between the "Core Memory" and "Asynchronous Summary" components of the conversation history system.
**Expected:** Core Memory is a sliding window that passes only the last N messages verbatim to the LLM (e.g., last 5 for the $10 plan). The Asynchronous Summary is a background process using the cheapest model (DeepSeek Flash) that compresses older/dropped messages into a short summary once the tier's history limit is exceeded.

**Q15.** Why does the document say vector embedding generation should not be charged to customers?
**Expected:** Because embedding models cost roughly $0.02 per 1 million tokens — a negligible, "near zero" backend cost, e.g., vectorizing a full FAQ page costs a fraction of a penny — so it's treated as a free feature.

---

## D. Structural / Ordering Recall (tests whether retrieval preserves structure, not just keywords)

**Q16.** List the five components of the standard API payload structure, in order.
**Expected:** 1) System Prompt, 2) Knowledge Base Context, 3) Conversation Summary, 4) Active Chat Window, 5) Latest User Question.

**Q17.** What are the three "rules" for keeping the pricing model profitable?
**Expected:** Rule #1: Use a Credit Wallet on the frontend. Rule #2: Enforce strict context caching. Rule #3: Cap max tokens per turn (truncate history after 15 rounds).

---

## E. Negative / Out-of-Scope Tests (tests hallucination resistance — nothing in the KB answers these)

**Q18.** What vector database is recommended — Pinecone, Qdrant, or pgvector?
**Expected:** The document mentions all three only as examples of vector database types (Pinecone, Qdrant, pgvector) without recommending one specifically. A good RAG answer should say no single one is recommended, rather than picking one authoritatively.

**Q19.** What is the refund policy if a customer exceeds their monthly credit limit?
**Expected:** Not covered in the document — the chatbot should say this information isn't available rather than inventing a policy.

**Q20.** What programming language or framework should the backend be built in?
**Expected:** Not specified anywhere in the document — correct answer is "not covered," not a guess like Python/Node.js.

**Q21.** Does the $50 plan support real-time voice or phone support?
**Expected:** Not mentioned in the document. Correct behavior is to state this isn't addressed, not to fabricate a feature.

---

## F. Comparison Questions (tests cross-section retrieval across the tier table)

**Q22.** Compare the max input window size across all three tiers.
**Expected:** $10 = 8,000 tokens (~6,000 words); $25 = 24,000 tokens (~18,000 words); $50 = 64,000 tokens (~48,000 words).

**Q23.** How does the knowledge base ingestion limit scale from the $10 to $50 tier?
**Expected:** $10 = 1MB (~500,000 words, text only); $25 = 10MB (~5 million words, basic PDFs/scrapes); $50 = 50MB (complex PDFs, tables, multi-page crawling).

---

## Suggested Scoring Rubric
- **Exact match (A/B questions):** Full credit only if the specific number/fact is correct.
- **Reasoning questions (C):** Credit if the answer captures the causal logic, even with different phrasing.
- **Structural (D):** Credit only if order/completeness is preserved (partial lists = partial credit).
- **Negative tests (E):** Full credit only if the bot declines to fabricate; any invented specific detail = fail, regardless of how plausible it sounds.
- **Comparison (F):** Credit only if all three tiers are correctly represented, not just one or two.
