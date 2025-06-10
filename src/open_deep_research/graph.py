from dotenv import load_dotenv
from langchain.globals import set_debug

load_dotenv()
set_debug(True)

import functools
import operator
import os
from typing import Annotated, Dict, List, Literal, Optional, TypedDict

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langgraph.graph import END, MessagesState, StateGraph
from pydantic import BaseModel, Field

from open_deep_research.configuration import Configuration
from open_deep_research.prompts import (
    ACTIONABLE_CHECKLIST_INSTRUCTIONS,
    CONFERENCE_FORMATS,
    EXECUTIVE_SUMMARY_INSTRUCTIONS,
    FEEDBACK_CRITERIA,
    FEEDBACK_INSTRUCTIONS_TEMPLATE,
    HOW_TO_WRITE_GOOD_REVIEWS,
    PAPER_SUMMARY_INSTRUCTIONS,
    RELATED_WORK_SIMILARITY_EXPLANATION_INSTRUCTIONS,
    REVIEW_EXAMPLES,
    REVIEW_SIMULATION_INSTRUCTIONS,
    SEARCH_KEYWORD_GENERATION_INSTRUCTIONS,
)
from open_deep_research.utils import semantic_scholar_search

# --- Pydantic Models for Structured Output ---


class DeepSummary(BaseModel):
    """A deep summary of an academic manuscript."""

    purpose_objective: str = Field(description="The main goal or research question.")
    methods: str = Field(description="Summary of the approach, methodology, or experimental design.")
    experimental_setup: str = Field(description="Details on models, algorithms, datasets, benchmarks, etc.")
    key_findings_results: str = Field(description="The most important findings or results.")
    conclusions_implications: str = Field(description="Principal conclusions and broader implications.")
    novelty_contribution: str = Field(description="The paper's originality and unique contribution.")
    limitations: str = Field(description="Acknowledged limitations or areas for future research.")


class SearchQueries(BaseModel):
    """A list of search queries for finding related work."""

    queries: List[str] = Field(description="A list of keyword-based search query strings.")


class SimilarityExplanation(BaseModel):
    """The explanation for why a related work is similar to the user's manuscript."""

    explanation: str = Field(description="A concise paragraph explaining the semantic similarity between two papers.")


# --- State Definitions ---


class PaperReviewState(MessagesState):
    """
    The global state for the TeachMe agent workflow. It manages all data
    passed between the different stages of the review process.
    """

    # 'messages' is inherited from MessagesState for conversation history.

    # --- Input Fields ---
    input_path_or_text: str

    # --- Stage 1: Summary ---
    original_paper_text: str
    original_paper_summary: DeepSummary

    # --- Stage 2: Search ---
    final_related_works: Optional[List[dict]]

    # --- Stage 3: Feedback ---
    clarity_feedback: Optional[str]
    novelty_feedback: Optional[str]
    methodology_feedback: Optional[str]
    technical_feedback: Optional[str]
    limitations_feedback: Optional[str]

    # --- Stage 4: Review & Report ---
    simulated_review: Optional[str]
    executive_summary: Optional[str]
    actionable_checklist: Optional[str]
    final_report: Optional[str]

    # --- Configuration (snapshot) ---
    config: Configuration


# --- Agent Nodes ---


# === STAGE 1: SUMMARY AGENT ===
async def summary_agent_node(state: PaperReviewState, config: RunnableConfig) -> Dict:
    """
    Parses the input paper (PDF or text) and generates a deep summary.
    This is the entry point of the workflow.
    """
    print("--- STAGE 1: SUMMARIZING PAPER ---")
    cfg = Configuration.from_runnable_config(config)

    if not state["messages"] or not isinstance(state["messages"][-1], HumanMessage):
        raise ValueError("The conversation must start with a user message containing the paper path or text.")

    input_val = state["messages"][-1].content
    print(f"Received input: {input_val[:100]}...")

    # 1. Parse Document
    if os.path.exists(input_val) and input_val.lower().endswith(".pdf"):
        loader = PyMuPDF4LLMLoader(input_val)
        docs = [doc async for doc in loader.alazy_load()]
        text = "\n\n".join([doc.page_content for doc in docs])
    else:
        text = input_val

    # 2. Summarize using LLM
    llm = init_chat_model(model=cfg.summary_agent_model, model_kwargs=cfg.summary_agent_model_kwargs or {})
    summarizer_chain = llm.with_structured_output(DeepSummary)
    summary = await summarizer_chain.ainvoke(PAPER_SUMMARY_INSTRUCTIONS.format(paper_text=text))

    return {
        "input_path_or_text": input_val,
        "original_paper_text": text,
        "original_paper_summary": summary,
        "config": cfg,  # Pass config to state for later nodes
    }


# === STAGE 2: SEARCH AGENT (Implemented as a Sub-Graph) ===


class SearchState(TypedDict):
    """State for the related works search sub-graph."""

    original_paper_summary: DeepSummary
    config: Configuration
    search_queries: List[str]
    retrieved_works: List[dict]
    final_related_works: List[dict]


async def generate_keywords_node(state: SearchState) -> Dict:
    """Generates search keywords based on the paper summary."""
    print("--- STAGE 2.1: GENERATING SEARCH KEYWORDS ---")
    cfg = state["config"]
    llm = init_chat_model(model=cfg.search_agent_model, model_kwargs=cfg.search_agent_model_kwargs or {})
    keyword_generator = llm.with_structured_output(SearchQueries)

    prompt = SEARCH_KEYWORD_GENERATION_INSTRUCTIONS.format(original_summary=state["original_paper_summary"].dict(), max_search_queries=cfg.max_search_queries)
    queries = await keyword_generator.ainvoke(prompt)
    return {"search_queries": queries.queries}


async def retrieve_works_node(state: SearchState) -> Dict:
    """Searches Semantic Scholar and deduplicates the results."""
    print("--- STAGE 2.2: RETRIEVING RELATED WORKS ---")
    cfg = state["config"]
    search_results = await semantic_scholar_search(search_queries=state["search_queries"], max_results_per_query=cfg.max_results_per_query)

    # Deduplicate papers based on paperId
    unique_papers = {}
    for result_set in search_results:
        for paper in result_set.get("results", []):
            if paper and paper.get("paperId") and paper.get("abstract"):  # Ensure paper has ID and abstract
                unique_papers[paper["paperId"]] = paper

    print(f"Found {len(unique_papers)} unique papers.")
    return {"retrieved_works": list(unique_papers.values())}


async def rank_and_filter_node(state: SearchState) -> Dict:
    """Ranks retrieved works using RAG and selects the top-k."""
    print("--- STAGE 2.3: RANKING & FILTERING WORKS VIA RAG ---")
    cfg = state["config"]
    papers = state["retrieved_works"]
    if not papers:
        return {"final_related_works": []}

    embeddings = init_embeddings(cfg.search_agent_embedding_model)

    # Use paper abstracts for embedding
    texts = [paper["abstract"] for paper in papers]
    metadatas = papers

    # Create a FAISS vector store in memory for ranking
    vectorstore = await FAISS.afrom_texts(texts, embedding=embeddings, metadatas=metadatas)

    # Query with the original paper's summary
    query_text = state["original_paper_summary"].json()
    retriever = vectorstore.as_retriever(search_kwargs={"k": cfg.max_final_related_works})

    top_k_docs = await retriever.ainvoke(query_text)

    # Extract the full paper data from the retrieved documents
    top_k_papers = [doc.metadata for doc in top_k_docs]
    print(f"Selected top {len(top_k_papers)} papers for explanation.")
    return {"final_related_works": top_k_papers}


async def explain_similarity_node(state: SearchState) -> Dict:
    """Generates a similarity explanation for each top-k paper in parallel."""
    print("--- STAGE 2.4: GENERATING SIMILARITY EXPLANATIONS ---")
    cfg = state["config"]
    top_k_papers = state["final_related_works"]
    if not top_k_papers:
        return {"final_related_works": []}

    llm = init_chat_model(model=cfg.search_agent_model, model_kwargs=cfg.search_agent_model_kwargs or {})
    explanation_prompt = ChatPromptTemplate.from_template(RELATED_WORK_SIMILARITY_EXPLANATION_INSTRUCTIONS)
    explanation_chain = explanation_prompt | llm

    # Prepare batch of inputs for parallel processing
    batch_inputs = [
        {
            "original_summary": state["original_paper_summary"].dict(),
            "related_work_title": paper.get("title", "N/A"),
            "related_work_abstract": paper.get("abstract", "N/A"),
        }
        for paper in top_k_papers
    ]

    explanations = await explanation_chain.abatch(batch_inputs)

    # Add the generated explanation to each paper's data
    for i, paper in enumerate(top_k_papers):
        paper["similarity_explanation"] = explanations[i].content

    return {"final_related_works": top_k_papers}


# === STAGE 3: FEEDBACK AGENT ===
async def feedback_agent_node(state: PaperReviewState, feedback_type: str) -> Dict:
    """Generic feedback agent that generates feedback based on a specific criterion."""
    print(f"--- STAGE 3: GENERATING '{feedback_type.upper()}' FEEDBACK ---")
    cfg = state["config"]
    model_name = cfg.feedback_agent_models.get(feedback_type, cfg.feedback_agent_default_model)
    llm = init_chat_model(model=model_name, model_kwargs=cfg.feedback_agent_model_kwargs or {})

    prompt = FEEDBACK_INSTRUCTIONS_TEMPLATE.format(feedback_criteria=FEEDBACK_CRITERIA[feedback_type], paper_text=state["original_paper_text"])
    feedback = await llm.ainvoke(prompt)

    # e.g., 'clarity_and_organization' -> 'clarity_feedback'
    output_key = f"{feedback_type.split('_')[0]}_feedback"
    return {output_key: feedback.content}


# === STAGE 4: REVIEW & REPORTING AGENTS ===


async def review_agent_node(state: PaperReviewState) -> Dict:
    """Simulates a peer review after all feedback and search is complete."""
    print("--- STAGE 4.1: SIMULATING PEER REVIEW ---")
    cfg = state["config"]
    llm = init_chat_model(model=cfg.review_agent_model, model_kwargs=cfg.review_agent_model_kwargs or {})

    aggregated_feedback = "\n\n".join(
        f"### {k.replace('_', ' ').title()}\n{v}"
        for k, v in [
            ("Clarity and Organization", state["clarity_feedback"]),
            ("Motivation and Novelty", state["novelty_feedback"]),
            ("Methodology and Evidence", state["methodology_feedback"]),
            ("Technical and Language Quality", state["technical_feedback"]),
            ("Limitations and Future Work", state["limitations_feedback"]),
        ]
        if v
    )
    formatted_related_works = "\n".join([f"- **{p.get('title', 'N/A')}** ({p.get('year', 'N/A')}, {p.get('venue', 'N/A')}): {p.get('similarity_explanation', '')}" for p in state.get("final_related_works", [])])

    selected_format = CONFERENCE_FORMATS.get(cfg.review_conference_format, CONFERENCE_FORMATS["neurips"])
    prompt = REVIEW_SIMULATION_INSTRUCTIONS.format(
        how_to_write_good_reviews=HOW_TO_WRITE_GOOD_REVIEWS,
        review_examples=REVIEW_EXAMPLES,
        original_summary=state["original_paper_summary"].dict(),
        final_related_works=formatted_related_works,
        aggregated_feedback=aggregated_feedback,
        review_format=selected_format,
    )
    review = await llm.ainvoke(prompt)
    return {"simulated_review": review.content}


async def executive_summary_node(state: PaperReviewState) -> Dict:
    """Generates the executive summary for the final report."""
    print("--- STAGE 4.2: GENERATING EXECUTIVE SUMMARY ---")
    cfg = state["config"]
    llm = init_chat_model(model=cfg.report_enhancement_agent_model, model_kwargs=cfg.report_enhancement_agent_model_kwargs or {})

    # Reuse formatted strings from the review node
    aggregated_feedback = "\n\n".join(f"- {k.replace('_feedback','').replace('_', ' ').title()}: {v}" for k, v in state.items() if k.endswith("_feedback") and v)
    formatted_related_works = "\n".join([f"- **{p.get('title', 'N/A')}**: {p.get('similarity_explanation', '')}" for p in state.get("final_related_works", [])])

    prompt = EXECUTIVE_SUMMARY_INSTRUCTIONS.format(
        original_summary=state["original_paper_summary"].dict(),
        final_related_works=formatted_related_works,
        aggregated_feedback=aggregated_feedback,
        simulated_review=state["simulated_review"],
    )
    summary = await llm.ainvoke(prompt)
    return {"executive_summary": summary.content}


async def actionable_checklist_node(state: PaperReviewState) -> Dict:
    """Generates the actionable checklist for the authors."""
    print("--- STAGE 4.3: GENERATING ACTIONABLE CHECKLIST ---")
    cfg = state["config"]
    llm = init_chat_model(model=cfg.report_enhancement_agent_model, model_kwargs=cfg.report_enhancement_agent_model_kwargs or {})

    aggregated_feedback = "\n\n".join(f"- {k.replace('_feedback','').replace('_', ' ').title()}: {v}" for k, v in state.items() if k.endswith("_feedback") and v)
    formatted_related_works = "\n".join([f"- **{p.get('title', 'N/A')}**: {p.get('similarity_explanation', '')}" for p in state.get("final_related_works", [])])

    prompt = ACTIONABLE_CHECKLIST_INSTRUCTIONS.format(final_related_works=formatted_related_works, aggregated_feedback=aggregated_feedback, simulated_review=state["simulated_review"])
    checklist = await llm.ainvoke(prompt)
    return {"actionable_checklist": checklist.content}


def compile_final_report_node(state: PaperReviewState) -> Dict:
    """Compiles all generated content into a single, formatted final report."""
    print("--- STAGE 4.4: ASSEMBLING FINAL REPORT ---")
    summary = state["original_paper_summary"]

    # Section 1: Deep Summary
    summary_section = f"### Section 1.1 Purpose and Objective\n\n{summary.purpose_objective}\n\n" f"### Section 1.2 Methods\n\n{summary.methods}\n\n" f"### Section 1.3 Experimental Setup\n\n{summary.experimental_setup}\n\n" f"### Section 1.4 Key Findings and Results\n\n{summary.key_findings_results}\n\n" f"### Section 1.5 Conclusions and Implications\n\n{summary.conclusions_implications}\n\n" f"### Section 1.6 Novelty and Contribution\n\n{summary.novelty_contribution}\n\n" f"### Section 1.7 Limitations\n\n{summary.limitations}"

    # Section 2: Related Works
    related_works_section = "No relevant related works were found."
    if state.get("final_related_works"):
        related_works_list = []
        for p in state["final_related_works"]:
            pdf_link = f"[PDF]({p['openAccessPdf']['url']})" if p.get("openAccessPdf") else "No PDF"
            related_works_list.append(f"#### [{p.get('title', 'N/A')}]({p.get('url', '#')})\n" f"**Venue:** {p.get('venue', 'N/A')} ({p.get('year', 'N/A')}) | " f"**Citations:** {p.get('citationCount', 0)} | {pdf_link}\n\n" f"**Similarity Explanation:**\n{p.get('similarity_explanation', 'N/A')}")
        related_works_section = "\n\n---\n\n".join(related_works_list)

    # Section 3: Detailed Feedback
    feedback_section = f"### Section 3.1 Clarity and Organization\n\n{state['clarity_feedback']}\n\n" f"### Section 3.2 Motivation, Novelty & Significance\n\n{state['novelty_feedback']}\n\n" f"### Section 3.3 Methodology & Evidence\n\n{state['methodology_feedback']}\n\n" f"### Section 3.4 Technical Accuracy & Language Quality\n\n{state['technical_feedback']}\n\n" f"### Section 3.5 Limitations & Future Work\n\n{state['limitations_feedback']}"

    # Assemble the final report
    final_report_str = f"""
## Section 1. Deep Summary of Manuscript
{summary_section.strip()}

---

## Section 2. Semantic Related Works Recommendation
{related_works_section.strip()}

---

## Section 3. Detailed Feedback by Criteria
{feedback_section.strip()}

---

## Section 4. Simulated Peer Review
{state['simulated_review'].strip()}

---

## Section 5. Executive Report & Action Plan

### Section 5.1 Executive Summary
{state['executive_summary'].strip()}

### Section 5.2 Actionable Enhancement Checklist
{state['actionable_checklist'].strip()}
""".strip()

    return {"final_report": final_report_str, "messages": [AIMessage(content=final_report_str)]}


# --- Graph Definition ---

# 1. Define the Search Sub-Graph
search_workflow = StateGraph(SearchState, config_schema=Configuration)
search_workflow.add_node("generate_keywords", generate_keywords_node)
search_workflow.add_node("retrieve_works", retrieve_works_node)
search_workflow.add_node("rank_and_filter", rank_and_filter_node)
search_workflow.add_node("explain_similarity", explain_similarity_node)

search_workflow.set_entry_point("generate_keywords")
search_workflow.add_edge("generate_keywords", "retrieve_works")
search_workflow.add_edge("retrieve_works", "rank_and_filter")
search_workflow.add_edge("rank_and_filter", "explain_similarity")
search_workflow.add_edge("explain_similarity", END)
search_graph = search_workflow.compile()


# This function adapts the main state to the sub-graph's input state
def prepare_search_graph_input(state: PaperReviewState) -> SearchState:
    """Prepares the input for the search sub-graph."""
    return {
        "original_paper_summary": state["original_paper_summary"],
        "config": state["config"],
    }


# 2. Define the Main Workflow
workflow = StateGraph(PaperReviewState, input=MessagesState, output=MessagesState, config_schema=Configuration)

# Add all nodes
workflow.add_node("summary_agent", summary_agent_node)

# Wrap the search sub-graph in a lambda to handle state mapping
search_agent_node = RunnableLambda(prepare_search_graph_input) | search_graph
workflow.add_node("search_agent", search_agent_node)

# Use functools.partial to create specific feedback nodes from the generic function
feedback_nodes = {
    "clarity_and_organization": functools.partial(feedback_agent_node, feedback_type="clarity_and_organization"),
    "novelty_and_motivation": functools.partial(feedback_agent_node, feedback_type="novelty_and_motivation"),
    "methodology_and_evidence": functools.partial(feedback_agent_node, feedback_type="methodology_and_evidence"),
    "technical_and_language_quality": functools.partial(feedback_agent_node, feedback_type="technical_and_language_quality"),
    "limitations_and_future_work": functools.partial(feedback_agent_node, feedback_type="limitations_and_future_work"),
}
for name, node in feedback_nodes.items():
    workflow.add_node(name, node)

workflow.add_node("review_agent", review_agent_node)
workflow.add_node("executive_summary_agent", executive_summary_node)
workflow.add_node("actionable_checklist_agent", actionable_checklist_node)
workflow.add_node("compile_report_agent", compile_final_report_node)

# 3. Define the Workflow Edges
workflow.set_entry_point("summary_agent")

# After summary, start search and all feedback tasks in parallel
workflow.add_edge("summary_agent", "search_agent")
for name in feedback_nodes:
    workflow.add_edge("summary_agent", name)

# Join the parallel branches: after search AND all feedback are done, simulate the review
# LangGraph automatically waits for all incoming edges to complete before running a node.
workflow.add_edge("search_agent", "review_agent")
for name in feedback_nodes:
    workflow.add_edge(name, "review_agent")

# After review, generate executive summary and checklist in parallel
workflow.add_edge("review_agent", "executive_summary_agent")
workflow.add_edge("review_agent", "actionable_checklist_agent")

# After both are generated, compile the final report
workflow.add_edge("executive_summary_agent", "compile_report_agent")
workflow.add_edge("actionable_checklist_agent", "compile_report_agent")

workflow.add_edge("compile_report_agent", END)

# Compile the final graph
graph = workflow.compile()
