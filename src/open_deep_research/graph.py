from langchain.globals import set_debug
from dotenv import load_dotenv

load_dotenv()
set_debug(True)

import functools
import operator
import os
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langgraph.graph import END, MessagesState, StateGraph
from pydantic import BaseModel, Field
from llama_cloud_services import LlamaParse
from open_deep_research.configuration import Configuration
from open_deep_research.prompts import (
    ACTIONABLE_CHECKLIST_INSTRUCTIONS,
    CONFERENCE_FORMATS,
    DIFFERENTIATION_ANALYSIS_INSTRUCTIONS,
    EXECUTIVE_SUMMARY_INSTRUCTIONS,
    FEEDBACK_CRITERIA,
    FEEDBACK_INSTRUCTIONS_TEMPLATE,
    HOW_TO_WRITE_GOOD_REVIEWS,
    PAPER_SUMMARY_INSTRUCTIONS,
    RELATED_WORK_RELEVANCE_CHECK_INSTRUCTIONS,
    REVIEW_EXAMPLES,
    REVIEW_SIMULATION_INSTRUCTIONS,
    SEARCH_KEYWORD_GENERATION_INSTRUCTIONS,
)
from open_deep_research.utils import duckduckgo_search


# --- Pydantic Models for Structured Output (이전과 동일) ---
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
    """List of search queries to find related work."""

    queries: List[str] = Field(description="A list of keyword-based search queries.")


class RelevanceDecision(BaseModel):
    """Decision on the relevance of a candidate paper."""

    is_relevant: Literal["yes", "no"] = Field(description="Decision on whether the paper is relevant.")
    justification: str = Field(description="Brief justification for the relevance decision.")


class RelatedWork(BaseModel):
    """Represents a single piece of related work."""

    title: str
    url: str
    abstract: str
    full_text: Optional[str] = None
    deep_summary: Optional[DeepSummary] = None


# --- State Management (NEW: Executive Summary & Checklist added) ---
class SearchAgentState(TypedDict):
    """State for the search agent subgraph."""

    original_summary: DeepSummary
    target_conferences: List[str]
    queries: List[str]
    candidate_works: Annotated[List[RelatedWork], operator.add]
    confirmed_works: Annotated[List[RelatedWork], operator.add]
    search_iteration: int


class PaperReviewState(MessagesState):
    """
    The global state for the TeachMe agent workflow, inheriting from MessagesState
    to manage conversation history.
    """

    # 'messages' is inherited from MessagesState

    # Inputs (will be populated by the first node)
    input_path_or_text: str
    review_conference_format: Literal["neurips", "iclr", "icml", "acl"]

    # Stage 1: Summary
    original_paper_text: str
    original_paper_summary: DeepSummary

    # Stage 2: Search
    differentiation_analysis: str

    # Stage 3: Feedback
    clarity_feedback: str
    novelty_feedback: str
    methodology_feedback: str
    technical_feedback: str
    limitations_feedback: str

    # Stage 4: Review & Report Generation
    simulated_review: str
    executive_summary: str
    actionable_checklist: str
    final_report: str


# --- Agent Nodes (Update init_chat_model calls) ---
# Stage 1 Node
async def summary_agent_node(state: PaperReviewState, config: RunnableConfig):
    """Parses input paper from the user's message and creates a deep summary."""
    print("--- STAGE 1: SUMMARIZING PAPER ---")
    cfg = Configuration.from_runnable_config(config)

    # Extract the user's input from the last message in the state
    # This makes the graph invokable with a simple chat history
    if not state["messages"] or not isinstance(state["messages"][-1], HumanMessage):
        raise ValueError("The conversation history must end with a user message containing the input.")

    input_val = state["messages"][-1].content
    print(f"Received input: {input_val[:100]}...")

    # 1. Parse Document
    if os.path.exists(input_val) and input_val.lower().endswith(".pdf"):
        loader = PyMuPDF4LLMLoader(input_val)
        docs = []
        async for doc in loader.alazy_load():
            docs.append(doc)
        text = "\n\n".join([doc.page_content for doc in docs])
        # parser = LlamaParse(
        #     api_key=os.environ["LLAMA_CLOUD_API_KEY"],  # can also be set in your env as LLAMA_CLOUD_API_KEY
        #     num_workers=4,  # if multiple files passed, split in `num_workers` API calls
        # )
        # result = await parser.aparse(input_val)
        # text = result.get_markdown_documents(split_by_page=True)
        # async with aiofiles.open("example.txt", "w", encoding="utf-8") as f:
        #     await f.write(text)
    else:
        text = input_val  # Assume raw text

    # 2. Summarize using LLM
    llm = init_chat_model(
        model=cfg.summary_agent_model,
        model_kwargs=cfg.summary_agent_model_kwargs or {},
    )
    summarizer_chain = llm.with_structured_output(DeepSummary)
    summary = await summarizer_chain.ainvoke(PAPER_SUMMARY_INSTRUCTIONS.format(paper_text=text))

    # Populate the state for subsequent nodes
    return {"input_path_or_text": input_val, "review_conference_format": cfg.review_conference_format, "original_paper_text": text, "original_paper_summary": summary, "differentiation_analysis": None}  # Also populate this from config


# Stage 3 Nodes (Parallel Execution)
async def feedback_agent_node(state: TypedDict, config: RunnableConfig, feedback_type: str):
    """
    Generic feedback agent node that dynamically selects a model based on feedback_type.
    """
    print(f"--- STAGE 3: GENERATING '{feedback_type.upper()}' FEEDBACK ---")
    cfg = Configuration.from_runnable_config(config)

    # Dynamically get the model for the given feedback type from the config dictionary.
    # The fallback value now comes from the configuration file instead of being hardcoded.
    model_name = cfg.feedback_agent_models.get(feedback_type, cfg.feedback_agent_default_model)

    llm = init_chat_model(
        model=model_name,
        model_kwargs=cfg.feedback_agent_model_kwargs or {},
    )

    prompt = FEEDBACK_INSTRUCTIONS_TEMPLATE.format(feedback_criteria=FEEDBACK_CRITERIA[feedback_type], paper_text=state["original_paper_text"])

    feedback = await llm.ainvoke(prompt)

    output_key = f"{feedback_type.split('_')[0]}_feedback"
    return {output_key: feedback.content}


# --- Search Agent Subgraph (Update init_chat_model calls) ---
async def generate_keywords_node(state: SearchAgentState, config: RunnableConfig):
    print("--- SEARCH SUBGRAPH: GENERATING KEYWORDS ---")
    cfg = Configuration.from_runnable_config(config)
    llm = init_chat_model(model=cfg.search_agent_model, model_kwargs=cfg.search_agent_model_kwargs or {})
    # ... rest of the function is the same ...
    query_gen_chain = llm.with_structured_output(SearchQueries)
    queries = await query_gen_chain.ainvoke(SEARCH_KEYWORD_GENERATION_INSTRUCTIONS.format(number_of_queries=cfg.max_search_queries_per_topic, deep_summary=state["original_summary"].dict(), target_conferences=", ".join(state["target_conferences"])))
    return {"queries": queries.queries}


async def web_search_node(state: SearchAgentState, config: RunnableConfig):
    # ... same as before ...
    print("--- SEARCH SUBGRAPH: PERFORMING WEB SEARCH ---")
    results = []
    if state.get("search_iteration", 0) > 0:
        return {"candidate_works": []}
    for query in state["queries"]:
        search_results_str = await duckduckgo_search.ainvoke(query, config)
        for res in search_results_str.split("\n\n"):
            try:
                title_part = res.split("URL: ")[0].split("] ")[1]
                url_part = res.split("URL: ")[1].split("\n")[0]
                results.append(RelatedWork(title=title_part.strip(), url=url_part.strip(), abstract=""))
            except IndexError:
                continue
    unique_works = {work.url: work for work in results}.values()
    return {"candidate_works": list(unique_works)}


async def relevance_check_node(state: SearchAgentState, config: RunnableConfig):
    print("--- SEARCH SUBGRAPH: CHECKING RELEVANCE ---")
    cfg = Configuration.from_runnable_config(config)
    llm = init_chat_model(model=cfg.search_agent_model, model_kwargs=cfg.search_agent_model_kwargs or {})
    # ... rest of the function is the same ...
    relevance_checker = llm.with_structured_output(RelevanceDecision)
    confirmed_works = []
    for work in state["candidate_works"]:
        if len(confirmed_works) >= cfg.max_related_works:
            break
        try:
            abstract_search = await duckduckgo_search.ainvoke(f"{work.title} abstract arxiv", config)
            work.abstract = abstract_search
            decision = await relevance_checker.ainvoke(
                RELATED_WORK_RELEVANCE_CHECK_INSTRUCTIONS.format(
                    original_summary=state["original_summary"].dict(),
                    candidate_title=work.title,
                    candidate_abstract=work.abstract,
                )
            )
            if decision.is_relevant == "yes":
                confirmed_works.append(work)
        except Exception as e:
            print(f"Could not process {work.title}: {e}")
            continue
    return {"confirmed_works": confirmed_works, "search_iteration": state.get("search_iteration", 0) + 1}


def should_continue_search(state: SearchAgentState, config: RunnableConfig):
    # ... same as before ...
    cfg = Configuration.from_runnable_config(config)
    if len(state["confirmed_works"]) >= cfg.max_related_works or state.get("search_iteration", 0) >= cfg.max_search_iterations:
        return "end"
    return "continue"


async def differentiation_node(state: SearchAgentState, config: RunnableConfig):
    print("--- SEARCH SUBGRAPH: PERFORMING DIFFERENTIATION ANALYSIS ---")
    if not state["confirmed_works"]:
        return {"differentiation_analysis": "No relevant related works were found to perform a differentiation analysis."}
    cfg = Configuration.from_runnable_config(config)
    llm = init_chat_model(model=cfg.search_agent_model, model_kwargs=cfg.search_agent_model_kwargs or {})
    # ... rest of the function is the same ...
    related_summaries_str = "\n\n---\n\n".join([f"Title: {w.title}\nAbstract: {w.abstract}" for w in state["confirmed_works"]])
    analysis = await llm.ainvoke(
        DIFFERENTIATION_ANALYSIS_INSTRUCTIONS.format(
            original_summary=state["original_summary"].dict(),
            related_summaries=related_summaries_str,
        )
    )
    return {"differentiation_analysis": analysis.content}


# Build Search Subgraph and Entry Node (same as before)
search_builder = StateGraph(SearchAgentState, config_schema=Configuration)
# ... add nodes and edges ...
search_builder.add_node("generate_keywords", generate_keywords_node)
search_builder.add_node("web_search", web_search_node)
search_builder.add_node("relevance_check", relevance_check_node)
search_builder.add_node("differentiate", differentiation_node)
search_builder.set_entry_point("generate_keywords")
search_builder.add_edge("generate_keywords", "web_search")
search_builder.add_edge("web_search", "relevance_check")
search_builder.add_conditional_edges("relevance_check", should_continue_search, {"continue": "web_search", "end": "differentiate"})
search_builder.add_edge("differentiate", END)
search_agent_graph = search_builder.compile()


async def search_agent_entry_node(state: PaperReviewState, config: RunnableConfig):
    # ... same as before ...
    print("--- ENTERING SEARCH AGENT SUBGRAPH ---")
    cfg = Configuration.from_runnable_config(config)
    subgraph_state = SearchAgentState(original_summary=state["original_paper_summary"], target_conferences=cfg.target_conferences, queries=[], candidate_works=[], confirmed_works=[], search_iteration=0)
    final_subgraph_state = await search_agent_graph.ainvoke(subgraph_state, config)
    return {"differentiation_analysis": final_subgraph_state["differentiation_analysis"]}


# --- NEW/MODIFIED STAGE 4 NODES ---


async def review_agent_node(state: PaperReviewState, config: RunnableConfig):
    """Simulates a peer review AFTER all feedback and search is complete."""
    print("--- STAGE 4.1: SIMULATING PEER REVIEW ---")
    cfg = Configuration.from_runnable_config(config)
    llm = init_chat_model(
        model=cfg.review_agent_model,
        model_kwargs=cfg.review_agent_model_kwargs or {},
    )

    # Aggregate feedback for the prompt
    aggregated_feedback = "\n\n".join(
        [
            f"### Clarity and Organization\n{state['clarity_feedback']}",
            f"### Motivation and Novelty\n{state['novelty_feedback']}",
            f"### Methodology and Evidence\n{state['methodology_feedback']}",
            f"### Technical and Language Quality\n{state['technical_feedback']}",
            f"### Limitations and Future Work\n{state['limitations_feedback']}",
        ]
    )

    # Dynamically select the review format based on configuration
    selected_format = CONFERENCE_FORMATS.get(cfg.review_conference_format, CONFERENCE_FORMATS["neurips"])

    # differentiation_analysis
    prompt = REVIEW_SIMULATION_INSTRUCTIONS.format(how_to_write_good_reviews=HOW_TO_WRITE_GOOD_REVIEWS, review_examples=REVIEW_EXAMPLES, original_summary=state["original_paper_summary"].dict(), differentiation_analysis=state["differentiation_analysis"], aggregated_feedback=aggregated_feedback, review_format=selected_format)
    review = await llm.ainvoke(prompt)
    return {"simulated_review": review.content}


async def executive_summary_node(state: PaperReviewState, config: RunnableConfig):
    """Generates the executive summary."""
    print("--- STAGE 4.2: GENERATING EXECUTIVE SUMMARY ---")
    cfg = Configuration.from_runnable_config(config)
    llm = init_chat_model(model=cfg.report_enhancement_agent_model, model_kwargs=cfg.report_enhancement_agent_model_kwargs or {})
    aggregated_feedback = "\n\n".join([f"- {k.replace('_feedback','').title()}: {v}" for k, v in state.items() if k.endswith("_feedback")])
    prompt = EXECUTIVE_SUMMARY_INSTRUCTIONS.format(original_summary=state["original_paper_summary"].dict(), differentiation_analysis=state["differentiation_analysis"], aggregated_feedback=aggregated_feedback, simulated_review=state["simulated_review"])
    summary = await llm.ainvoke(prompt)
    return {"executive_summary": summary.content}


async def actionable_checklist_node(state: PaperReviewState, config: RunnableConfig):
    """Generates the actionable checklist."""
    print("--- STAGE 4.3: GENERATING ACTIONABLE CHECKLIST ---")
    cfg = Configuration.from_runnable_config(config)
    llm = init_chat_model(model=cfg.report_enhancement_agent_model, model_kwargs=cfg.report_enhancement_agent_model_kwargs or {})
    aggregated_feedback = "\n\n".join([f"- {k.replace('_feedback','').title()}: {v}" for k, v in state.items() if k.endswith("_feedback")])
    prompt = ACTIONABLE_CHECKLIST_INSTRUCTIONS.format(differentiation_analysis=state["differentiation_analysis"], aggregated_feedback=aggregated_feedback, simulated_review=state["simulated_review"])
    checklist = await llm.ainvoke(prompt)
    return {"actionable_checklist": checklist.content}


def compile_final_report_node(state: PaperReviewState):
    """Compiles all generated content into a beautifully formatted final report."""
    print("--- STAGE 4.4: ASSEMBLING FINAL REPORT ---")

    summary = state["original_paper_summary"]

    # Section 1: Deep Summary (Formatted nicely)
    summary_section = f"""
### 1.1 Purpose and Objective

{summary.purpose_objective.strip()}

### 1.2 Methods

{summary.methods.strip()}

### 1.3 Experimental Setup

{summary.experimental_setup.strip()}

### 1.4 Key Findings and Results

{summary.key_findings_results.strip()}

### 1.5 Conclusions and Implications

{summary.conclusions_implications.strip()}

### 1.6 Novelty and Contribution

{summary.novelty_contribution.strip()}

### 1.7 Limitations

{summary.limitations.strip()}
""".strip()

    # Section 2: Related Work (Already formatted markdown from the agent)
    related_work_section = state["differentiation_analysis"]

    # Section 3: Detailed Feedback (Formatted with subheadings)
    feedback_section = f"""
### 3.1 Clarity and Organization

{state['clarity_feedback'].strip()}

### 3.2 Motivation, Novelty & Significance

{state['novelty_feedback'].strip()}

### 3.3 Methodology & Evidence

{state['methodology_feedback'].strip()}

### 3.4 Technical Accuracy & Language Quality

{state['technical_feedback'].strip()}

### 3.5 Limitations & Future Work

{state['limitations_feedback'].strip()}
""".strip()

    # Section 4: Simulated Review (Already formatted markdown)
    review_section = state["simulated_review"]

    # Section 5: Executive Report (Formatted with subheadings)
    executive_section = f"""
### 5.1 Executive Summary

{state['executive_summary'].strip()}

### 5.2 Actionable Enhancement Checklist

{state['actionable_checklist'].strip()}
""".strip()

    # Assemble the final report using f-string
    final_report_str = f"""
# TeachMe: Automated Paper Review & Enhancement Report

---

## 1. Deep Summary of Manuscript

{summary_section.strip()}

---

## 2. Related Work & Differentiation Analysis

{related_work_section}

---

## 3. Detailed Feedback by Criteria

{feedback_section.strip()}

---

## 4. Simulated Peer Review

{review_section.strip()}

---

## 5. Executive Report & Action Plan

{executive_section.strip()}
""".strip()

    # Create the final assistant message and update the state
    return {"final_report": final_report_str, "messages": [AIMessage(content=final_report_str)]}


# --- Orchestrator Graph (NEW WORKFLOW) ---
workflow = StateGraph(PaperReviewState, input=MessagesState, output=MessagesState, config_schema=Configuration)

# Stage 1
workflow.add_node("summary_agent", summary_agent_node)

# Stage 2 (Subgraph)
workflow.add_node("search_agent", search_agent_entry_node)

# Stage 3 (Parallel Nodes using functools.partial)
# Create 5 logical nodes from one function by pre-filling the 'feedback_type' argument.
clarity_node = functools.partial(feedback_agent_node, feedback_type="clarity_and_organization")
novelty_node = functools.partial(feedback_agent_node, feedback_type="novelty_and_motivation")
methodology_node = functools.partial(feedback_agent_node, feedback_type="methodology_and_evidence")
technical_node = functools.partial(feedback_agent_node, feedback_type="technical_and_language_quality")
limitations_node = functools.partial(feedback_agent_node, feedback_type="limitations_and_future_work")

workflow.add_node("clarity_feedback_agent", clarity_node)
workflow.add_node("novelty_feedback_agent", novelty_node)
workflow.add_node("methodology_feedback_agent", methodology_node)
workflow.add_node("technical_feedback_agent", technical_node)
workflow.add_node("limitations_feedback_agent", limitations_node)

# Stage 4
workflow.add_node("review_agent", review_agent_node)
workflow.add_node("executive_summary_agent", executive_summary_node)
workflow.add_node("actionable_checklist_agent", actionable_checklist_node)
workflow.add_node("compile_report_agent", compile_final_report_node)

# --- Define the NEW Workflow ---
workflow.set_entry_point("summary_agent")

# After summary, start search and all feedback tasks in parallel
# workflow.add_edge("summary_agent", "search_agent")
workflow.add_edge("summary_agent", "clarity_feedback_agent")
workflow.add_edge("summary_agent", "novelty_feedback_agent")
workflow.add_edge("summary_agent", "methodology_feedback_agent")
workflow.add_edge("summary_agent", "technical_feedback_agent")
workflow.add_edge("summary_agent", "limitations_feedback_agent")

# Join the parallel branches: after search AND all feedback tasks are done, simulate the review
# workflow.add_edge("search_agent", "review_agent")
workflow.add_edge("clarity_feedback_agent", "review_agent")
workflow.add_edge("novelty_feedback_agent", "review_agent")
workflow.add_edge("methodology_feedback_agent", "review_agent")
workflow.add_edge("technical_feedback_agent", "review_agent")
workflow.add_edge("limitations_feedback_agent", "review_agent")

# After review, generate executive summary and checklist in parallel
workflow.add_edge("review_agent", "executive_summary_agent")
workflow.add_edge("review_agent", "actionable_checklist_agent")

# After both are generated, compile the final report
workflow.add_edge("executive_summary_agent", "compile_report_agent")
workflow.add_edge("actionable_checklist_agent", "compile_report_agent")

workflow.add_edge("compile_report_agent", END)

# Compile the graph
graph = workflow.compile()
