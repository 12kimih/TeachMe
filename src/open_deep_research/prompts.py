# TeachMe Agent Prompts

# Stage 1: Summary Agent
PAPER_SUMMARY_INSTRUCTIONS = """
You are an expert AI researcher. Your task is to conduct a deep summarization of the provided academic manuscript text.
Analyze the text and extract the following information based on the specified criteria.
The output MUST be a JSON object that follows the provided schema.

<Manuscript Text>
{paper_text}
</Manuscript Text>

<Criteria>
1.  **Purpose/Objective:** The main goal or research question addressed in the paper.
2.  **Methods:** A summary of the approach, methodology, or experimental design.
3.  **Experimental Setup:** Details on models, algorithms, datasets, benchmarks, evaluation metrics, and baselines.
4.  **Key Findings/Results:** The most important findings or results.
5.  **Conclusions/Implications:** The principal conclusions and broader implications of the study.
6.  **Novelty/Contribution:** The paper's originality and its unique contribution to the field.
7.  **Limitations:** Any acknowledged limitations or areas for future research mentioned in the paper.
</Criteria>

<Format>
Call the `DeepSummary` tool with the extracted information.
</Format>
""".strip()

# Stage 2: Search Agent
SEARCH_KEYWORD_GENERATION_INSTRUCTIONS = """
You are an expert search strategist for academic literature.
Based on the deep summary of a manuscript, your goal is to generate {number_of_queries} targeted, keyword-based search queries to find related work.
The queries should be concise and effective for searching academic databases and search engines like Google Scholar or arXiv.
Prefix each query with one of the target conferences to narrow down the search scope.

<Manuscript Deep Summary>
{deep_summary}
</Manuscript Deep Summary>

<Target Conferences>
{target_conferences}
</Target Conferences>

<Task>
Generate {number_of_queries} search queries.
For example: "NeurIPS graph neural networks for drug discovery" or "ICLR self-supervised learning for computer vision".
</Task>

<Format>
Call the `SearchQueries` tool.
</Format>
""".strip()

RELATED_WORK_RELEVANCE_CHECK_INSTRUCTIONS = """
You are a research assistant. Your task is to determine if a candidate paper is relevant to the original manuscript.
Compare the candidate paper's title and abstract with the original manuscript's summary.

<Original Manuscript Summary>
{original_summary}
</Original Manuscript Summary>

<Candidate Paper Title>
{candidate_title}
</Candidate Paper Title>

<Candidate Paper Abstract>
{candidate_abstract}
</Candidate Paper Abstract>

<Task>
Assess the relevance. The candidate paper is relevant if it addresses a similar problem, uses a related methodology, or builds upon similar concepts.
Provide a clear 'yes' or 'no' decision and a brief justification.
</Task>

<Format>
Call the `RelevanceDecision` tool with your assessment.
</Format>
""".strip()

DIFFERENTIATION_ANALYSIS_INSTRUCTIONS = """
You are an expert academic reviewer. Your task is to perform a comparative and differentiation analysis.
You will be given the deep summary of an original manuscript and a list of deep summaries from related works.
Your goal is to identify commonalities, differences, and unique contributions, then provide actionable recommendations.

<Original Manuscript Summary>
{original_summary}
</Original Manuscript Summary>

<Related Works Summaries>
{related_summaries}
</Related Works Summaries>

<Task>
Analyze the provided information and produce a detailed differentiation analysis covering the following points:
1.  **Core Commonalities and Differences:** What are the key similarities and differences in methodology, goals, and findings between the original manuscript and the related works?
2.  **Distinct or Innovative Aspects:** In what specific ways is the original manuscript distinct or innovative? Highlight its unique contributions.
3.  **Actionable Recommendations:** Suggest concrete, actionable experiments, analyses, or framing adjustments that could further enhance the manuscript's uniqueness and clearly differentiate it from the related work.
</Task>

<Format>
Provide your analysis as a well-structured markdown text.
</Format>
""".strip()

# Stage 3: Feedback Agent
FEEDBACK_INSTRUCTIONS_TEMPLATE = """
You are an expert academic reviewer providing constructive feedback on a manuscript.
Your feedback should be specific, actionable, and aimed at improving the quality of the paper.
Focus ONLY on the specific criteria assigned to you.

<Assigned Feedback Criteria>
{feedback_criteria}
</Assigned Feedback Criteria>

<Manuscript Text>
{paper_text}
</Manuscript Text>

<Task>
Review the manuscript text thoroughly based *only* on your assigned criteria.
Generate detailed, targeted improvement suggestions. For any claims lacking sufficient evidence, propose specific additional experiments or analyses to address these gaps.
</Task>

<Format>
Provide your feedback as a well-structured markdown text.
</Format>
""".strip()

FEEDBACK_CRITERIA = {
    "clarity_and_organization": """
    - **Clarity & Organization:**
        - **Clear Research Question or Thesis:** Is the research question/hypothesis explicitly stated and clear?
        - **Logical Structure:** Is the paper well-structured with a coherent flow of ideas?
        - **Engaging Abstract and Introduction:** Do the abstract and introduction provide a clear roadmap?
        - **Clarity and Precision:** Is the writing clear, concise, and free of unnecessary jargon?
    """.strip(),
    "novelty_and_motivation": """
    - **Motivation, Novelty & Significance:**
        - **Strong Motivation:** Is the research problem well-justified? Are gaps in existing literature identified?
        - **Novelty and Contribution:** Are the unique contributions explicitly highlighted?
        - **Significance of Results:** Is the broader impact of the findings well-articulated?
    """.strip(),
    "methodology_and_evidence": """
    - **Methodology & Evidence:**
        - **Well-Defined Methods:** Is the methodology described in enough detail for reproducibility?
        - **Evidence and Reasoning:** Are all claims substantiated with solid evidence, citations, or logical arguments?
    """.strip(),
    "technical_and_language_quality": """
    - **Technical Accuracy & Language Quality:**
        - **Verification of Formulas:** Are mathematical proofs and formulas correct? (Acknowledge if you cannot verify fully but point out potential issues).
        - **Grammar and Spelling:** Is the manuscript free of grammatical and spelling errors?
    """.strip(),
    "limitations_and_future_work": """
    - **Limitations & Future Work:**
        - **Acknowledged Limitations:** Are the study's limitations transparently discussed?
        - **Future Work:** Are thoughtful avenues for future research suggested?
    """.strip(),
}

# --- Stage 4: Review Agent Prompts (NEW) ---

HOW_TO_WRITE_GOOD_REVIEWS = """
## **How to Write Good Reviews**

- Take the time to write good reviews. Ideally, you should read a paper and then think about it over the course of several days before you write your review.
- Short reviews are unhelpful to authors, other reviewers, and Area Chairs. If you have agreed to review a paper, you should take enough time to write a thoughtful and detailed review. Bullet lists with one short sentence per bullet are NOT a detailed review.
- Be specific when you suggest that the writing needs to be improved. If there is a particular section that is unclear, point it out and give suggestions for how it can be clarified.
- Be specific about novelty. Claims in a review that the submitted work “has been done before” MUST be backed up with specific references and an explanation of how closely they are related. At the same time, for a positive review, be sure to summarize what novel aspects are most interesting in the Strengths section.
- Do not reject papers solely because they are missing citations or comparisons to prior work that has only been published without review (e.g., arXiv or technical reports). Refer to the FAQ below for more details on handling arXiv prior art.
- Do not give away your identity by asking the authors to cite several of your own papers.
- If you think the paper is out of scope for CVPR's subject areas, clearly explain why in the review. Then suggest other publication possibilities (journals, conferences, workshops) that would be a better match for the paper. However, unless the area mismatch is extreme, you should keep an open mind, because we want a diverse set of good papers at the conference.
- The tone of your review is important. A harshly written review will be resented by the authors, regardless of whether your criticisms are true. If you take care, it is always possible to word your review constructively while staying true to your thoughts about the paper.
- Avoid referring to the authors in the second person (“you”). It is best to avoid the term “the authors” as well, because you are reviewing their work and not the person. Instead, use the third person (“the paper”). Referring to the authors as “you” can be perceived as being confrontational, even though you may not mean it this way.
- Be generous about giving the authors new ideas for how they can improve their work. You might suggest a new technical tool that could help, a dataset that could be tried, an application area that might benefit from their work, or a way to generalize their idea to increase its impact.
- Please refer to the Ethics Guidelines and Suggested Practices for Authors page to find out how to handle specific issues that may arise.
""".strip()

REVIEW_EXAMPLES = """
## **Review Examples**

Here are two sample reviews from previous conferences that give an example of what we consider a good review for the case of leaning-to-accept and leaning-to-reject.

---

### **Review for a Paper where Leaning-to-Accept**

This paper proposes a method, Dual-AC, for optimizing the actor (policy) and critic (value function) simultaneously which takes the form of a zero-sum game resulting in a principled method for using the critic to optimize the actor. In order to achieve that, they take the linear programming approach of solving the Bellman optimality equations, outline the deficiencies of this approach, and propose solutions to mitigate those problems. The discussion on the deficiencies of the naive LP approach is mostly well done. Their main contribution is extending the single step LP formulation to a multi-step dual form that reduces the bias and makes the connection between policy and value function optimization much clearer without losing convexity by applying a regularization. They perform an empirical study in the Inverted Double Pendulum domain to conclude that their extended algorithm outperforms the naive linear programming approach without the improvements. Lastly, there are empirical experiments done to conclude the superior performance of Dual-AC in contrast to other actor-critic algorithms.

Overall, this paper could be a significant algorithmic contribution, with the caveat for some clarifications on the theory and experiments. Given these clarifications in an author response, I would be willing to increase the score.

For the theory, there are a few steps that need clarification and further clarification on novelty. For novelty, it is unclear if Theorem 2 and Theorem 3 are both being stated as novel results. It looks like Theorem 2 has already been shown in "Randomized Linear Programming Solves the Discounted Markov Decision Problem in Nearly-Linear Running Time”. There is a statement that “Chen & Wang (2016); Wang (2017) apply stochastic first-order algorithms (Nemirovski et al., 2009) for the one-step Lagrangian of the LP problem in reinforcement learning setting. However, as we discussed in Section 3, their algorithm is restricted to tabular parametrization”. Is your Theorem 2 somehow an extension? Is Theorem 3 completely new?

This is particularly called into question due to the lack of assumptions about the function class for value functions. It seems like the value function is required to be able to represent the true value function, which can be almost as restrictive as requiring tabular parameterizations (which can represent the true value function). This assumption seems to be used right at the bottom of Page 17, where U^{pi*} = V^*. Further, eta_v must be chosen to ensure that it does not affect (constrain) the optimal solution, which implies it might need to be very small. More about conditions on eta_v would be illuminating.

There is also one step in the theorem that I cannot verify. On Page 18, how is the squared removed for difference between U and Upi? The transition from the second line of the proof to the third line is not clear. It would also be good to more clearly state on page 14 how you get the first inequality, for || V^* ||_{2,mu}^2.

**For the experiments, the following should be addressed.**

1. It would have been better to also show the performance graphs with and without the improvements for multiple domains.

2. The central contribution is extending the single step LP to a multi-step formulation. It would be beneficial to empirically demonstrate how increasing k (the multi-step parameter) affects the performance gains.

3. Increasing k also comes at a computational cost. I would like to see some discussions on this and how long dual-AC takes to converge in comparison to the other algorithms tested (PPO and TRPO).

4. The authors concluded the presence of local convexity based on hessian inspection due to the use of path regularization. It was also mentioned that increasing the regularization parameter size increases the convergence rate. Empirically, how does changing the regularization parameter affect the performance in terms of reward maximization? In the experimental section of the appendix, it is mentioned that multiple regularization settings were tried but their performance is not mentioned. Also, for the regularization parameters that were tried, based on hessian inspection, did they all result in local convexity? A bit more discussion on these choices would be helpful.

**Minor comments:**

1. Page 2: In equation 5, there should not be a 'ds' in the dual variable constraint

---

### **Review for a Paper where Leaning-to-Reject**

This paper introduces a variation on temporal difference learning for the function approximation case that attempts to resolve the issue of over-generalization across temporally-successive states. The new approach is applied to both linear and non-linear function approximation, and for prediction and control problems. The algorithmic contribution is demonstrated with a suite of experiments in classic benchmark control domains (Mountain Car and Acrobot), and in Pong.

This paper should be rejected because (1) the algorithm is not well justified either by theory or practice, (2) the paper never clearly demonstrates the existence of problem they are trying to solve (nor differentiates it from the usual problem of generalizing well), (3) the experiments are difficult to understand, missing many details, and generally do not support a significant contribution, and (4) the paper is imprecise and unpolished.

**Main argument**

The paper does not do a great job of demonstrating that the problem it is trying to solve is a real thing. There is no experiment in this paper that clearly shows how this temporal generalization problem is different from the need to generalize well with function approximation. The paper points to references to establish the existence of the problem, but for example the Durugkar and Stone paper is a workshop paper and the conference version of that paper was rejected from ICLR 2018 and the reviewers highlighted serious issues with the paper—that is not work to build upon. Further the paper under review here claims this problem is most pressing in the non-linear case, but the analysis in section 4.1 is for the linear case.

The resultant algorithm does not seem well justified, and has a different fixed point than TD, but there is no discussion of this other than section 4.4, which does not make clear statements about the correctness of the algorithm or what it converges to. Can you provide a proof or any kind of evidence that the proposed approach is sound, or how it's fixed point relates to TD?

The experiments do not provide convincing evidence of the correctness of the proposed approach or its utility compared to existing approaches. There are so many missing details it is difficult to draw many conclusions:

1. What was the policy used in exp1 for policy evaluation in MC?
2. Why Fourier basis features?
3. In MC with DQN how did you adjust the parameters and architecture for the MC task?
4. Was the reward in MC and Acrobot -1 per step or something else
5. How did you tune the parameters in the MC and Acrobot experiments?
6. Why so few runs in MC, none of the results presented are significant?
7. Why is the performance so bad in MC?
8. Did you evaluate online learning or do tests with the greedy policy?
9. How did you initialize the value functions and weights?
10. Why did you use experience replay for the linear experiments?
11. IN MC and Acrobot why only a one layer MLP?

Ignoring all that, the results are not convincing. Most of the results in the paper are not statistically significant. The policy evaluation results in MC show little difference to regular TD. The Pong results show DQN is actually better. This makes the reader wonder if the result with DQN on MC and Acrobot are only worse because you did not properly tune DQN for those domains, whereas the default DQN architecture is well tuned for Atari and that is why you method is competitive in the smaller domains.

The differences in the “average change in value plots” are very small if the rewards are -1 per step. Can you provide some context to understand the significance of this difference? In the last experiment linear FA and MC, the step-size is set equal for all methods—this is not a valid comparison. Your method may just work better with alpha = 0.1.

**The paper has many imprecise parts, here are a few:**

1. The definition of the value function would be approximate not equals unless you specify some properties of the function approximation architecture. Same for the Bellman equation
2. equation 1 of section 2.1 is neither an algorithm or a loss function
3. TD does not minimize the squared TD. Saying that is the objective function of TD learning in not true
4. end of section 2.1 says “It is computed as” but the following equation just gives a form for the partial derivative
5. equation 2, x is not bounded
6. You state TC-loss has an unclear solution property, I don't know what that means and I don't think your approach is well justified either
7. Section 4.1 assumes linear FA, but its implied up until paragraph 2 that it has not assumed linear
8. treatment of n_t in alg differs from appendix (t is no time episode number)
9. Your method has a n_t parameter that is adapted according to a schedule seemingly giving it an unfair advantage over DQN.
10. Over-claim not supported by the results: “we see that HR-TD is able to find a representation that is better at keeping the target value separate than TC is “. The results do not show this.
11. Section 4.4 does not seem to go anywhere or produce and tangible conclusions

**Things to improve the paper that did not impact the score:**

1. It's hard to follow how the prox operator is used in the development of the alg, this could use some higher level explanation
2. Intro p2 is about bootstrapping, use that term and remove the equations
3. It's not clear why you are talking about stochastic vs deterministic in P3
4. Perhaps you should compare against a MC method in the experiments to demonstrate the problem with TD methods and generalization
5. Section 2: “can often be a regularization term” >> can or must be?
6. update law is an odd term
7. tends to alleviate” >> odd phrase
8. section 4 should come before section 3
9. Alg 1 in not helpful because it just references an equation
10. section 4.4 is very confusing, I cannot follow the logic of the statements
11. Q learning >> Q-learning
12. Not sure what you mean with the last sentence of p2 section 5
13. where are the results for Acrobot linear function approximation
14. appendix Q-learning with linear FA is not DQN (table 2)
""".strip()

# Conference Review Formats
NEURIPS_REVIEW_FORMAT = """
1. **Summary:** Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary. This is also not the place to paste the abstract—please provide the summary in your own understanding after reading.
2. **Strengths and Weaknesses:** Please provide a thorough assessment of the strengths and weaknesses of the paper. A good mental framing for strengths and weaknesses is to think of reasons you might accept or reject the paper. Please touch on the following dimensions:
    1. *Quality*: Is the submission technically sound? Are claims well supported (e.g., by theoretical analysis or experimental results)? Are the methods used appropriate? Is this a complete piece of work or work in progress? Are the authors careful and honest about evaluating both the strengths and weaknesses of their work?
    2. *Clarity*: Is the submission clearly written? Is it well organized? (If not, please make constructive suggestions for improving its clarity.) Does it adequately inform the reader? (Note that a superbly written paper provides enough information for an expert reader to reproduce its results.)
    3. *Significance:* Are the results impactful for the community? Are others (researchers or practitioners) likely to use the ideas or build on them? Does the submission address a difficult task in a better way than previous work? Does it advance our understanding/knowledge on the topic in a demonstrable way? Does it provide unique data, unique conclusions about existing data, or a unique theoretical or experimental approach?
    4. *Originality:* Does the work provide new insights, deepen understanding, or highlight important properties of existing methods? Is it clear how this work differs from previous contributions, with relevant citations provided? Does the work introduce novel tasks or methods that advance the field? Does this work offer a novel combination of existing techniques, and is the reasoning behind this combination well-articulated? **As the questions above indicates, originality does not necessarily require introducing an entirely new method. Rather, a work that provides novel insights by evaluating existing methods, or demonstrates improved efficiency, fairness, etc. is also equally valuable.**You can incorporate Markdown and LaTeX into your review.
3. **Quality:** Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the quality of the work.
    1. 4 excellent
    2. 3 good
    3. 2 fair
    4. 1 poor
4. **Clarity:** Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the clarity of the paper.
    1. 4 excellent
    2. 3 good
    3. 2 fair
    4. 1 poor
5. **Significance:** Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the significance of the paper.
    1. 4 excellent
    2. 3 good
    3. 2 fair
    4. 1 poor
6. **Originality:** Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the originality of the paper.
    1. 4 excellent
    2. 3 good
    3. 2 fair
    4. 1 poor
7. **Questions:** Please list up and carefully describe questions and suggestions for the authors, which should focus on key points (ideally around 3–5) that are actionable with clear guidance. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation. You are strongly encouraged to state the clear criteria under which your evaluation score could increase or decrease. This can be very important for a productive rebuttal and discussion phase with the authors.
8. **Limitations:** Have the authors adequately addressed the limitations and potential negative societal impact of their work? If so, simply leave “yes”; if not, please include constructive suggestions for improvement. In general, authors should be rewarded rather than punished for being up front about the limitations of their work and any potential negative societal impact. You are encouraged to think through whether any critical points are missing and provide these as feedback for the authors.
9. **Overall:** Please provide an "overall score" for this submission. Choices:
    1. 6: Strong Accept: Technically flawless paper with groundbreaking impact on one or more areas of AI, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations.
    2. 5: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
    3. 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
    4. 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
    5. 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
    6. 1: Strong Reject: For instance, a paper with well-known results or unaddressed ethical considerations
10. **Confidence:** Please provide a "confidence score" for your assessment of this submission to indicate how confident you are in your evaluation. Choices
    1. 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
    2. 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
    3. 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
    4. 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
    5. 1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked.
""".strip()

ICLR_REVIEW_FORMAT = """
1. **Read the paper:** It's important to carefully read through the entire paper, and to look up any related work and citations that will help you comprehensively evaluate it. Be sure to give yourself sufficient time for this step.
2. **While reading, consider the following:**
    1. Objective of the work: What is the goal of the paper? Is it to better address a known application or problem, draw attention to a new application or problem, or to introduce and/or explain a new theoretical finding? A combination of these? Different objectives will require different considerations as to potential value and impact.
    2. Strong points: is the submission clear, technically correct, experimentally rigorous, reproducible, does it present novel findings (e.g. theoretically, algorithmically, etc.)?
    3. Weak points: is it weak in any of the aspects listed in b.?
    4. Be mindful of potential biases and try to be open-minded about the value and interest a paper can hold for the entire ICLR community, even if it may not be very interesting for you.
3. **Answer four key questions for yourself to make a recommendation to Accept or Reject:**
    1. What is the specific question and/or problem tackled by the paper?
    2. Is the approach well motivated, including being well-placed in the literature?
    3. Does the paper support the claims? This includes determining if results, whether theoretical or empirical, are correct and if they are scientifically rigorous.
    4. What is the significance of the work? Does it contribute new knowledge and sufficient value to the community? Note, this does not necessarily require state-of-the-art results. Submissions bring value to the ICLR community when they convincingly demonstrate new, relevant, impactful knowledge (incl., empirical, theoretical, for practitioners, etc).
4. **Write and submit your initial review, organizing it as follows:**
    1. Summarize what the paper claims to contribute. Be positive and constructive.
    2. List strong and weak points of the paper. Be as comprehensive as possible.
    3. Clearly state your initial recommendation (accept or reject) with one or two key reasons for this choice.
    4. Provide supporting arguments for your recommendation.
    5. Ask questions you would like answered by the authors to help you clarify your understanding of the paper and provide the additional evidence you need to be confident in your assessment.
    6. Provide additional feedback with the aim to improve the paper. Make it clear that these points are here to help, and not necessarily part of your decision assessment.
""".strip()

ICML_REVIEW_FORMAT = """
### **Summary**

1. Briefly summarize the paper (including the main findings, main results, main algorithmic/conceptual ideas, etc. that the paper claims to contribute). This summary should not be used to critique the paper. A well-written summary should not be disputed by the authors of the paper or other readers.

### **Claims and Evidence**

1. Are the claims made in the submission supported by clear and convincing evidence? If not, which claims are problematic and why?
2. Do proposed methods and/or evaluation criteria (e.g., benchmark datasets) make sense for the problem or application at hand?
3. Did you check the correctness of any proofs for theoretical claims? Please specify which ones, and discuss any issues.
4. Did you check the soundness/validity of any experimental designs or analyses? Please specify which ones, and discuss any issues.
5. Did you review the supplementary material? Which parts?

### **Relation to Prior Works**

1. How are the key contributions of the paper related to the broader scientific literature? Be specific in terms of prior related findings/results/ideas/etc.
2. Are there related works that are essential to understanding the (context for) key contributions of the paper, but are not currently cited/discussed in the paper? Be specific in terms of prior related findings/results/ideas/etc. (Example: “The key contribution is a linear-time algorithm, and only cites a quadratic-time algorithm for the same problem as prior work, but there was also an O(n log n) time algorithm for this problem discovered last year, namely Algorithm 3 from [ABC'24] published in ICML 2024.”)
3. How well-versed are you with the literature related to this paper? (Examples: “I keep up with the literature in this area.”; “I am only familiar with a few key papers in this area, namely [ABC'02], [DEF'04], and [GHI'05].”) **Note: Your response to this item will not be visible to authors. Please also see instructions regarding concurrent work.**

### **Other Aspects**

1. Enter any comments on other strengths and weaknesses of the paper, such as those concerning originality, significance, and clarity. We encourage you to be open-minded in terms of potential strengths. For example, originality may arise from creative combinations of existing ideas, removing restrictive assumptions from prior theoretical results, or application to a real-world use case (particularly for application-driven ML papers, indicated in the flag above and described in the Reviewer Instructions).
2. If you have any other comments or suggestions (e.g., a list of typos), please write them here.

### **Questions for Authors**

1. If you have any important questions for the authors, please carefully formulate them here. Please reserve your questions for cases where the response would likely change your evaluation of the paper, clarify a point in the paper that you found confusing, or address a critical limitation you identified. Please number your questions so authors can easily refer to them in the response, and explain how possible responses would change your evaluation of the paper.

### **Overall Recommendation**

1. Indicate an overall recommendation:
    - 5. **Strong accept**
    - 4. **Accept**
    - 3. **Weak accept** (i.e., leaning towards accept, but could also be rejected)
    - 2. **Weak reject** (i.e., leaning towards reject, but could also be accepted)
    - 1. **Reject**
""".strip()

ACL_REVIEW_FORMAT = """
### **Paper Summary**

Describe what this paper is about. This should help action editors and area chairs to understand the topic of the work and highlight any possible misunderstandings.

### **Summary of Strengths**

What are the major reasons to publish this paper at a selective *ACL venue? These could include novel and useful methodology, insightful empirical results or theoretical analysis, clear organization of related literature, or any other reason why interested readers of *ACL papers may find the paper useful.

### **Summary of Weaknesses**

What are the concerns that you have about the paper that would cause you to favor prioritizing other high-quality papers that are also under consideration for publication? These could include concerns about correctness of the results or argumentation, limited perceived impact of the methods or findings (note that impact can be significant both in broad or in narrow sub-fields), lack of clarity in exposition, or any other reason why interested readers of *ACL papers may gain less from this paper than they would from other papers under consideration. Where possible, please number your concerns so authors may respond to them individually.

### **Comments/Suggestions/Typos**

If you have any comments to the authors about how they may improve their paper, other than addressing the concerns above, please list them here.

### **Reviewer Confidence**

- 5 = Positive that my evaluation is correct. I read the paper very carefully and am familiar with related work.
- 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
- 3 = Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design.
- 2 = Willing to defend my evaluation, but it is fairly likely that I missed some details, didn't understand some central points, or can't be sure about the novelty of the work.
- 1 = Not my area, or paper is very hard to understand. My evaluation is just an educated guess.

### **Soundness**

Given that this is a short/long paper, is it sufficiently sound and thorough? Does it clearly state scientific claims and provide adequate support for them? For experimental papers: consider the depth and/or breadth of the research questions investigated, technical soundness of experiments, methodological validity of evaluation. For position papers, surveys: consider whether the current state of the field is adequately represented and main counter-arguments acknowledged. For resource papers: consider the data collection methodology, resulting data & the difference from existing resources are described in sufficient detail.

- 5 = Excellent: This study is one of the most thorough I have seen, given its type.
- 4.5
- 4 = Strong: This study provides sufficient support for all of its claims. Some extra experiments could be nice, but not essential.
- 3.5
- 3 = Acceptable: This study provides sufficient support for its main claims. Some minor points may need extra support or details.
- 2.5
- 2 = Poor: Some of the main claims are not sufficiently supported. There are major technical/methodological problems.
- 1.5
- 1 = Major Issues: This study is not yet sufficiently thorough to warrant publication or is not relevant to ACL.

### **Excitement**

How exciting is this paper for you? Excitement is *subjective*, and does not necessarily follow what is popular in the field. We may perceive papers as transformational/innovative/surprising, e.g. because they present conceptual breakthroughs or evidence challenging common assumptions/methods/datasets/metrics. We may be excited about the possible impact of the paper on some community (not necessarily large or our own), e.g. lowering barriers, reducing costs, enabling new applications. We may be excited for papers that are relevant, inspiring, or useful for our own research. These factors may combine in different ways for different reviewers.

- 5 = Highly Exciting: I would recommend this paper to others and/or attend its presentation in a conference.
- 4.5
- 4 = Exciting: I would mention this paper to others and/or make an effort to attend its presentation in a conference.
- 3.5
- 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.
- 2.5
- 2 = Potentially Interesting: this paper does not resonate with me, but it might with others in the *ACL community.
- 1.5
- 1 = Not Exciting: this paper does not resonate with me, and I don't think it would with others in the *ACL community (e.g. it is in no way related to computational processing of language).

### **Overall Assessment**

If this paper was committed to an *ACL conference, do you believe it should be accepted? If you recommend conference, Findings and or even award consideration, you can still suggest minor revisions (e.g. typos, non-core missing refs, etc.).

Outstanding papers should be either fascinating, controversial, surprising, impressive, or potentially field-changing. Awards will be decided based on the camera-ready version of the paper. ACL award policy: https://www.aclweb.org/adminwiki/index.php/ACL_Conference_Awards_Policy

Main vs Findings papers: the main criteria for Findings are soundness and reproducibility. Conference recommendations may also consider novelty, impact and other factors.

- 5 = Consider for Award: I think this paper could be considered for an outstanding paper award at an *ACL conference (up to top 2.5% papers).
- 4.5 = Borderline Award
- 4 = Conference: I think this paper could be accepted to an *ACL conference.
- 3.5 = Borderline Conference
- 3 = Findings: I think this paper could be accepted to the Findings of the ACL.
- 2.5 = Borderline Findings
- 2 = Resubmit next cycle: I think this paper needs substantial revisions that can be completed by the next ARR cycle.
- 1.5 = Resubmit after next cycle: I think this paper needs substantial revisions that cannot be completed by the next ARR cycle.
- 1 = Do not resubmit: This paper has to be fully redone, or it is not relevant to the *ACL community (e.g. it is in no way related to computational processing of language).

### **Best paper justification**

If your overall assessment for this paper is either 'Consider for award' or 'Borderline award', please briefly describe why.
""".strip()

CONFERENCE_FORMATS = {
    "neurips": NEURIPS_REVIEW_FORMAT,
    "iclr": ICLR_REVIEW_FORMAT,
    "icml": ICML_REVIEW_FORMAT,
    "acl": ACL_REVIEW_FORMAT,
}

REVIEW_SIMULATION_INSTRUCTIONS = """
You are a simulated peer reviewer for a top-tier AI conference.
Your task is to generate a realistic conference review for the manuscript, based on all the information gathered so far, and a specific conference review format.
You must provide both a quantitative score (if applicable in the format) and detailed qualitative comments.

<Background Information for Review>
- **How to Write a Good Review:** {how_to_write_good_reviews}
- **Examples of Good Reviews:** {review_examples}
- **Manuscript Deep Summary:** {original_summary}
- **Differentiation Analysis:** {differentiation_analysis}
- **Aggregated Feedback:** {aggregated_feedback}
</Background Information for Review>

<Task>
Carefully read all the background information. Then, fill out the review form below.
Your comments should be critical but constructive, reflecting the strengths and weaknesses identified in the provided materials.
The tone should be professional and academic, following the guidelines.
</Task>

<Conference Review Format to Use>
{review_format}
</Conference Review Format to Use>

<Format>
Produce the final review as a markdown document, strictly following the provided conference format.
</Format>
""".strip()

# Final Report Generation Prompts (NEW)
EXECUTIVE_SUMMARY_INSTRUCTIONS = """
You are an expert editor creating an executive summary for a manuscript review report.
Synthesize all the provided information to generate a high-level overview.

<Context>
- **Original Manuscript Summary:** {original_summary}
- **Differentiation Analysis:** {differentiation_analysis}
- **Aggregated Feedback:** {aggregated_feedback}
- **Simulated Peer Review:** {simulated_review}
</Context>

<Task>
Write a concise **Executive Summary**. It should be a high-level overview highlighting the manuscript's main purpose, key findings, and novelty. Briefly mention its core strengths and the major areas for improvement identified in the review. The summary is intended to provide editors, authors, or reviewers with a quick yet comprehensive snapshot of the manuscript's status.
</Task>

<Format>
Provide the summary as a well-structured markdown text.
</Format>
""".strip()

ACTIONABLE_CHECKLIST_INSTRUCTIONS = """
You are a pragmatic managing editor creating an actionable enhancement checklist for authors.
Based on all the feedback and analysis, create a prioritized list of concrete actions.

<Context>
- **Differentiation Analysis:** {differentiation_analysis}
- **Aggregated Feedback:** {aggregated_feedback}
- **Simulated Peer Review:** {simulated_review}
</Context>

<Task>
Create an **Actionable Enhancement Checklist**. This should be a bullet-pointed, prioritized list of concrete actions the authors should take to improve their manuscript. Group the suggestions by type (e.g., Experimental, Structural, Language/Clarity, Differentiation) or by manuscript section (e.g., Abstract, Methods, Discussion) for clarity.
</Task>

<Format>
Provide the checklist as a well-structured markdown list.
</Format>
""".strip()
