question_groundedness_critique_prompt = """
You will evaluate a context and a question asked by a science journalist.
Your task is to provide a 'total rating' scoring how well one can answer the given research-related question with the given context.

Rate on a scale of 1 to 5, where:
1 = The question cannot be answered at all using the context
2 = The question can be partially answered, but critical information is missing
3 = The question can be adequately answered, but some details are lacking
4 = The question can be well answered with most necessary details
5 = The question can be comprehensively and precisely answered with all relevant details

Consider these factors:
- Whether key scientific facts needed to answer are present in the context
- Whether expert opinions or research findings mentioned in the question are covered
- Whether the context provides enough depth for a substantial response

Provide your answer as follows:

Answer:::
Evaluation: (your BRIEF rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Question: {question}
Context: {context}
Answer:::
"""

question_relevance_critique_prompt = """
You will evaluate a question asked by a science journalist.
Your task is to rate how useful this question is for retrieving and synthesizing relevant scientific articles for an evidence-based answer.

Rate on a scale of 1 to 5, where:
1 = The question is too vague, generic, or off-topic for scientific journalism
2 = The question has minimal scientific merit, poorly formulated, or focuses on specific, named scientists
3 = The question is adequate but could be more focused or scientifically rigorous
4 = The question is well-formulated and would retrieve relevant scientific information
5 = The question is excellent, precisely targeted, and would yield high-quality scientific sources

Consider these factors:
- Specificity: Does it target a clear scientific concept or finding?
- Searchability: Does it contain specific keywords that would match relevant sources?
- Journalistic relevance: Would the answer be valuable to readers of science journalism?

Provide your answer as follows:

Answer:::
Evaluation: (your BRIEF rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Question: {question}
Answer:::
"""

question_standalone_critique_prompt = """
You will evaluate a question asked by a science journalist.
Your task is to provide a 'total rating' representing how general and universally applicable the question is.

Rate on a scale of 1 to 5, where:
1 = The question depends heavily on unstated context OR asks about specific named individuals/studies OR contains multiple subquestions
2 = The question references particular research groups, specific papers, or named scientists OR is compound (contains "and how" or similar conjunctions linking distinct inquiries)
3 = The question is somewhat specific but applies to a recognizable field of research
4 = The question is general with minor improvements possible
5 = The question is completely general and could be answered from multiple scientific sources

IMPORTANT CRITERIA:
- PENALIZE questions that ask about specific people by name (e.g., "Welche Erkenntnisse hat Professor Müller in seiner Studie gewonnen?")
- PENALIZE questions that reference specific studies (e.g., "Was sind die Haupterkenntnisse der Studie?")

Consider these factors:
- Generality: Does it ask about scientific concepts rather than specific research outputs?
- Independence from named sources: Does it avoid referring to specific people or papers?

Provide your answer as follows:

Answer:::
Evaluation: (your BRIEF rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Question: {question}
Answer:::
"""