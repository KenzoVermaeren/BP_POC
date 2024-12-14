from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import Literal
# from langchain_openai import OpenAI

PROMPT = """
# Data
Title: {title}

Mentioned negatives: '{negatives}'
Mentioned positives: '{positives}'

Review: {content}


# Review Analysis Task

Objective: Conduct a comprehensive analysis of customer reviews to derive actionable insights and emotional intelligence.

Detailed Analysis Requirements:

1. Emotional Mapping:
   - Create a structured mapping of each review to a specific emotion
   - Criteria for emotion selection:
     * Analyze both the numerical "score" and textual "content"
     * Provide a clear, logical rationale for each emotion assignment
     * Use only following emotions: (angry, frustrated, sad, neutral, happy)

2. Overall Sentiment Assessment:
   - Develop a holistic emotional interpretation of the review collection
   - Explain the reasoning behind the general feeling
   - Support conclusions with specific review excerpts or statistical insights

3. Quantitative Analysis:
   - Calculate the precise average rating
   - Provide statistical context for the average (e.g., distribution, median, mode)

4. Temporal Emotional Tracking:
   - Create a chronological timeline mapping reviews to their emotional tone
   - Highlight any significant emotional trends or shifts over time

5. Detailed Feedback Breakdown:
   - Identify and categorize the most frequent positive ("plus") aspects
   - Systematically analyze the most common negative ("min") points
   - For each significant negative aspect:
     * Propose specific, actionable solutions
     * Provide market-informed recommendations for improvement

6. Strategic Market Recommendation:
   - Develop a comprehensive market strategy recommendation
   - Base suggestions on the holistic review analysis
   - Include insights on product positioning, potential improvements, and market opportunities

Deliverable Format:
- Provide a well-structured dictionary/library with clear sections
- Ensure each section is thoroughly explained
- Include both qualitative insights and quantitative data

Context Considerations:
- Consider the broader market context
- Analyze reviews with an objective, nuanced perspective
- Balance emotional intelligence with data-driven insights

{format_instructions}
"""




class Output(BaseModel):
    emotion: Literal['angry', 'frustrated', 'sad', 'neutral', 'happy'] = Field("The emotion associated with the result")
    score: int = Field("The score you think the user would give to this product based on the review. 0 for very unhappy, 10 for very happy",)
    


parser = PydanticOutputParser(
    pydantic_object=Output
)

template = PromptTemplate.from_template(
    PROMPT, partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm = ChatOpenAI(model="gpt-4o-mini") 

chain = template | llm.with_retry() | parser.with_retry()

