from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import Literal
# from langchain_openai import OpenAI

PROMPT = """
# Data

Title: {title}

list positive_aspects: '{positives}'
list negative_aspects: '{negatives}'

Mentioned negatives: '{negatives}'
Mentioned positives: '{positives}'

Review: {content}

analysis_emotion: {analysis_emotion}
analysis_score: {analysis_score}

# Review Analysis Task

Objective: Conduct a comprehensive analysis of customer reviews to derive actionable insights and emotional intelligence.

Detailed Analysis Requirements:

1. Analyze Positive and Negative Aspects:
Review the plus_list and min_list to identify the top 3 most frequently mentioned positive aspects and the top 3 most frequently mentioned negative aspects.
Highlight the most common negative point and propose a practical solution to address it.

2. Provide a Health Ratio:
Based on all analysis emotions and analysis scores between 0 and 100, calculate and summarize an overall health ratio for the product. This should reflect the general sentiment and quality as perceived by users.

3. Develop a Market Solution:
Using all the provided data, create a market strategy or solution for a retailer specializing in electronics. 
Focus on actionable insights that can enhance sales, improve customer satisfaction, or optimize product offerings.
   - Develop a comprehensive market strategy recommendation
   - Base suggestions on the holistic review analysis
   - Include insights on product positioning, potential improvements, and market opportunities

{format_instructions}
"""




class Output(BaseModel):
    positives: list[str] = Field("The list of 3 most frequently mentioned positive aspects")
    negatives: list[str] = Field("The list of 3 most frequently mentioned negative aspects")
    negative_solution: list[str] = Field("The solutions for the most negative aspect")
    negative_reasoning: list[str] = Field("The reasoning behind the solutions for the negative aspects")
    score: int = Field("A percentage between 0 and 100, based on the content and weight of emotion, that shows the overall healthratio of the product")
    score_reasoning: list[str] = Field("The reasoning behind the the given healthratio")
    Market_solution: list[str] = Field("A list of market strategies that can be taken for the product")   
    Market_reasoning: list[str] = Field("The reasoning behind the market strategies") 


parser = PydanticOutputParser(
    pydantic_object=Output
)

template = PromptTemplate.from_template(
    PROMPT, partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm = ChatOpenAI(model="gpt-4o-mini") 

chain1 = template | llm.with_retry() | parser.with_retry()