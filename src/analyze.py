from typing import Dict, List
from langchain_core.runnables import Runnable
import pandas as pd
import asyncio

def analyze_reviews(reviews: pd.DataFrame, chain: Runnable) -> List[Dict[str, dict]]:
    """
    Analyze a list of reviews using a language model chain for sentiment or detailed analysis.

    This function takes a list of reviews and processes each review through a provided
    language model chain. It returns a list of dictionaries, each containing the original
    review and its corresponding analysis.

    Args:
        reviews (List[dict]): A list of review dictionaries to be analyzed.
            Each review should be a dictionary containing review details.
        chain (Runnable): A LangChain runnable (typically an LLM chain) that will
            process each review and return an analysis.

    Returns:
        List[Dict[str, dict]]: A list of dictionaries, where each dictionary contains:
            - 'review': The original review dictionary
            - 'analysis': The result of processing the review through the chain

    Example:
        ```python
        from langchain_openai import ChatOpenAI
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate

        # Create an example chain for sentiment analysis
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Analyze the sentiment of this review: {text}"
        )
        sentiment_chain = LLMChain(llm=llm, prompt=prompt)

        reviews = [
            {"text": "Great product, highly recommended!"},
            {"text": "Terrible experience, would not buy again."}
        ]

        analyzed_reviews = analyze_reviews(reviews, sentiment_chain)
        ```

    Note:
        - Ensure the chain can handle the input dictionary structure of your reviews.
        - The function assumes the chain will return a dictionary or compatible analysis result.
    """
    # Initialize an empty list to store review analyses
    resp: List[Dict[str, dict]] = []

    async def process_reviews_async(reviews: List[dict]):
        """Asynchronously process reviews using asyncio"""
        # Use asyncio to process reviews concurrently
        tasks = []
        for review in reviews:
            tasks.append(asyncio.create_task(process_single_review(review)))
        return await asyncio.gather(*tasks)

    async def process_single_review(review: dict):
        """Process a single review asynchronously"""
        # Invoke the chain with the current review
        response = await chain.ainvoke({
            "title": review["title"],
            "positives": review["plus"],
            "negatives": review["min"],
            "content": review["content"]
        })  # Async invocation
        return {
            "review": review,
            "analysis": response,
        }

    # Run the async processing in the event loop
    try:
        # Use asyncio.run if this is the main event loop
        resp = asyncio.run(process_reviews_async(reviews.to_dict(orient='records')))
    except AttributeError:
        # Fallback to synchronous processing if async not supported
        for review in reviews:
            response = chain.invoke(review)
            resp.append({
                "review": review,
                "analysis": response,
            })

    return resp