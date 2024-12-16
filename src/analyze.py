from typing import Dict, List
from langchain_core.runnables import Runnable
import pandas as pd
import asyncio

def analyze_reviews(reviews: pd.DataFrame, chain: Runnable) -> List[Dict[str, dict]]:
    resp: List[Dict[str, dict]] = []

    async def process_reviews_async(reviews: List[dict]):
        """Asynchronously process reviews using asyncio"""
        tasks = []
        for review in reviews:
            tasks.append(asyncio.create_task(process_single_review(review)))
        return await asyncio.gather(*tasks)

    async def process_single_review(review: dict):
        """Process a single review asynchronously"""
        response = await chain.ainvoke({
            "title": review["title"],
            "positives": review["plus"],
            "negatives": review["min"],
            "content": review["content"]
        }) 
        return {
            "review": review,
            "analysis": response,
        }

    try:
        resp = asyncio.run(process_reviews_async(reviews.to_dict(orient='records')))
    except AttributeError:
        for review in reviews:
            response = chain.invoke(review)
            resp.append({
                "review": review,
                "analysis": response,
            })

    return resp


def analyze_reviews_batch(reviews: pd.DataFrame, chain: Runnable) -> List[Dict[str, dict]]:
    all_data_df = reviews.to_dict()
    response = chain.invoke(all_data_df)  
    return {
            "reviews": reviews,
            "analysis": response,
    }