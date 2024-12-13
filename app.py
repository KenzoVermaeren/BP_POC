import streamlit as st
from src import get_data, analyze
from src.app.analyseChain import chain
from src.app.analyseChain_overall import chain1
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from collections import Counter
from datetime import datetime
import statistics
from collections import Counter


def main():
    # https://docs.streamlit.io/
    st.title("Coolblue sentiment dashboard")

    url = st.sidebar.text_input(
        label="Coolblue URL", placeholder="Provide your Coolblue URL here"
    )

    if st.sidebar.button("Confirm", type="primary"):
        # Get the data
        data = get_data.get_reviews(url)  # TODO
        # Analyze the data
        analyzed_reviews = analyze.analyze_reviews(data, chain)
        st.code(analyzed_reviews)
        analyzed_reviews = transform_reviews(analyzed_reviews)
        st.code(analyzed_reviews)
        # Get data for overall score
        result = calculate_scores(analyzed_reviews)
        st.code(result)
        # scores_list = result['scores']
        """average, median, most_common_emote, bar => ready for dashboard"""
        average = result['average']
        st.code(average)
        median = result['median']
        st.code(median)
        most_common_emote = most_common_emotion(analyzed_reviews)
        st.code(most_common_emote)
        # bar = plot_emotion_frequency(analyzed_reviews)
        alldata = extract_review_data(analyzed_reviews)
        st.code(alldata)
        secondanalysis_reviews = analyze.analyze_reviews(data, chain1)
        st.code(secondanalysis_reviews)


        # st.line_chart(plot_emotions_over_time)
        # st.bar_chart(plot_emotion_frequency)
        # TODO: KENZO -> Use this information to create a visual dashboard (e.v.t w/ LLM's)

def calculate_scores(data):
    """
    Extracts scores from the 'analysis' field and calculates the average and median.

    Args:
    - data (list): List of dictionaries containing an 'analysis' key with a 'score'.

    Returns:
    - dict: A dictionary containing the extracted scores, average, and median.
    """
    scores = []

    for item in data:
        # Ensure 'analysis' exists and is a dictionary with a 'score' key
        analysis = item.get('analysis', None)
        if isinstance(analysis, dict) and 'score' in analysis:
            scores.append(analysis['score'])
        else:
            print(f"Skipping invalid or missing score in item: {item}")

    if not scores:
        return {
            "scores": [],
            "average": None,
            "median": None
        }

    # Calculate average and median
    average = sum(scores) / len(scores)
    median = statistics.median(scores)

    return {
        "scores": scores,
        "average": average,
        "median": median
    }

def most_common_emotion(data):
    """
    Bepaalt de vaakst voorkomende emotie in het 'analysis'-veld van de dataset.

    Args:
    - data (list): Lijst van dictionaries met een 'analysis'-key die een 'emotion' bevat.

    Returns:
    - str: De meest voorkomende emotie, of None als er geen emoties zijn.
    """
    emotions = []

    for item in data:
        # Check of 'analysis' bestaat en een emotie bevat
        analysis = item.get('analysis', None)
        if isinstance(analysis, dict) and 'emotion' in analysis:
            emotions.append(analysis['emotion'])
        else:
            print(f"Skipping invalid or missing emotion in item: {item}")

    if not emotions:
        return None  # Geen emoties gevonden

    # Tel de frequenties en geef de vaakst voorkomende emotie terug
    emotion_counts = Counter(emotions)
    most_common = emotion_counts.most_common(1)[0]  # Tuple: (emotie, frequentie)
    
    return most_common[0]  # Alleen de emotie retourneren

def extract_review_data(data):
    extracted_data = []
    for item in data:
        review = item['review']
        analysis = item['analysis']

        # Split plus and min into lists, removing empty strings
        plus_list = [p.strip() for p in review['plus'].split(';') if p.strip()] if review['plus'] else []
        min_list = [m.strip() for m in review['min'].split(';') if m.strip()] if review['min'] else []

        combined = {
            'score': review['score'],
            'title': review['title'],
            'plus_list': plus_list,  # Now a list
            'min_list': min_list,    # Now a list
            'plus': review['plus'],
            'min': review['min'],
            'content': review['content'],
            'date': review['date'],
            'analysis_emotion': analysis['emotion'],
            'analysis_score': analysis['score']
        }
        extracted_data.append(combined)
    
    return pd.DataFrame(extracted_data)

def transform_reviews(input_data):
    # Create a dictionary to track unique reviews
    unique_reviews = {}
    
    for item in input_data:
        # Create a hashable representation of the review
        review_key = (
            item['review']['score'], 
            item['review']['title'], 
            item['review']['plus'], 
            item['review']['min'], 
            item['review']['content'], 
            item['review']['date']
        )
        
        # If this review is not already in our unique reviews, add it
        if review_key not in unique_reviews:
            unique_reviews[review_key] = {
                'review': {
                    'score': item['review']['score'],
                    'title': item['review']['title'],
                    'plus': item['review']['plus'],
                    'min': item['review']['min'],
                    'content': item['review']['content'],
                    'date': item['review']['date']
                },
                'analysis': {
                    'emotion': item['analysis'].emotion,
                    'score': item['analysis'].score
                }
            }
    
    # Convert the dictionary values to a list
    transformed_data = list(unique_reviews.values())
    
    return transformed_data

def plot_emotion_frequency(analyzed_reviews):
    """
    Creates a bar chart of emotion frequency.

    Parameters:
    analyzed_reviews (list): List of dictionaries containing review and analysis data.
    """
    # Define allowed emotions
    allowed_emotions = ['angry', 'frustrated', 'sad', 'resentment', 'happy']

    # Count frequency of emotions for bar chart, filtering only allowed emotions
    emotions = [entry['analysis']['emotion'] for entry in analyzed_reviews if entry['analysis']['emotion'] in allowed_emotions]
    emotion_counts = Counter(emotions)

    # Ensure all allowed emotions are included, even if count is zero
    for emotion in allowed_emotions:
        if emotion not in emotion_counts:
            emotion_counts[emotion] = 0

    # Bar Chart: Frequency of Emotions
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(x=list(emotion_counts.keys()), y=list(emotion_counts.values()), name='Emotion Frequency'))
    bar_fig.update_layout(
        title='Frequency of Emotions',
        xaxis_title='Emotion',
        yaxis_title='Count',
        template='plotly'
    )
    bar_fig.show()

# def plot_emotions_over_time(analyzed_reviews):
#     """
#     Returns a DataFrame of emotions over time for use in Streamlit or other plots.

#     Parameters:
#     analyzed_reviews (list): List of dictionaries containing review and analysis data.
#     """
#     # Prepare data for line chart
#     dates_emotions = []
#     for entry in analyzed_reviews:
#         date = entry['review']['date']
#         emotion = entry['analysis']['emotion']
#         if date:
#             # Convert date to a consistent format
#             parsed_date = datetime.strptime(date, '%d %B %Y')
#             dates_emotions.append((parsed_date, emotion))
    
#     # Sort by date
#     dates_emotions.sort()
#     if not dates_emotions:
#         return pd.DataFrame(columns=['Date', 'Emotion'])  # Return an empty DataFrame if no data
    
#     # Create a DataFrame
#     df = pd.DataFrame(dates_emotions, columns=['Date', 'Emotion'])
#     return df


if __name__ == "__main__":
    main()
