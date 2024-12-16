import dotenv

dotenv.load()

import streamlit as st
from src import get_data, analyze
from src.app.analyseChain_GPT import chain as GPTchain
from src.app.analyseChain_overall_GPT import chain1 as GPTchain1
from src.app.analyseChain_claude import chain as claudechain
from src.app.analyseChain_overall_claude import chain1 as claudechain1
from src.app.analyseChain_LLaMA import chain as llamachain
from src.app.analyseChain_overall_LLaMA import chain1 as llamachain1
from src.app.analyseChain_gemini import chain as geminichain
from src.app.analyseChain_overall_gemini import chain1 as geminichain1
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from collections import Counter
from datetime import datetime
from dateutil import parser
import statistics
from collections import Counter
import ast
from babel.dates import parse_date
import matplotlib.pyplot as plt
import time

chains_to_use = {
    "GPT": (GPTchain, GPTchain1),
    "Claude": (claudechain, claudechain1),
    "Llama": (llamachain, llamachain1),
    "Gemini": (geminichain, geminichain1)
}

def main():
    st.title("Coolblue Sentiment Dashboard")
    url = st.sidebar.text_input("Coolblue URL", placeholder="Provide your Coolblue URL here")

    chain = st.sidebar.selectbox(
    "What model would you like",
        ("GPT", "Claude", "Llama", "Gemini"),
        index=0
    )
    
    if st.sidebar.button("Confirm", type="primary"):
        data = get_data.get_reviews(url)
        analyzed_reviews = analyze.analyze_reviews(data, chains_to_use[chain][0])
        analyzed_reviews = transform_reviews(analyzed_reviews)
        
        most_common_emote = most_common_emotion(analyzed_reviews)
        display_emote(most_common_emote)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            result = calculate_scores(analyzed_reviews)
            st.markdown("#### Average Score")
            st.plotly_chart(create_gauge_chart(result['average']), use_container_width=True)
        
        with col2:
            st.markdown("#### Median Score")
            st.plotly_chart(create_gauge_chart(result['median']), use_container_width=True)
        
        st.markdown("## Emotion Analysis")
        
        st.markdown("### Emotion Frequency")
        plot_emotion_frequency_streamlit(analyzed_reviews)
        
        st.markdown("### Emotions Over Time")
        linechart = prepare_data_for_line_chart(extract_dates_and_emotions(analyzed_reviews))
        plot_line_chart(linechart)
        
        alldata = extract_review_data(analyzed_reviews)
        if chain == "Llama":
            time.sleep(70)
        secondanalysis_reviews = analyze.analyze_reviews_batch(alldata, chains_to_use[chain][1])
        secondanalysis_reviews = transform_second(secondanalysis_reviews)
        preprocess_columns(secondanalysis_reviews)

        with col3:
            st.markdown("#### Product Health")
            gezondratio = secondanalysis_reviews['score'].iloc[0]
            st.plotly_chart(create_gezondratio_chart(gezondratio), use_container_width=True)
        
        st.markdown("## Reviews and Market Insights")
        display_reviews(
            secondanalysis_reviews['positives'], 
            secondanalysis_reviews['negatives'], 
            secondanalysis_reviews['negative_reasoning'], 
            secondanalysis_reviews['negative_solution'], 
            secondanalysis_reviews['market_solution'], 
            secondanalysis_reviews['market_reasoning']
        )
        
def preprocess_columns(df):
    for column in ['positives', 'negatives', 'negative_reasoning', 'negative_solution', 'market_solution', 'market_reasoning']:
        df[column] = df[column].apply(lambda x: eval(x) if isinstance(x, str) else x)

def display_reviews(positives_column, negatives_column, reasoning_column, solution_column, market_reasoning_column, market_solution_column):
    st.subheader("Product Reviews")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Positive Reviews")
        for positives in positives_column:
            if isinstance(positives, list):
                for review in positives:
                    st.markdown(f"✅ {review}")

    with col2:
        st.markdown("### Negative Reviews")
        for negatives in negatives_column:
            if isinstance(negatives, list):
                for review in negatives:
                    st.markdown(f"❌ {review}")

    st.markdown("### Negative Reasoning and Solutions")
    for solution, reasoning in zip(solution_column, reasoning_column):
        st.markdown("#### Suggested Solutions:")
        if isinstance(solution, list):
            for sol in solution:
                st.markdown(f"- 🛠️ {sol}")
        st.markdown("#### Negative Reasoning:")
        if isinstance(reasoning, list):
            for reason in reasoning:
                st.markdown(f"- 💭 {reason}")

    st.markdown("### Market Insights")
    for market_reasoning, market_solution in zip(market_solution_column, market_reasoning_column):
        st.markdown("#### Market Solutions:")
        if isinstance(market_solution, list):
            for sol in market_solution:
                st.markdown(f"- 🚀 {sol}")
        st.markdown("#### Market Reasoning:")
        if isinstance(market_reasoning, list):
            for reason in market_reasoning:
                st.markdown(f"- 📊 {reason}")





def display_emote(feeling):
    """
    Displays an emote on the dashboard based on the given feeling.
    
    Parameters:
    feeling (str): The current feeling, one of ['angry', 'frustrated', 'sad', 'neutral', 'happy'].
    """
    emotes = {
        'angry': '😡',
        'frustrated': '😠',
        'sad': '😢',
        'neutral': '😐',
        'happy': '😊'
    }
    
    emote = emotes.get(feeling.lower(), '🤔') 
    
    st.subheader("Current Feeling")
    st.markdown(f"### {feeling.capitalize()} {emote}")

def create_gezondratio_chart(value):
    """
    Creates a gauge chart for a value from 0 to 100 with a transparent background.
    
    Parameters:
    value (float): The value to display on the gauge chart.
    Returns:
    fig: Plotly figure object
    """

    if value < 50:
        color = "red"
    elif 50 <= value <= 75:
        color = "orange"
    else:
        color = "green"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Percentage Gauge"},
        gauge={
            'axis': {'range': [0, 100]},  
            'bar': {'color': color},  
            'steps': [
                {'range': [0, 100], 'color': "rgba(0,0,0,0)"}  
            ]
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",  
        plot_bgcolor="rgba(0,0,0,0)"   
    )
    return fig

def create_gauge_chart(value):
    """
    Creates a gauge chart with a transparent background and dynamic slider color.
    
    Parameters:
    value (float): The value to display on the gauge chart.
    Returns:
    fig: Plotly figure object
    """

    if value < 5:
        color = "red"
    elif 5 <= value <= 7:
        color = "orange"
    else:
        color = "green"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Value Gauge"},
        gauge={
            'axis': {'range': [0, 10]},  
            'bar': {'color': color},  
            'steps': [
                {'range': [0, 10], 'color': "rgba(0,0,0,0)"}  
            ]
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",  
        plot_bgcolor="rgba(0,0,0,0)"  
    )
    
    return fig


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

    average = sum(scores) / len(scores)
    median = statistics.median(scores)

    return {
        "scores": scores,
        "average": average,
        "median": median
    }

def display_product_reviews(reviews_df):
    if 'positives' not in reviews_df.columns or 'negatives' not in reviews_df.columns:
        st.error("DataFrame must contain 'positives' and 'negatives' columns")
        return
    
    positives = ast.literal_eval(reviews_df['positives'].iloc[0])
    negatives = ast.literal_eval(reviews_df['negatives'].iloc[0])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Positives')
        for positive in positives:
            st.markdown(f"<span style='color:green'>+</span> {positive}", unsafe_allow_html=True)

    with col2:
        st.subheader('Negatives')
        for negative in negatives:
            st.markdown(f"<span style='color:red'>-</span> {negative}", unsafe_allow_html=True)


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
        analysis = item.get('analysis', None)
        if isinstance(analysis, dict) and 'emotion' in analysis:
            emotions.append(analysis['emotion'])
        else:
            print(f"Skipping invalid or missing emotion in item: {item}")

    if not emotions:
        return None  

    emotion_counts = Counter(emotions)
    most_common = emotion_counts.most_common(1)[0]  
    
    return most_common[0]  

def extract_review_data(data):
    extracted_data = []
    for item in data:
        review = item['review']
        analysis = item['analysis']

        plus_list = [p.strip() for p in review['plus'].split(';') if p.strip()] if review['plus'] else []
        min_list = [m.strip() for m in review['min'].split(';') if m.strip()] if review['min'] else []

        combined = {
            'score': review['score'],
            'title': review['title'],
            'plus_list': plus_list,  
            'min_list': min_list,    
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
    unique_reviews = {}
    
    for item in input_data:
        review_key = (
            item['review']['score'], 
            item['review']['title'], 
            item['review']['plus'], 
            item['review']['min'], 
            item['review']['content'], 
            item['review']['date']
        )
        
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
    
    transformed_data = list(unique_reviews.values())
    
    return transformed_data



def transform_second(input_data):
    full_analysis = input_data['analysis']
    
    analysis_df = pd.DataFrame({
        'positives': [full_analysis.positives],
        'negatives': [full_analysis.negatives],
        'negative_solution': [full_analysis.negative_solution],
        'negative_reasoning': [full_analysis.negative_reasoning],
        'score': [full_analysis.score],
        'score_reasoning': [full_analysis.score_reasoning],
        'market_solution': [full_analysis.Market_solution],
        'market_reasoning': [full_analysis.Market_reasoning]
    })
    
    return analysis_df

def plot_emotion_frequency_streamlit(analyzed_reviews):
    allowed_emotions = ['sad', 'neutral', 'happy', 'frustrated', 'angry']

    emotions = [entry['analysis']['emotion'] for entry in analyzed_reviews if entry['analysis']['emotion'] in allowed_emotions]
    emotion_counts = Counter(emotions)

    emotion_frequencies = {emotion: emotion_counts.get(emotion, 0) for emotion in allowed_emotions}

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(x=list(emotion_frequencies.keys()), y=list(emotion_frequencies.values()), name='Emotion Frequency'))
    bar_fig.update_layout(
        title='Frequency of Emotions',
        xaxis_title='Emotion',
        yaxis_title='Count',
        template='plotly'
    )
    
    st.plotly_chart(bar_fig)

def map_dutch_to_english_month(date_str):
    month_translation = {
        'januari': 'January', 'februari': 'February', 'maart': 'March',
        'april': 'April', 'mei': 'May', 'juni': 'June', 'juli': 'July',
        'augustus': 'August', 'september': 'September', 'oktober': 'October',
        'november': 'November', 'december': 'December'
    }
    
    for dutch, english in month_translation.items():
        if dutch in date_str:
            date_str = date_str.replace(dutch, english)
            break 
    return date_str

def extract_dates_and_emotions(mock_data):
    dates = []
    emotions = []
    for entry in mock_data:
        date = entry['review']['date']
        emotion = entry['analysis']['emotion']
        if date:
            date = map_dutch_to_english_month(date)
            dates.append(date)
            emotions.append(emotion)

    df_emotions = pd.DataFrame({
        'Date': dates,
        'Emotion': emotions
    })
    df_emotions = df_emotions.drop_duplicates()
    return df_emotions

emotion_mapping = {
    'sad': 0,
    'neutral': 1,
    'happy': 2,
    'frustrated': 3,
    'angry': 4,
}

reverse_emotion_mapping = {v: k for k, v in emotion_mapping.items()}

def prepare_data_for_line_chart(df_emotions):
    df_emotions['Date'] = pd.to_datetime(df_emotions['Date'], errors='coerce')
    
    df_emotions['Emotion_Num'] = df_emotions['Emotion'].map(emotion_mapping)
    
    df_emotions = df_emotions.sort_values(by='Date')

    return df_emotions

def plot_line_chart(df_emotions_prepared):
    df_emotions_prepared.set_index('Date', inplace=True)
    
    y_tick_vals = list(range(len(emotion_mapping)))  
    y_tick_labels = [reverse_emotion_mapping[i] for i in y_tick_vals]  
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_emotions_prepared.index,  
        y=df_emotions_prepared['Emotion_Num'], 
        mode='lines+markers',  
        marker=dict(color='blue'),
        line=dict(shape='linear')
    ))
    
    fig.update_layout(
        title='Emotions Over Time',
        xaxis=dict(
            title='Date',
            tickangle=45
        ),
        yaxis=dict(
            title='Emotion',
            tickmode='array',
            tickvals=y_tick_vals,  
            ticktext=y_tick_labels  
        ),
        margin=dict(l=40, r=40, t=40, b=40) 
    )
    
    st.plotly_chart(fig)
    
if __name__ == "__main__":
    main()
