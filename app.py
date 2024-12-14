import streamlit as st
from src import get_data, analyze
from src.app.analyseChain import chain
from src.app.analyseChain_overall import chain1
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

month_translation = {
    "januari": "January", "februari": "February", "maart": "March", "april": "April",
    "mei": "May", "juni": "June", "juli": "July", "augustus": "August", 
    "september": "September", "oktober": "October", "november": "November", "december": "December"
}


def main():
    st.title("Coolblue Sentiment Dashboard")
    url = st.sidebar.text_input("Coolblue URL", placeholder="Provide your Coolblue URL here")

    if st.sidebar.button("Confirm", type="primary"):
        # Get the data
        data = get_data.get_reviews(url)
        analyzed_reviews = analyze.analyze_reviews(data, chain)
        analyzed_reviews = transform_reviews(analyzed_reviews)
        
        # Current Feeling - Full Width
        most_common_emote = most_common_emotion(analyzed_reviews)
        display_emote(most_common_emote)
        
        # Mean, Average, and Health Ratio in Three Columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Average score gauge
            result = calculate_scores(analyzed_reviews)
            st.markdown("#### Average Score")
            st.plotly_chart(create_gauge_chart(result['average']), use_container_width=True)
        
        with col2:
            # Median score gauge
            st.markdown("#### Median Score")
            st.plotly_chart(create_gauge_chart(result['median']), use_container_width=True)
        
        # Emotion Analysis Section
        st.markdown("## Emotion Analysis")
        
        # Bar Chart: Emotion Frequency (Full Width)
        st.markdown("### Emotion Frequency")
        plot_emotion_frequency_streamlit(analyzed_reviews)
        
        # Line Chart: Emotions Over Time (Full Width)
        st.markdown("### Emotions Over Time")
        linechart = prepare_data_for_line_chart(extract_dates_and_emotions(analyzed_reviews))
        plot_line_chart(linechart)
        
        # Extract and analyze additional review data
        alldata = extract_review_data(analyzed_reviews)
        secondanalysis_reviews = analyze.analyze_reviews_batch(alldata, chain1)
        secondanalysis_reviews = transform_second(secondanalysis_reviews)
        preprocess_columns(secondanalysis_reviews)

        with col3:
            # Health Ratio
            st.markdown("#### Product Health")
            gezondratio = secondanalysis_reviews['score'].iloc[0]
            st.plotly_chart(create_gezondratio_chart(gezondratio), use_container_width=True)
        
        # Reviews and Insights Section
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

    # Create two columns for positives and negatives
    col1, col2 = st.columns(2)

    # Positive Reviews in the first column
    with col1:
        st.markdown("### Positive Reviews")
        for positives in positives_column:
            if isinstance(positives, list):
                for review in positives:
                    st.markdown(f"✅ {review}")

    # Negative Reviews in the second column
    with col2:
        st.markdown("### Negative Reviews")
        for negatives in negatives_column:
            if isinstance(negatives, list):
                for review in negatives:
                    st.markdown(f"❌ {review}")

    # Add reasoning and solution below the lists
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

    # Add market reasoning and solutions
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
    feeling (str): The current feeling, one of ['angry', 'frustrated', 'sad', 'resentment', 'happy'].
    """
    # Mapping feelings to emotes
    emotes = {
        'angry': '😡',
        'frustrated': '😠',
        'sad': '😢',
        'resentment': '😤',
        'happy': '😊'
    }
    
    # Default emote if the feeling is not recognized
    emote = emotes.get(feeling.lower(), '🤔')  # Use '🤔' for unrecognized feelings
    
    # Display the feeling and the emote
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

    # Set color based on value
    if value < 50:
        color = "red"
    elif 50 <= value <= 75:
        color = "orange"
    else:
        color = "green"
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Percentage Gauge"},
        gauge={
            'axis': {'range': [0, 100]},  # Range is now 0 to 100
            'bar': {'color': color},  # The bar color
            'steps': [
                {'range': [0, 100], 'color': "rgba(0,0,0,0)"}  # Transparent background
            ]
        }
    ))
    
    # Set transparent layout
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        plot_bgcolor="rgba(0,0,0,0)"   # Transparent plot area
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

    # Set color based on value
    if value < 5:
        color = "red"
    elif 5 <= value <= 7:
        color = "orange"
    else:
        color = "green"
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Value Gauge"},
        gauge={
            'axis': {'range': [0, 10]},  # Set the range of the gauge
            'bar': {'color': color},  # The bar color
            'steps': [
                {'range': [0, 10], 'color': "rgba(0,0,0,0)"}  # Transparent background
            ]
        }
    ))
    
    # Set transparent layout
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        plot_bgcolor="rgba(0,0,0,0)"   # Transparent plot area
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

def display_product_reviews(reviews_df):
    """
    Display product reviews with green '+' for positives and red '-' for negatives.
    
    Parameters:
    -----------
    reviews_df : pandas.DataFrame
        DataFrame containing 'positives' and 'negatives' columns
    """
    # Validate input
    if 'positives' not in reviews_df.columns or 'negatives' not in reviews_df.columns:
        st.error("DataFrame must contain 'positives' and 'negatives' columns")
        return
    
    # Parse positives and negatives using ast.literal_eval
    positives = ast.literal_eval(reviews_df['positives'].iloc[0])
    negatives = ast.literal_eval(reviews_df['negatives'].iloc[0])

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    # Display Positives
    with col1:
        st.subheader('Positives')
        for positive in positives:
            st.markdown(f"<span style='color:green'>+</span> {positive}", unsafe_allow_html=True)

    # Display Negatives
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



def transform_second(input_data):
    """
    Convert full analysis from the LangChain output to a DataFrame.
    
    Args:
        input_data (dict): Dictionary containing reviews DataFrame and overall analysis
    
    Returns:
        pd.DataFrame: DataFrame with analysis values
    """
    # Extract full analysis from the Output class
    full_analysis = input_data['analysis']
    
    # Create a DataFrame
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
    """
    Displays a bar chart of emotion frequency on a Streamlit dashboard, 
    with emotions displayed in the order: sad → happy → frustrated → angry → resentment.

    Parameters:
    analyzed_reviews (list): List of dictionaries containing review and analysis data.
    """
    # Define allowed emotions in the desired order
    allowed_emotions = ['sad', 'happy', 'frustrated', 'angry', 'resentment']

    # Count frequency of emotions for bar chart, filtering only allowed emotions
    emotions = [entry['analysis']['emotion'] for entry in analyzed_reviews if entry['analysis']['emotion'] in allowed_emotions]
    emotion_counts = Counter(emotions)

    # Ensure all allowed emotions are included, even if count is zero
    emotion_frequencies = {emotion: emotion_counts.get(emotion, 0) for emotion in allowed_emotions}

    # Bar Chart: Frequency of Emotions
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(x=list(emotion_frequencies.keys()), y=list(emotion_frequencies.values()), name='Emotion Frequency'))
    bar_fig.update_layout(
        title='Frequency of Emotions',
        xaxis_title='Emotion',
        yaxis_title='Count',
        template='plotly'
    )
    
    # Display the chart on the Streamlit dashboard
    st.plotly_chart(bar_fig)

# Function to map Dutch month names to English
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
            break  # Only replace the first found month
    return date_str

# Function to extract dates and emotions, with duplicates removed
def extract_dates_and_emotions(mock_data):
    dates = []
    emotions = []
    for entry in mock_data:
        date = entry['review']['date']
        emotion = entry['analysis']['emotion']
        if date:
            # Handle Dutch month format by translating
            date = map_dutch_to_english_month(date)
            dates.append(date)
            emotions.append(emotion)

    df_emotions = pd.DataFrame({
        'Date': dates,
        'Emotion': emotions
    })
    df_emotions = df_emotions.drop_duplicates()
    return df_emotions

# Map emotions to numerical values (for y-axis)
emotion_mapping = {
    'sad': 0,
    'happy': 1,
    'frustrated': 2,
    'angry': 3,
    'resentment': 4
}

# Reverse the emotion mapping to display emotions on y-axis
reverse_emotion_mapping = {v: k for k, v in emotion_mapping.items()}

# Function to prepare the data for the line chart with emotions on y-axis
def prepare_data_for_line_chart(df_emotions):
    # Convert the 'Date' column to datetime
    df_emotions['Date'] = pd.to_datetime(df_emotions['Date'], errors='coerce')
    
    # Map emotions to numerical values
    df_emotions['Emotion_Num'] = df_emotions['Emotion'].map(emotion_mapping)
    
    # Sort emotions based on date
    df_emotions = df_emotions.sort_values(by='Date')

    return df_emotions

# Function to plot the line chart with actual emotions on y-axis
def plot_line_chart(df_emotions_prepared):
    # Set the 'Date' column as index
    df_emotions_prepared.set_index('Date', inplace=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the line chart
    ax.plot(df_emotions_prepared.index, df_emotions_prepared['Emotion_Num'], marker='o', linestyle='-', color='b')
    
    # Set y-ticks to emotions instead of numbers
    ax.set_yticks(range(len(emotion_mapping)))  # Set y-ticks for each emotion
    ax.set_yticklabels([reverse_emotion_mapping[i] for i in range(len(emotion_mapping))])  # Map numbers to emotions
    
    # Set the labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Emotion')
    ax.set_title('Emotions Over Time')

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45)
    
    # Display the plot
    st.pyplot(fig)
if __name__ == "__main__":
    main()
