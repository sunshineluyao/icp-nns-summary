import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from wordcloud import WordCloud
import base64
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK components are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the CSV data from the provided link
url = "https://media.githubusercontent.com/media/sunshineluyao/icp-nns-db/main/data/proposals_no_empty.csv"
df = pd.read_csv(url)

# Preprocess the data
def preprocess_text(text):
    # Check if the text is a string
    if not isinstance(text, str):
        return ''  # Return an empty string if text is not a string

    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return ' '.join(tokens)

df['processed_summary'] = df['summary'].apply(preprocess_text)

# Function to create a word cloud image
def create_wordcloud(text):
    # Generate the word cloud with higher resolution
    wordcloud = WordCloud(width=1600, height=800, background_color='white', colormap='RdBu').generate(text)

    # Save the word cloud to a buffer
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, 'png', quality=100)
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Layout of the app
app.layout = html.Div([
    html.H1("ICP NNS Proposal Summaries Word Cloud"),
    dcc.Dropdown(
        id='topic-dropdown',
        options=[{'label': topic, 'value': topic} for topic in df['topic'].unique()],
        value=df['topic'].unique()[0],
        multi=False
    ),
    html.Img(id='wordcloud-image', style={'width': '100%', 'height': 'auto'})
])

# Callback for updating the word cloud based on the selected topic
@app.callback(
    Output('wordcloud-image', 'src'),
    Input('topic-dropdown', 'value')
)
def update_wordcloud(selected_topic):
    filtered_df = df[df['topic'] == selected_topic]
    combined_text = ' '.join(filtered_df['processed_summary'])
    return create_wordcloud(combined_text)

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
