import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure necessary NLTK resources are downloaded
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

class RedditMentalHealthAnalyzer:
    def __init__(self):
        """
        Initialize the analyzer with VADER sentiment analyzer and predefined risk keywords.
        """
        self.sia = SentimentIntensityAnalyzer()
        self.high_risk_keywords = [
            "don't want to be here", "end my life", "suicide", 
            "kill myself", "want to die", "no reason to live", 
            "suicidal thoughts", "i want to die", "feeling hopeless",
            "can't go on", "life is too hard"
        ]
        self.moderate_risk_keywords = [
            "lost", "struggle", "need help", "depressed", 
            "feeling down", "overwhelming", "can't cope", 
            "mental health", "anxiety", "feeling sad", 
            "difficult time", "stressed", "lonely"
        ]
    
    def get_sentiment(self, text):
        """
        Determine sentiment using VADER sentiment analyzer.
        
        Args:
            text (str): Input text
        
        Returns:
            str: Sentiment category (Positive, Negative, Neutral)
        """
        # Handle potential NaN values
        if pd.isna(text):
            return 'Neutral'
        
        score = self.sia.polarity_scores(str(text))['compound']
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    def classify_risk(self, text):
        """
        Classify risk level based on keywords.
        
        Args:
            text (str): Input text
        
        Returns:
            str: Risk level (High-Risk, Moderate Concern, Low Concern)
        """
        # Handle potential NaN values
        if pd.isna(text):
            return 'Low Concern'
        
        text_lower = str(text).lower()
        if any(phrase in text_lower for phrase in self.high_risk_keywords):
            return 'High-Risk'
        elif any(phrase in text_lower for phrase in self.moderate_risk_keywords):
            return 'Moderate Concern'
        else:
            return 'Low Concern'
    
    def analyze_reddit_posts(self, df, text_column='Cleaned_Content'):
        """
        Analyze Reddit posts and add sentiment and risk level columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of the column containing text to analyze
        
        Returns:
            pd.DataFrame: DataFrame with added sentiment and risk level columns
        """
        # Create a copy to avoid modifying the original DataFrame
        analysis_df = df.copy()
        
        # Add sentiment and risk level columns
        analysis_df['sentiment'] = analysis_df[text_column].apply(self.get_sentiment)
        analysis_df['risk_level'] = analysis_df[text_column].apply(self.classify_risk)
        
        return analysis_df
    
    def visualize_analysis(self, df):
        """
        Create visualizations of analysis results.
        
        Args:
            df (pd.DataFrame): DataFrame with analysis results
        """
        plt.figure(figsize=(15, 6))
        
        # Subplot 1: Risk Level Distribution
        plt.subplot(1, 2, 1)
        risk_counts = df['risk_level'].value_counts()
        sns.barplot(x=risk_counts.index, y=risk_counts.values, palette='coolwarm')
        plt.title('Distribution of Risk Levels', fontsize=12)
        plt.xlabel('Risk Level')
        plt.ylabel('Number of Posts')
        plt.xticks(rotation=45)
        
        # Subplot 2: Sentiment Distribution by Risk Level
        plt.subplot(1, 2, 2)
        risk_sentiment_counts = df.groupby(['risk_level', 'sentiment']).size().unstack(fill_value=0)
        risk_sentiment_counts.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='RdYlGn')
        plt.title('Sentiment Distribution by Risk Level', fontsize=12)
        plt.xlabel('Risk Level')
        plt.ylabel('Number of Posts')
        plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, df):
        """
        Generate a summary report of the analysis.
        
        Args:
            df (pd.DataFrame): DataFrame with analysis results
        
        Returns:
            dict: Summary statistics
        """
        report = {
            'total_posts': len(df),
            'risk_level_distribution': df['risk_level'].value_counts().to_dict(),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'risk_sentiment_cross_tab': pd.crosstab(df['risk_level'], df['sentiment']).to_dict()
        }
        return report

# Main execution
def main():
    # Load the CSV file
    df = pd.read_csv('cleaned_reddit_posts.csv')
    
    # Print basic information about the dataset
    print("Dataset Information:")
    print(f"Total number of posts: {len(df)}")
    print("\nColumns in the dataset:")
    print(df.columns.tolist())
    
    # Create analyzer instance
    analyzer = RedditMentalHealthAnalyzer()
    
    # Analyze posts using 'Cleaned_Content' column
    analyzed_df = analyzer.analyze_reddit_posts(df)
    
    # Visualize results
    analyzer.visualize_analysis(analyzed_df)
    
    # Generate and print report
    report = analyzer.generate_report(analyzed_df)
    print("\nAnalysis Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
    
    # Optional: Save analyzed results
    analyzed_df.to_csv('reddit_posts_analysis.csv', index=False)
    print("\nFull analysis saved to 'reddit_posts_analysis.csv'")

if __name__ == "__main__":
    main()