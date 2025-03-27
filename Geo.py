import pandas as pd
import numpy as np
import re
import spacy
import folium
from folium import plugins  # Explicitly import plugins
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

class LocationCrisisAnalyzer:
    def __init__(self):
        """
        Initialize the location crisis analyzer with NLP and geocoding tools.
        """
        # Load spaCy's English model for named entity recognition
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy English model. This may take a few minutes.")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize geocoder
        self.geolocator = Nominatim(user_agent="mental_health_crisis_analyzer")
        
        # Crisis-related keywords
        self.crisis_keywords = [
            'suicide', 'depression', 'anxiety', 'mental health', 
            'struggling', 'help', 'crisis', 'overwhelmed', 
            'hopeless', 'need support'
        ]
    
    def extract_locations_from_text(self, text):
        """
        Extract location names from text using spaCy's NER.
        
        Args:
            text (str): Input text to extract locations from
        
        Returns:
            list: Extracted location names
        """
        if pd.isna(text):
            return []
        
        # Process the text with spaCy
        doc = self.nlp(str(text))
        
        # Extract GPE (Geo-Political Entities) and their labels
        locations = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
        
        return locations
    
    def geocode_location(self, location):
        """
        Convert location name to latitude and longitude.
        
        Args:
            location (str): Location name
        
        Returns:
            tuple: (latitude, longitude) or None if geocoding fails
        """
        try:
            # Add state abbreviation to help with disambiguation
            location_query = f"{location}, USA"
            
            # Geocode with timeout
            geo_location = self.geolocator.geocode(location_query, timeout=10)
            
            if geo_location:
                return (geo_location.latitude, geo_location.longitude)
        except (GeocoderTimedOut, GeocoderServiceError):
            print(f"Geocoding failed for {location}")
        
        return None
    
    def is_crisis_post(self, text):
        """
        Determine if a post is crisis-related based on keywords.
        
        Args:
            text (str): Post text
        
        Returns:
            bool: True if crisis-related, False otherwise
        """
        if pd.isna(text):
            return False
        
        text_lower = str(text).lower()
        return any(keyword in text_lower for keyword in self.crisis_keywords)
    
    def analyze_location_crisis(self, df, text_column='Cleaned_Content'):
        """
        Analyze location-based crisis discussions.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Column containing post text
        
        Returns:
            pd.DataFrame: DataFrame with location and crisis analysis
        """
        # Create a copy of the DataFrame
        analysis_df = df.copy()
        
        # Extract locations and mark crisis posts
        analysis_df['extracted_locations'] = analysis_df[text_column].apply(self.extract_locations_from_text)
        analysis_df['is_crisis_post'] = analysis_df[text_column].apply(self.is_crisis_post)
        
        # Geocode locations
        def process_locations(locations):
            geocoded_locations = []
            for loc in locations:
                coords = self.geocode_location(loc)
                if coords:
                    geocoded_locations.append(coords)
            return geocoded_locations
        
        analysis_df['geocoded_locations'] = analysis_df['extracted_locations'].apply(process_locations)
        
        return analysis_df
    
    def create_crisis_heatmap(self, location_df):
        """
        Create a heatmap of crisis discussions.
        
        Args:
            location_df (pd.DataFrame): DataFrame with geocoded locations and crisis posts
        
        Returns:
            folium.Map: Heatmap of crisis discussions
        """
        # Collect crisis post locations
        crisis_locations = []
        for locations in location_df[location_df['is_crisis_post']]['geocoded_locations']:
            crisis_locations.extend(locations)
        
        # Create base map centered on US
        crisis_map = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
        
        # Add heatmap layer
        if crisis_locations:
            plugins.HeatMap(crisis_locations).add_to(crisis_map)
        
        # Save the map
        crisis_map.save('crisis_heatmap.html')
        
        return crisis_map
    
    def analyze_top_crisis_locations(self, location_df):
        """
        Analyze and return top locations with crisis discussions.
        
        Args:
            location_df (pd.DataFrame): DataFrame with location analysis
        
        Returns:
            pd.DataFrame: Top 5 locations with crisis discussions
        """
        # Count crisis posts by location
        location_crisis_counts = {}
        for index, row in location_df[location_df['is_crisis_post']].iterrows():
            for loc in row['extracted_locations']:
                location_crisis_counts[loc] = location_crisis_counts.get(loc, 0) + 1
        
        # Convert to DataFrame and sort
        top_locations = pd.DataFrame.from_dict(location_crisis_counts, orient='index', columns=['crisis_post_count'])
        top_locations = top_locations.sort_values('crisis_post_count', ascending=False).head(5)
        
        return top_locations

# Main execution
def main():
    # Load the CSV file
    df = pd.read_csv('cleaned_reddit_posts.csv')
    
    # Create analyzer instance
    analyzer = LocationCrisisAnalyzer()
    
    # Analyze locations and crisis posts
    location_analysis_df = analyzer.analyze_location_crisis(df)
    
    # Create crisis heatmap
    crisis_heatmap = analyzer.create_crisis_heatmap(location_analysis_df)
    
    # Get top crisis locations
    top_crisis_locations = analyzer.analyze_top_crisis_locations(location_analysis_df)
    
    # Print results
    print("\nTop 5 Locations with Crisis Discussions:")
    print(top_crisis_locations)
    
    print("\nHeatmap saved as 'crisis_heatmap.html'")
    
    # Optional: Save location analysis results
    location_analysis_df.to_csv('location_crisis_analysis.csv', index=False)

if __name__ == "__main__":
    main()