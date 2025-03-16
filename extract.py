import praw 
import pandas as pd
import re 
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

nltk.download('punkt',quiet=True)
nltk.download('stopwords',quiet=True)



reddit = praw.Reddit(
    client_id="6VEps92ZxwUsTXpgAVLajA",
    client_secret="RyBD0bEu8ZuplFsBcVHDKc04krBaJQ",
    user_agent= "windows:MentalHealthScraper (by u/MaybeAltruistic7371)"
)


subreddits= ["mentalhealth", "depression", "addiction","mentalhealthsupport"]
keywords = ["depressed", "suicidal", "anxiety", "panic attack", "overwhelmed", "therapy", "self harm", "relapse","Suicide","help"]
##cleaning data so we can sent it out to nlp 
def clean_text(text):
    text=re.sub(r'http\S+|www\S+', '', text)
    text=re.sub(r'[^a-zA-Z\s]', '', text)
    text=text.lower().strip()
    # Tokenization
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]  
    return ' '.join(tokens)

##getting the posts 
posts=[]
for subreddit_name in subreddits:
    subreddit=reddit.subreddit(subreddit_name)
    #fetching top 300 posts this is due to my system is not powerful enough and takes a lot of time to fetch 
    for post in subreddit.hot(limit=300):
        if any(keyword in post.title.lower() or keyword in post.selftext.lower() for keyword in keywords):
            posts.append([
                post.id,              # Unique Post ID
                post.created_utc,     # Post creation time (UNIX timestamp)
                post.title,  
                post.selftext,
                post.score,
                post.num_comments,
                post.permalink 
            ])





df = pd.DataFrame(posts, columns=["Post_ID", "Timestamp", "Title", "Content", "Upvotes", "Comments", "Link"])
df["Cleaned_Content"] = df["Content"].apply(clean_text)


df.to_csv("cleaned_reddit_posts.csv", index=False)

print(f"Extracted {len(df)} relevant posts. Data saved to cleaned_reddit_posts.csv.")
