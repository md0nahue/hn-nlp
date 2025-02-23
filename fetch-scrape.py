import requests
import time
import os
import datetime
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# SQLite Database Setup
DATABASE_URL = "sqlite:///hackernews.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# Define Comment Model
class Comment(Base):
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    story_id = Column(Integer, index=True)
    parent_id = Column(Integer, ForeignKey("comments.id"), nullable=True)
    text = Column(Text, nullable=False)
    author = Column(String(255), nullable=True)
    timestamp = Column(DateTime, nullable=False)
    date = Column(String(10), nullable=False)  # Date in YYYY-MM-DD format

    parent = relationship("Comment", remote_side=[id], backref="replies")

# Create the database table
if not os.path.exists("hackernews_scraped.db"):
    Base.metadata.create_all(engine)

# Get top stories
TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"

def get_top_stories():
    """Fetch top story IDs from Hacker News."""
    response = requests.get(TOP_STORIES_URL)
    if response.status_code == 200:
        return response.json()[:50]  # Limit to 50 stories
    return []

def scrape_comments(story_id):
    """Scrape comments from a Hacker News story page."""
    url = f"https://news.ycombinator.com/item?id={story_id}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch {url}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    comments = []
    
    comment_divs = soup.select(".comment")  # Select all comment divs
    for div in comment_divs:
        text = div.select_one(".commtext")
        author = div.find_previous("a", class_="hnuser")  # Get username
        time_element = div.find_previous("span", class_="age")

        if not text or not time_element:
            continue

        comment_text = text.get_text()
        author_name = author.get_text() if author else "Unknown"
        
        # Convert timestamp to datetime
        timestamp_text = time_element["title"].split()[0]  # Extract only the ISO timestamp part
        timestamp = datetime.datetime.fromisoformat(timestamp_text)

        date = timestamp.strftime("%Y-%m-%d")

        comments.append(Comment(
            story_id=story_id,
            text=comment_text,
            author=author_name,
            timestamp=timestamp,
            date=date
        ))

    return comments

def scrape_and_store_comments():
    """Scrape comments from top stories and store them in the database."""
    session = SessionLocal()
    
    try:
        story_ids = get_top_stories()
        print(f"Found {len(story_ids)} top stories.")

        for story_id in story_ids:
            print(f"Scraping comments for story {story_id}...")
            comments = scrape_comments(story_id)

            if comments:
                session.add_all(comments)
                session.commit()
                print(f"Stored {len(comments)} comments for story {story_id}.")
            else:
                print(f"No comments found for story {story_id}.")

            time.sleep(1)  # Avoid getting rate-limited

    except Exception as e:
        print(f"Error: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    scrape_and_store_comments()
    print("✅ All comments scraped and saved to the database!")
