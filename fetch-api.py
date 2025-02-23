import requests
import datetime
import time
import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# SQLite Database Setup (File will be created automatically)
DATABASE_URL = "sqlite:///hackernews.db"

# SQLAlchemy ORM Setup
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# Define the Comment Model (Similar to Active Record)
class Comment(Base):
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    hn_comment_id = Column(Integer, unique=True, index=True)
    story_id = Column(Integer, index=True)  # The story this comment belongs to
    parent_id = Column(Integer, ForeignKey("comments.hn_comment_id"), nullable=True)  # For nested comments
    text = Column(Text, nullable=False)
    author = Column(String(255), nullable=True)
    timestamp = Column(DateTime, nullable=False)
    date = Column(String(10), nullable=False)  # Added: Store date in YYYY-MM-DD format

    parent = relationship("Comment", remote_side=[hn_comment_id], backref="replies")

# Create the Database Tables
if not os.path.exists("hackernews.db"):
    Base.metadata.create_all(engine)

# Hacker News API URLs
TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{}.json"

def get_top_stories():
    """Fetch top stories of the day from Hacker News."""
    try:
        response = requests.get(TOP_STORIES_URL)
        story_ids = response.json()[:50]  # Fetch top 50 stories
        return story_ids
    except Exception as e:
        print(f"Error fetching stories: {e}")
        return []

def fetch_comment(comment_id, story_id):
    """Recursively fetch a comment and its replies."""
    url = ITEM_URL.format(comment_id)
    comment_data = requests.get(url).json()
    
    if not comment_data or "text" not in comment_data:
        return []

    timestamp = datetime.datetime.fromtimestamp(comment_data["time"])
    comment_date = timestamp.strftime("%Y-%m-%d")  # Extract date in YYYY-MM-DD format

    comment = Comment(
        hn_comment_id=comment_data["id"],
        story_id=story_id,
        parent_id=comment_data.get("parent"),
        text=comment_data["text"],
        author=comment_data.get("by"),
        timestamp=timestamp,
        date=comment_date  # Store date separately
    )

    comments = [comment]

    # Recursively fetch nested comments
    if "kids" in comment_data:
        for reply_id in comment_data["kids"]:
            comments.extend(fetch_comment(reply_id, story_id))

    return comments

def fetch_and_store_comments():
    """Fetch all comments from top stories and store them in the SQLite database."""
    session = SessionLocal()
    
    try:
        story_ids = get_top_stories()
        print(f"Found {len(story_ids)} top stories.")

        for story_id in story_ids:
            print(f"Fetching comments for story {story_id}...")

            # Get story data
            story_data = requests.get(ITEM_URL.format(story_id)).json()
            if not story_data or "kids" not in story_data:
                continue  # Skip if no comments
            
            # Fetch all comments recursively
            all_comments = []
            for comment_id in story_data["kids"]:
                all_comments.extend(fetch_comment(comment_id, story_id))

            # Save to SQLite
            session.add_all(all_comments)
            session.commit()
            print(f"Stored {len(all_comments)} comments for story {story_id}.")

            time.sleep(1)  # Prevent rate-limiting

    except Exception as e:
        print(f"Error during fetching: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    fetch_and_store_comments()
    print("✅ All comments saved to SQLite database!")
