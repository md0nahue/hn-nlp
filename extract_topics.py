import os
import spacy
from collections import Counter
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

Base = declarative_base()  # Define Base

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

# Database setup
DATABASE_URL = "sqlite:///hackernews.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

# Create the database table if it doesn't exist
if not os.path.exists("hackernews_scraped.db"):
    Base.metadata.create_all(engine)

# Fetch all comments
comments = session.query(Comment).all()

nlp = spacy.load("en_core_web_sm")  # Load NLP model

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "PRODUCT"]]

topics = []
for comment in comments:
    topics.extend(extract_entities(comment.text))  # Fix: Extract from comment.text

# Count occurrences of each topic
topic_counts = Counter(topics).most_common(30)  # Get top 10 topics
print(topic_counts)
