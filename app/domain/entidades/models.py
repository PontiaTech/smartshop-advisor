from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY
from app.database.database import Base


# definition of the User table for the DB
class UserDB(Base):
    
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    surname = Column(String, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    contact_mail = Column(String, unique=True, index=True, nullable=False)
    age = Column(Integer, index=True, nullable=True)
    
    is_active = Column(Boolean, default=True)
    date_added = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    date_deactivated = Column(DateTime(timezone=True), nullable=True)
    
    user_preferences = relationship("UserPreferencesDB", back_populates="user", uselist=False)
    user_history = relationship("UserHistoryDB", back_populates="user")
    
 
 
# definition of the UserPreferences for the DB

class UserPreferencesDB(Base):
    
    __tablename__ = "user_preferences"
    
    preference_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), unique=True, nullable=False)
    foot_size = Column(Float, nullable=True)
    shirt_size = Column(String, nullable=True)
    pants_size = Column(String, nullable=True)
    
    # this two solutions can be implemented by using a JSON field too if we would like to maki it more complex
    preferred_colours = Column(ARRAY(String), nullable=True)
    preferred_brands = Column(ARRAY(String), nullable=True)
    
    user = relationship("UserDB", back_populates="user_preferences")
    
   
# definition of the Product table for the DB

class ProductDB(Base):
    
    __tablename__ = "products"
    
    product_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    description = Column(String, nullable=True)
    price = Column(Float, nullable=False)
    brand = Column(String, index=True, nullable=True)
    colour = Column(String, index=True, nullable=True)
    size = Column(String, index=True, nullable=True)
    
    date_added = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    user_history = relationship("UserHistoryDB", back_populates="product")
    
# definition of the UserHistory table for the DB

class UserHistoryDB(Base):
    
    __tablename__ = "user_history"
    
    history_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.product_id"), nullable=False)
    query = Column(String, nullable=True)
    image_url = Column(String, nullable=True)
    shown_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    user = relationship("UserDB", back_populates="user_history")
    product = relationship("ProductDB", back_populates="product")
    
    
# definition of the chat messages and the chat for the db
    
class ChatSessionDB(Base):
    
    __tablename__ = "chat_sessions"
    
    chat_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    messages = relationship("ChatMessage", back_populates="session")
    


class ChatMessageDB(Base):
    
    __tablename__ = "chat_messages"
    
    message_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions._chat_id"))
    
    sender = Column(String, nullable=False)   # it indicates if this is going to be the user or the assictant
    content = Column(Text, nullable=True)   # this contains the query and answer together
    image_url = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    session = relationship("ChatSession", back_populates="messages")
   
   

    
    