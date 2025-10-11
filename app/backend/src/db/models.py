from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Table, Numeric, JSON
from sqlalchemy.sql import func
from typing import Optional, Literal, List 
from abc import ABC, abstractmethod
from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime, timezone


ConsentStatus = Literal["concedido", "rechazado", "retirado"]

def get_utc_now():
    return datetime.now().replace(tzinfo=timezone.utc)


class User(SQLModel, table=True):
    
    id : int | None = Field(default=None, primary_key=True)
    name : str 
    email : str 
    clothes_sizes : Optional[list[str]] = Field(default=None, sa_column=Column(JSON))
    # igual podríamos meter también una tabla de preferencias o lista para ñas marcas favoritas de cada usuario
    # consent: Optional[bool] = Field(default=None, description="None = not requested, True = given, False = denied")
    auth_credentials : str 
    created_at: datetime = Field(default_factory=get_utc_now, sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()))
    updated_at: datetime = Field(default_factory=get_utc_now)
    
    consent : Optional["UserConsent"] = Relationship(back_populates="user",sa_relationship_kwargs={"uselist": False, "cascade": "all, delete-orphan"})
    chart: Optional["Chart"] = Relationship(back_populates="user",sa_relationship_kwargs={"uselist": False, "cascade": "all, delete-orphan"})
    
    
class UserConsent(SQLModel, table=True):
    
    id : int | None = Field(default=None, primary_key=True)
    status: ConsentStatus = Field(default="concedido")
    active: bool = Field(default=True)
    date_given : datetime = Field(default_factory=get_utc_now, sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()))
    withdrawn_at : Optional[datetime] = None
    updated_at: datetime = Field(default_factory=get_utc_now)
    
    user_id: int = Field(foreign_key="user.id", unique=True, index=True)
    user : "User" = Relationship(back_populates="consent")
    

class Product(SQLModel, table=True):
    
    id : int | None = Field(default=None, primary_key=True)
    name : str
    size : str 
    price : float
    description : str
    stock : int
    url : str
    created_at: datetime = Field(default_factory=get_utc_now, sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()))
    updated_at: datetime = Field(default_factory=get_utc_now)
    
    chart_items : List["ProductChart"] = Relationship(back_populates="product")
    

class Chart(SQLModel, table=True):
    
    id : int | None = Field(default=None, primary_key=True)
    
    user : User = Relationship(back_populates="chart")
    items: List["ProductChart"] = Relationship(back_populates="chart", sa_relationship_kwargs={"cascade": "all, delete-orphan"})
    
    
class ProductChart(SQLModel, table=True):
    
    quantity: int = Field(default=1, ge=1)
    total_price: float = Field(sa_column=Column(Numeric(10, 2)))
    
    product_id: Optional[int] = Field(foreign_key="product.id", primary_key=True)
    chart_id: Optional[int] = Field(foreign_key="chart.id", primary_key=True)
    chart: "Chart" = Relationship(back_populates="items")
    product: "Product" = Relationship(back_populates="chart_items")
    
    
    

