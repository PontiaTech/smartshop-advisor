from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# los valores por defecto se pueden cambiar pero deben coincidir despues con el archivo yml donde los definamos
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_USER = os.getenv('DB_USER', 'user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
DB_NAME = os.getenv('DB_NAME', 'smartshop_db')

# Las variables habría que declararlas en github (o en un .env local) y en el YAML para mayor seguridad
SSA_DATABASE_URL = "postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}}.db"  

engine = create_engine(SSA_DATABASE_URL)
Local_Session = sessionmaker(bind=engine, autocommit=False, autoflush=False) 

Base = declarative_base()


def get_db():
    db = Local_Session()
    try:
        yield db
    finally:
        db.close()