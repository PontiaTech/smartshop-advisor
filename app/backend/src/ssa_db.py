import os
import sqlmodel
from sqlmodel import Session, SQLModel
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
print(f"La URL de la base de datos obtenida es: {DATABASE_URL}")


if DATABASE_URL == "":
    raise NotImplementedError("You must set a 'DATABASE_URL'")

engine = sqlmodel.create_engine(DATABASE_URL)


def init_db():
    print("Creating the database tables ....")
    SQLModel.metadata.create_all(engine)

# api routes
def get_session():
    with Session(engine) as session:
        yield session