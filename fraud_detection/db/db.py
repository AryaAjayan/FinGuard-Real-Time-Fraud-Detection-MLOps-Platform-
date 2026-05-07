from sqlalchemy import create_engine

DATABASE_URL = "postgresql://user:password@db:5432/frauddb"

engine = create_engine(DATABASE_URL)
