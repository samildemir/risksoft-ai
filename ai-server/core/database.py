import os
import logging
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.orm.scoping import ScopedSession
from sqlalchemy.exc import SQLAlchemyError

from dotenv import load_dotenv
from utils.helper import get_env

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a base class for declarative models
Base = declarative_base()


def get_database_uri() -> Optional[str]:
    """Returns the database URI after replacing the initial part if necessary."""
    url = os.getenv("POSTGRES_URI")
    if url and url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg2://", 1)
    return url


def create_database_engine(url: str) -> Engine:
    """Create and return the SQLAlchemy engine."""
    try:
        return create_engine(url)
    except SQLAlchemyError as exc:
        logger.error("Failed to create database engine: %s", exc)
        raise


def create_database_session(engine: Engine) -> ScopedSession:
    """Creates and returns a database session bound to the provided engine."""
    try:
        session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return scoped_session(session_factory)
    except SQLAlchemyError as exc:
        logger.error("Failed to create database session: %s", exc)
        raise


def initialize_database(session: Optional[ScopedSession] = None) -> None:
    """Initializes the database using the provided session or the global engine."""
    env = get_env("NODE_ENV")
    try:
        if env != "production":
            bind = session.get_bind() if session is not None else database_engine
            Base.metadata.create_all(bind=bind)
    except SQLAlchemyError as exc:
        logger.error("Failed to initialize database: %s", exc)


# Get the database URL from environment variables
database_url = get_database_uri()

# Check if the database_url is not None or empty
if not database_url:
    raise ValueError("No 'POSTGRES_URI' set in .env file")

# Create a database engine and session
database_engine: Engine = create_database_engine(database_url)
db_session: ScopedSession = create_database_session(database_engine)

# Initialize the database
initialize_database(db_session)
