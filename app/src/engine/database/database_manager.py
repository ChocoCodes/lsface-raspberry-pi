from src.engine.database.feature_db import FeatureDB
from src.config.config import DB 
from pathlib import Path 

class DatabaseManager:
    @classmethod
    def load(cls, db_path: Path = DB['La Salle Database']) -> FeatureDB:
        if not db_path.exists():
            raise ValueError(f"Unknown Database at path: {db_path}")

        return FeatureDB.load(db_path)