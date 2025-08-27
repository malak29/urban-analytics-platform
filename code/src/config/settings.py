import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

BASE_DIR = Path(__file__).parent.parent.parent


@dataclass
class APIConfig:
    """NYC Open Data API configuration"""
    BASE_URL: str = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
    APP_TOKEN: str = os.getenv("NYC_OPEN_DATA_TOKEN", "")
    REQUEST_TIMEOUT: int = 30
    RATE_LIMIT_DELAY: float = 0.1
    MAX_RETRIES: int = 3
    BATCH_SIZE: int = 50000


@dataclass
class DataConfig:
    """Data processing configuration"""
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    EXTERNAL_DATA_DIR: Path = BASE_DIR / "data" / "external"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    
    # Data validation rules
    MIN_COMPLAINT_DATE: str = "2010-01-01"
    MAX_COMPLAINT_DATE: str = "2024-12-31"
    VALID_BOROUGHS: list = None
    REQUIRED_COLUMNS: list = None

    def __post_init__(self):
        self.VALID_BOROUGHS = [
            "MANHATTAN", "BROOKLYN", "QUEENS", 
            "BRONX", "STATEN ISLAND"
        ]
        self.REQUIRED_COLUMNS = [
            "unique_key", "created_date", "complaint_type",
            "incident_zip", "borough", "latitude", "longitude"
        ]


@dataclass
class VisualizationConfig:
    """Visualization settings"""
    FIGURE_SIZE: tuple = (12, 8)
    DPI: int = 300
    COLOR_PALETTE: str = "viridis"
    FONT_SIZE: int = 12
    SAVE_FORMAT: str = "png"
    
    # Map settings
    MAP_CENTER: tuple = (40.7128, -74.0060)  # NYC coordinates
    MAP_ZOOM: int = 11
    HEATMAP_RADIUS: int = 15


@dataclass
class LoggingConfig:
    """Logging configuration"""
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Path = BASE_DIR / "logs" / "noise_analytics.log"
    MAX_LOG_SIZE: int = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT: int = 5


class Settings:
    """Main settings class"""
    
    def __init__(self):
        self.api = APIConfig()
        self.data = DataConfig()
        self.visualization = VisualizationConfig()
        self.logging = LoggingConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create required directories if they don't exist"""
        directories = [
            self.data.RAW_DATA_DIR,
            self.data.PROCESSED_DATA_DIR,
            self.data.EXTERNAL_DATA_DIR,
            self.data.OUTPUT_DIR,
            self.data.OUTPUT_DIR / "reports",
            self.data.OUTPUT_DIR / "visualizations",
            self.data.OUTPUT_DIR / "exports",
            self.logging.LOG_FILE.parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_query_params(self, date_range: tuple = None) -> Dict[str, Any]:
        """Generate query parameters for API requests"""
        base_query = {
            '$select': ','.join([
                'unique_key', 'created_date', 'closed_date', 'agency',
                'agency_name', 'complaint_type', 'descriptor', 'location_type',
                'incident_zip', 'incident_address', 'street_name', 'city',
                'borough', 'latitude', 'longitude', 'status'
            ]),
            '$limit': self.api.BATCH_SIZE,
            '$offset': 0
        }
        
        if date_range:
            start_date, end_date = date_range
            base_query['$where'] = f"closed_date BETWEEN '{start_date}T00:00:00' AND '{end_date}T23:59:59'"
            
        return base_query


# Global settings instance
settings = Settings()