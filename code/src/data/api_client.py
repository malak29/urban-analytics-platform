import requests
import pandas as pd
import time
import logging
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlencode
from pathlib import Path

from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class NYCDataAPIClient:
    """Production-ready client for NYC Open Data API"""
    
    def __init__(self):
        self.base_url = settings.api.BASE_URL
        self.timeout = settings.api.REQUEST_TIMEOUT
        self.rate_limit_delay = settings.api.RATE_LIMIT_DELAY
        self.max_retries = settings.api.MAX_RETRIES
        self.batch_size = settings.api.BATCH_SIZE
        
        self.session = requests.Session()
        if settings.api.APP_TOKEN:
            self.session.headers.update({
                'X-App-Token': settings.api.APP_TOKEN
            })
    
    def _make_request(self, params: Dict[str, Any]) -> List[Dict]:
        """Make API request with retry logic and rate limiting"""
        
        for attempt in range(self.max_retries):
            try:
                # Add rate limiting
                time.sleep(self.rate_limit_delay)
                
                # Build query URL
                query_url = f"{self.base_url}?{urlencode(params)}"
                logger.debug(f"Making request: {query_url}")
                
                response = self.session.get(query_url, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                logger.info(f"Fetched {len(data)} records (attempt {attempt + 1})")
                
                return data
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt == self.max_retries - 1:
                    logger.error("All retry attempts failed for request")
                    raise
                
                # Exponential backoff
                time.sleep(2 ** attempt)
        
        return []
    
    def fetch_noise_complaints(self, 
                             date_range: Optional[Tuple[str, str]] = None,
                             complaint_types: Optional[List[str]] = None,
                             boroughs: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch noise complaint data with filters
        
        Args:
            date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format
            complaint_types: List of complaint types to filter by
            boroughs: List of boroughs to filter by
            
        Returns:
            DataFrame containing noise complaint data
        """
        
        logger.info("Starting data fetch operation")
        
        # Build base query parameters
        base_params = self._build_query_params(date_range, complaint_types, boroughs)
        
        all_data = []
        offset = 0
        
        while True:
            # Update offset for pagination
            params = base_params.copy()
            params['$offset'] = offset
            
            logger.info(f"Fetching batch: offset {offset}")
            
            # Make API request
            batch_data = self._make_request(params)
            
            # Check if we got any data
            if not batch_data:
                logger.info("No more data to fetch")
                break
            
            all_data.extend(batch_data)
            
            # Check if we got less than batch size (last page)
            if len(batch_data) < self.batch_size:
                logger.info("Reached last page of data")
                break
                
            offset += self.batch_size
            
            # Safety check to prevent infinite loops
            if len(all_data) > 5_000_000:  # 5M records limit
                logger.warning("Reached maximum data limit, stopping fetch")
                break
        
        logger.info(f"Fetch complete: {len(all_data)} total records")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        if df.empty:
            logger.warning("No data returned from API")
            return df
        
        # Basic data type conversions
        df = self._process_dataframe(df)
        
        return df
    
    def _build_query_params(self, 
                          date_range: Optional[Tuple[str, str]],
                          complaint_types: Optional[List[str]],
                          boroughs: Optional[List[str]]) -> Dict[str, Any]:
        """Build query parameters for API request"""
        
        params = {
            '$select': ','.join([
                'unique_key', 'created_date', 'closed_date', 'agency',
                'agency_name', 'complaint_type', 'descriptor', 'location_type',
                'incident_zip', 'incident_address', 'street_name', 'city',
                'borough', 'latitude', 'longitude', 'status',
                'resolution_description', 'community_board'
            ]),
            '$limit': self.batch_size,
            '$order': 'created_date DESC'
        }
        
        # Build WHERE clause
        where_conditions = []
        
        # Date filter
        if date_range:
            start_date, end_date = date_range
            where_conditions.append(
                f"created_date BETWEEN '{start_date}T00:00:00' AND '{end_date}T23:59:59'"
            )
        
        # Complaint type filter (focus on noise-related)
        noise_types = complaint_types or [
            "Noise - Residential", "Noise - Street/Sidewalk", "Noise - Commercial",
            "Noise - Vehicle", "Noise - Helicopter", "Illegal Parking"
        ]
        
        if noise_types:
            complaint_filter = " OR ".join([f"complaint_type = '{ct}'" for ct in noise_types])
            where_conditions.append(f"({complaint_filter})")
        
        # Borough filter
        if boroughs:
            borough_filter = " OR ".join([f"borough = '{b.upper()}'" for b in boroughs])
            where_conditions.append(f"({borough_filter})")
        
        if where_conditions:
            params['$where'] = ' AND '.join(where_conditions)
        
        return params
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic DataFrame processing and type conversion"""
        
        # Convert date columns
        date_columns = ['created_date', 'closed_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert coordinates to numeric
        coord_columns = ['latitude', 'longitude']
        for col in coord_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean text columns
        text_columns = ['complaint_type', 'borough', 'agency_name']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        
        logger.info(f"DataFrame processed: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def health_check(self) -> bool:
        """Check API availability"""
        try:
            test_params = {
                '$select': 'unique_key',
                '$limit': 1
            }
            
            response = self.session.get(
                f"{self.base_url}?{urlencode(test_params)}", 
                timeout=5
            )
            response.raise_for_status()
            
            logger.info("API health check passed")
            return True
            
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False


class DataCache:
    """Simple file-based caching for API responses"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or settings.data.RAW_DATA_DIR
        self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def get_cache_key(self, params: Dict[str, Any]) -> str:
        """Generate cache key from parameters"""
        # Sort params for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True)
        return hashlib.md5(sorted_params.encode()).hexdigest()
    
    def get(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data"""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        if cache_file.exists():
            try:
                logger.info(f"Loading data from cache: {cache_key}")
                return pd.read_parquet(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_key}: {e}")
        
        return None
    
    def set(self, cache_key: str, df: pd.DataFrame) -> None:
        """Store data in cache"""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        try:
            df.to_parquet(cache_file)
            logger.info(f"Data cached: {cache_key}")
        except Exception as e:
            logger.error(f"Failed to cache data {cache_key}: {e}")


def get_api_client() -> NYCDataAPIClient:
    """Factory function to get API client instance"""
    return NYCDataAPIClient()