import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Production-grade data validation engine"""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
        self.validation_results = {}
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules from configuration"""
        
        return {
            'required_columns': settings.data.REQUIRED_COLUMNS,
            'valid_boroughs': settings.data.VALID_BOROUGHS,
            'date_range': {
                'min_date': pd.to_datetime(settings.data.MIN_COMPLAINT_DATE),
                'max_date': pd.to_datetime(settings.data.MAX_COMPLAINT_DATE)
            },
            'coordinate_bounds': {
                'lat_min': 40.4774, 'lat_max': 40.9176,  # NYC bounds
                'lon_min': -74.2591, 'lon_max': -73.7004
            },
            'zip_code_pattern': r'^\d{5}$'
        }
    
    def validate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive dataset validation
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validated DataFrame with quality metrics
        """
        
        logger.info(f"Starting validation for {len(df)} records")
        
        initial_count = len(df)
        self.validation_results = {'initial_count': initial_count}
        
        # Run validation checks
        df = self._validate_schema(df)
        df = self._validate_data_types(df)
        df = self._validate_business_rules(df)
        df = self._validate_coordinates(df)
        df = self._validate_temporal_data(df)
        
        final_count = len(df)
        self.validation_results['final_count'] = final_count
        self.validation_results['records_dropped'] = initial_count - final_count
        self.validation_results['data_quality_score'] = self._calculate_quality_score(df)
        
        logger.info(f"Validation complete: {final_count} valid records "
                   f"({self.validation_results['records_dropped']} dropped)")
        
        return df
    
    def _validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate DataFrame schema and required columns"""
        
        missing_columns = set(self.validation_rules['required_columns']) - set(df.columns)
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            
            # Add missing columns with NaN values
            for col in missing_columns:
                df[col] = np.nan
        
        self.validation_results['schema_valid'] = len(missing_columns) == 0
        return df
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and correct data types"""
        
        logger.debug("Validating data types")
        
        # Date columns
        date_columns = ['created_date', 'closed_date']
        for col in date_columns:
            if col in df.columns:
                invalid_dates = pd.isna(pd.to_datetime(df[col], errors='coerce'))
                if invalid_dates.any():
                    logger.warning(f"Found {invalid_dates.sum()} invalid dates in {col}")
        
        # Numeric columns
        numeric_columns = ['latitude', 'longitude', 'unique_key']
        for col in numeric_columns:
            if col in df.columns:
                original_values = len(df[col].dropna())
                df[col] = pd.to_numeric(df[col], errors='coerce')
                final_values = len(df[col].dropna())
                
                if original_values != final_values:
                    logger.warning(f"Converted {original_values - final_values} invalid numeric values in {col}")
        
        return df
    
    def _validate_business_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate business logic and data integrity"""
        
        logger.debug("Validating business rules")
        
        initial_count = len(df)
        
        # Valid borough names
        if 'borough' in df.columns:
            valid_boroughs = df['borough'].isin(self.validation_rules['valid_boroughs'])
            invalid_borough_count = (~valid_boroughs).sum()
            
            if invalid_borough_count > 0:
                logger.warning(f"Found {invalid_borough_count} records with invalid borough names")
                df = df[valid_boroughs]
        
        # Remove records without unique key
        if 'unique_key' in df.columns:
            df = df.dropna(subset=['unique_key'])
        
        dropped_count = initial_count - len(df)
        if dropped_count > 0:
            logger.info(f"Dropped {dropped_count} records due to business rule violations")
        
        return df
    
    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate geographic coordinates"""
        
        if not all(col in df.columns for col in ['latitude', 'longitude']):
            logger.warning("Coordinate columns not found, skipping coordinate validation")
            return df
        
        logger.debug("Validating coordinates")
        
        bounds = self.validation_rules['coordinate_bounds']
        
        # Check coordinate bounds
        valid_coords = (
            (df['latitude'].between(bounds['lat_min'], bounds['lat_max'])) &
            (df['longitude'].between(bounds['lon_min'], bounds['lon_max']))
        )
        
        invalid_coord_count = (~valid_coords & df[['latitude', 'longitude']].notna().all(axis=1)).sum()
        
        if invalid_coord_count > 0:
            logger.warning(f"Found {invalid_coord_count} records with coordinates outside NYC bounds")
            
            # Flag invalid coordinates instead of dropping
            df.loc[~valid_coords, ['latitude', 'longitude']] = np.nan
        
        self.validation_results['invalid_coordinates'] = invalid_coord_count
        return df
    
    def _validate_temporal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate temporal data integrity"""
        
        if 'created_date' not in df.columns:
            logger.warning("Created date column not found, skipping temporal validation")
            return df
        
        logger.debug("Validating temporal data")
        
        initial_count = len(df)
        
        # Remove records with created_date outside valid range
        date_bounds = self.validation_rules['date_range']
        valid_created_dates = df['created_date'].between(
            date_bounds['min_date'], 
            date_bounds['max_date']
        )
        
        df = df[valid_created_dates | df['created_date'].isna()]
        
        # Check for logical inconsistencies (closed before created)
        if 'closed_date' in df.columns:
            logical_dates = (
                df['closed_date'].isna() | 
                (df['closed_date'] >= df['created_date'])
            )
            
            illogical_count = (~logical_dates).sum()
            if illogical_count > 0:
                logger.warning(f"Found {illogical_count} records with closed_date before created_date")
                df.loc[~logical_dates, 'closed_date'] = np.nan
        
        dropped_count = initial_count - len(df)
        if dropped_count > 0:
            logger.info(f"Dropped {dropped_count} records due to invalid dates")
        
        return df
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-100)"""
        
        if df.empty:
            return 0.0
        
        scores = []
        
        # Completeness scores for critical fields
        critical_fields = ['complaint_type', 'borough', 'created_date']
        for field in critical_fields:
            if field in df.columns:
                completeness = (1 - df[field].isna().mean()) * 100
                scores.append(completeness)
        
        # Coordinate completeness
        if all(col in df.columns for col in ['latitude', 'longitude']):
            coord_completeness = (1 - df[['latitude', 'longitude']].isna().any(axis=1).mean()) * 100
            scores.append(coord_completeness)
        
        # Valid ZIP code percentage
        if 'incident_zip' in df.columns:
            valid_zip_pct = df['incident_zip'].str.match(r'^\d{5}$', na=False).mean() * 100
            scores.append(valid_zip_pct)
        
        # Consistency score (no duplicates)
        if 'unique_key' in df.columns:
            uniqueness = (1 - df['unique_key'].duplicated().mean()) * 100
            scores.append(uniqueness)
        
        overall_score = sum(scores) / len(scores) if scores else 0
        return round(overall_score, 2)
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Generate validation report"""
        
        return {
            'validation_timestamp': datetime.now().isoformat(),
            'validation_results': self.validation_results,
            'quality_checks': {
                'schema_validation': self.validation_results.get('schema_valid', False),
                'coordinate_validation': True,
                'temporal_validation': True,
                'business_rules': True
            }
        }