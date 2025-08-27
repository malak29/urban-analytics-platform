import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from ..config.settings import settings
from ..utils.logger import get_logger
from .validators import DataValidator

logger = get_logger(__name__)


class NoiseDataProcessor:
    """Core data processing engine for noise complaint analysis"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.processed_data: Optional[pd.DataFrame] = None
    
    def process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw noise complaint data through complete pipeline
        
        Args:
            df: Raw DataFrame from NYC Open Data
            
        Returns:
            Processed and validated DataFrame
        """
        
        logger.info(f"Starting data processing pipeline for {len(df)} records")
        
        # Step 1: Data validation and cleaning
        df = self._clean_data(df)
        
        # Step 2: Feature engineering
        df = self._engineer_features(df)
        
        # Step 3: Data validation
        df = self.validator.validate_dataset(df)
        
        # Step 4: Statistical calculations
        df = self._calculate_metrics(df)
        
        self.processed_data = df
        logger.info(f"Processing complete: {len(df)} valid records")
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize raw data"""
        
        logger.info("Cleaning raw data")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['unique_key'])
        logger.info(f"Removed {initial_count - len(df)} duplicate records")
        
        # Clean text fields
        text_columns = ['complaint_type', 'borough', 'agency_name', 'descriptor']
        for col in text_columns:
            if col in df.columns:
                df[col] = (df[col]
                          .astype(str)
                          .str.strip()
                          .str.upper()
                          .replace('NAN', np.nan))
        
        # Convert coordinates
        coord_columns = ['latitude', 'longitude']
        for col in coord_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean zip codes
        if 'incident_zip' in df.columns:
            df['incident_zip'] = (df['incident_zip']
                                 .astype(str)
                                 .str.replace(r'[^0-9]', '', regex=True)
                                 .str.slice(0, 5))
            df.loc[df['incident_zip'].str.len() != 5, 'incident_zip'] = np.nan
        
        # Convert dates
        date_columns = ['created_date', 'closed_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        logger.info("Data cleaning completed")
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for analysis"""
        
        logger.info("Engineering features")
        
        # Time-based features
        if 'created_date' in df.columns:
            df['hour'] = df['created_date'].dt.hour
            df['day_of_week'] = df['created_date'].dt.day_name()
            df['month'] = df['created_date'].dt.month
            df['year'] = df['created_date'].dt.year
            df['week_of_year'] = df['created_date'].dt.isocalendar().week
            df['is_weekend'] = df['created_date'].dt.weekday >= 5
        
        # Response time calculation
        if all(col in df.columns for col in ['created_date', 'closed_date']):
            df['response_time_hours'] = (
                (df['closed_date'] - df['created_date']).dt.total_seconds() / 3600
            )
            df['response_time_days'] = df['response_time_hours'] / 24
        
        # Complaint categorization
        if 'complaint_type' in df.columns:
            df['noise_category'] = df['complaint_type'].apply(self._categorize_noise_type)
            df['is_noise_complaint'] = df['complaint_type'].str.contains('NOISE', na=False)
        
        # Geographic features
        if all(col in df.columns for col in ['latitude', 'longitude']):
            df['has_coordinates'] = ~(df['latitude'].isna() | df['longitude'].isna())
        
        # Status categorization
        if 'status' in df.columns:
            df['is_resolved'] = df['status'].isin(['CLOSED', 'RESOLVED'])
        
        logger.info("Feature engineering completed")
        return df
    
    def _categorize_noise_type(self, complaint_type: str) -> str:
        """Categorize noise complaints into main types"""
        
        if pd.isna(complaint_type):
            return 'UNKNOWN'
        
        complaint_type = str(complaint_type).upper()
        
        if 'VEHICLE' in complaint_type or 'CAR' in complaint_type:
            return 'VEHICLE'
        elif 'RESIDENTIAL' in complaint_type or 'NEIGHBOR' in complaint_type:
            return 'RESIDENTIAL'
        elif 'COMMERCIAL' in complaint_type or 'BUSINESS' in complaint_type:
            return 'COMMERCIAL'
        elif 'HELICOPTER' in complaint_type or 'AIRCRAFT' in complaint_type:
            return 'AIRCRAFT'
        elif 'CONSTRUCTION' in complaint_type or 'BUILDING' in complaint_type:
            return 'CONSTRUCTION'
        elif 'STREET' in complaint_type or 'SIDEWALK' in complaint_type:
            return 'STREET'
        else:
            return 'OTHER'
    
    def _calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical metrics and KPIs"""
        
        logger.info("Calculating metrics")
        
        # Add complaint density by zip code
        if 'incident_zip' in df.columns:
            zip_counts = df['incident_zip'].value_counts()
            df['zip_complaint_density'] = df['incident_zip'].map(zip_counts)
        
        # Add borough complaint rankings
        if 'borough' in df.columns:
            borough_counts = df['borough'].value_counts()
            borough_rank = {borough: rank for rank, borough in enumerate(borough_counts.index, 1)}
            df['borough_complaint_rank'] = df['borough'].map(borough_rank)
        
        # Add temporal metrics
        if 'hour' in df.columns:
            hour_counts = df['hour'].value_counts()
            df['hour_complaint_density'] = df['hour'].map(hour_counts)
        
        logger.info("Metric calculations completed")
        return df
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        
        if self.processed_data is None:
            raise ValueError("No processed data available. Run process_raw_data() first.")
        
        df = self.processed_data
        
        stats = {
            'total_complaints': len(df),
            'date_range': {
                'start': df['created_date'].min().strftime('%Y-%m-%d') if not df['created_date'].isna().all() else None,
                'end': df['created_date'].max().strftime('%Y-%m-%d') if not df['created_date'].isna().all() else None
            },
            'complaint_types': df['complaint_type'].value_counts().head(10).to_dict(),
            'borough_distribution': df['borough'].value_counts().to_dict(),
            'hourly_distribution': df['hour'].value_counts().sort_index().to_dict(),
            'response_times': {
                'mean_hours': df['response_time_hours'].mean() if 'response_time_hours' in df.columns else None,
                'median_hours': df['response_time_hours'].median() if 'response_time_hours' in df.columns else None
            },
            'data_quality': {
                'missing_coordinates': df[['latitude', 'longitude']].isna().any(axis=1).sum(),
                'missing_zip_codes': df['incident_zip'].isna().sum(),
                'completion_rate': (df['status'] == 'CLOSED').mean() * 100 if 'status' in df.columns else None
            }
        }
        
        logger.info("Summary statistics generated")
        return stats
    
    def export_processed_data(self, filename: Optional[str] = None) -> str:
        """Export processed data to file"""
        
        if self.processed_data is None:
            raise ValueError("No processed data available")
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"processed_noise_data_{timestamp}.parquet"
        
        output_path = settings.data.PROCESSED_DATA_DIR / filename
        self.processed_data.to_parquet(output_path)
        
        logger.info(f"Processed data exported to {output_path}")
        return str(output_path)


class AnalyticsEngine:
    """High-level analytics and insights generation"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def generate_hotspot_analysis(self) -> Dict[str, Any]:
        """Identify geographic hotspots for complaints"""
        
        # Top complaint zip codes
        top_zips = self.df['incident_zip'].value_counts().head(10)
        
        # Borough analysis
        borough_stats = self.df.groupby('borough').agg({
            'unique_key': 'count',
            'response_time_hours': 'mean'
        }).round(2)
        
        return {
            'top_zip_codes': top_zips.to_dict(),
            'borough_statistics': borough_stats.to_dict(),
            'geographic_concentration': self._calculate_geographic_concentration()
        }
    
    def generate_temporal_analysis(self) -> Dict[str, Any]:
        """Analyze temporal patterns in complaints"""
        
        hourly_pattern = self.df.groupby('hour')['unique_key'].count()
        daily_pattern = self.df.groupby('day_of_week')['unique_key'].count()
        monthly_trend = self.df.groupby(['year', 'month'])['unique_key'].count()
        
        return {
            'peak_hours': hourly_pattern.nlargest(3).to_dict(),
            'peak_days': daily_pattern.nlargest(3).to_dict(),
            'seasonal_trends': monthly_trend.groupby(level=1).mean().to_dict(),
            'year_over_year': monthly_trend.groupby(level=0).sum().to_dict()
        }
    
    def _calculate_geographic_concentration(self) -> Dict[str, float]:
        """Calculate geographic concentration metrics"""
        
        # Herfindahl-Hirschman Index for geographic concentration
        zip_shares = self.df['incident_zip'].value_counts(normalize=True)
        hhi = (zip_shares ** 2).sum()
        
        # Top 10% concentration
        top_10_pct = zip_shares.head(int(len(zip_shares) * 0.1)).sum()
        
        return {
            'hhi_index': round(hhi, 4),
            'top_10_percent_concentration': round(top_10_pct, 4)
        }