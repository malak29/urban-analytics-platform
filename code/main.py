import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

from src.config.settings import settings
from src.data.api_client import get_api_client, DataCache
from src.data.data_processor import NoiseDataProcessor, AnalyticsEngine
from src.visualization.charts import NoiseAnalyticsVisualizer
from src.visualization.maps import MapGenerator
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


class NoiseAnalyticsPlatform:
    """Main application class for noise analytics platform"""
    
    def __init__(self):
        self.api_client = get_api_client()
        self.cache = DataCache()
        self.processor = NoiseDataProcessor()
        self.visualizer = NoiseAnalyticsVisualizer()
        self.map_generator = MapGenerator()
        
    def run_full_analysis(self, 
                         date_range: tuple = None,
                         use_cache: bool = True) -> dict:
        """Execute complete noise analysis pipeline"""
        
        logger.info("Starting full noise analysis pipeline")
        
        try:
            # Step 1: Data acquisition
            df = self._get_data(date_range, use_cache)
            
            if df.empty:
                logger.error("No data available for analysis")
                return {"error": "No data available", "status": "failed"}
            
            # Step 2: Data processing
            df_processed = self.processor.process_raw_data(df)
            
            # Step 3: Generate analytics
            analytics = AnalyticsEngine(df_processed)
            hotspot_analysis = analytics.generate_hotspot_analysis()
            temporal_analysis = analytics.generate_temporal_analysis()
            
            # Step 4: Create visualizations
            chart_exports = self.visualizer.export_all_charts(df_processed)
            
            # Step 5: Generate maps
            map_exports = self.map_generator.create_all_maps(df_processed)
            
            # Step 6: Generate summary report
            summary_stats = self.processor.get_summary_statistics()
            dashboard_summary = self.visualizer.create_dashboard_summary(df_processed)
            
            # Step 7: Export processed data
            data_export_path = self.processor.export_processed_data()
            
            # Compile results
            results = {
                "execution_timestamp": datetime.now().isoformat(),
                "data_summary": summary_stats,
                "dashboard_metrics": dashboard_summary,
                "analytics": {
                    "hotspot_analysis": hotspot_analysis,
                    "temporal_analysis": temporal_analysis
                },
                "exports": {
                    "charts": chart_exports,
                    "maps": map_exports,
                    "processed_data": data_export_path
                },
                "status": "success"
            }
            
            # Save results summary
            self._save_results_summary(results)
            
            logger.info("Analysis pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def _get_data(self, date_range: tuple, use_cache: bool):
        """Get data from API or cache"""
        
        # Check API health
        if not self.api_client.health_check():
            logger.error("API health check failed")
            raise ConnectionError("NYC Open Data API is not accessible")
        
        # Generate cache key
        cache_params = {
            "date_range": date_range,
            "timestamp": datetime.now().strftime("%Y-%m-%d")
        }
        cache_key = self.cache.get_cache_key(cache_params)
        
        # Try to load from cache
        if use_cache:
            cached_df = self.cache.get(cache_key)
            if cached_df is not None:
                logger.info("Using cached data")
                return cached_df
        
        # Fetch fresh data
        logger.info("Fetching fresh data from API")
        df = self.api_client.fetch_noise_complaints(date_range=date_range)
        
        # Cache the results
        if not df.empty:
            self.cache.set(cache_key, df)
        
        return df
    
    def _save_results_summary(self, results: dict):
        """Save analysis results summary to JSON"""
        
        output_path = settings.data.OUTPUT_DIR / "reports" / f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results summary saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results summary: {e}")


def main():
    """Main application entry point"""
    
    # Set up logging
    setup_logging()
    
    parser = argparse.ArgumentParser(description="NYC Noise Analytics Platform")
    parser.add_argument("--date-start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--date-end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache usage")
    parser.add_argument("--quick-run", action="store_true", help="Run with last 30 days data only")
    
    args = parser.parse_args()
    
    # Determine date range
    if args.quick_run:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        date_range = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        logger.info("Quick run mode: analyzing last 30 days")
    elif args.date_start and args.date_end:
        date_range = (args.date_start, args.date_end)
        logger.info(f"Custom date range: {date_range[0]} to {date_range[1]}")
    else:
        # Default: last 12 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        logger.info("Default mode: analyzing last 12 months")
    
    # Initialize platform
    platform = NoiseAnalyticsPlatform()
    
    # Execute analysis
    results = platform.run_full_analysis(
        date_range=date_range,
        use_cache=not args.no_cache
    )
    
    if results.get("status") == "success":
        print("\n" + "="*60)
        print("NOISE ANALYSIS COMPLETE")
        print("="*60)
        print(f"Total Complaints Analyzed: {results['data_summary']['total_complaints']:,}")
        print(f"Date Range: {results['data_summary']['date_range']['start']} to {results['data_summary']['date_range']['end']}")
        print(f"Top Complaint Type: {list(results['data_summary']['complaint_types'].keys())[0]}")
        print(f"Data Quality Score: {results['dashboard_metrics']['data_quality_score']}%")
        print("\nExported Files:")
        for category, files in results['exports'].items():
            print(f"\n{category.upper()}:")
            if isinstance(files, dict):
                for name, path in files.items():
                    print(f"  - {name}: {path}")
            else:
                print(f"  - {files}")
        print("\n" + "="*60)
        return 0
    else:
        print(f"\nAnalysis failed: {results.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())