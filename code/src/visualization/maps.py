import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MapGenerator:
    """Production-ready geographic visualization engine"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.data.OUTPUT_DIR / "visualizations"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.map_center = settings.visualization.MAP_CENTER
        self.map_zoom = settings.visualization.MAP_ZOOM
        self.heatmap_radius = settings.visualization.HEATMAP_RADIUS
    
    def create_complaint_heatmap(self, 
                               df: pd.DataFrame, 
                               save: bool = True) -> folium.Map:
        """Create interactive heatmap of noise complaints"""
        
        logger.info("Creating complaint heatmap")
        
        # Filter valid coordinates
        df_coords = df.dropna(subset=['latitude', 'longitude'])
        
        if df_coords.empty:
            logger.warning("No valid coordinates found for heatmap")
            return None
        
        # Create base map
        map_obj = folium.Map(
            location=self.map_center,
            zoom_start=self.map_zoom,
            tiles='OpenStreetMap'
        )
        
        # Prepare heatmap data
        heat_data = [[row['latitude'], row['longitude']] for _, row in df_coords.iterrows()]
        
        # Add heatmap layer
        HeatMap(
            heat_data,
            min_opacity=0.2,
            radius=self.heatmap_radius,
            blur=15,
            max_zoom=1,
            gradient={
                0.2: 'blue',
                0.4: 'lime', 
                0.6: 'orange',
                0.8: 'red'
            }
        ).add_to(map_obj)
        
        # Add title
        title_html = '''
        <h3 align="center" style="font-size:16px"><b>NYC Noise Complaints Heatmap</b></h3>
        '''
        map_obj.get_root().html.add_child(folium.Element(title_html))
        
        if save:
            output_path = self.output_dir / "noise_heatmap.html"
            map_obj.save(str(output_path))
            logger.info(f"Heatmap saved to {output_path}")
        
        return map_obj
    
    def create_choropleth_map(self, 
                            df: pd.DataFrame,
                            save: bool = True) -> folium.Map:
        """Create choropleth map showing complaints by ZIP code"""
        
        logger.info("Creating choropleth map")
        
        # Filter noise complaints
        noise_df = df[df['complaint_type'].str.contains('NOISE', case=False, na=False)]
        
        if noise_df.empty:
            logger.warning("No noise complaints found for choropleth")
            return None
        
        # Count complaints by ZIP code
        zip_counts = noise_df['incident_zip'].value_counts().reset_index()
        zip_counts.columns = ['zip_code', 'complaint_count']
        
        # Create base map
        map_obj = folium.Map(
            location=self.map_center,
            zoom_start=self.map_zoom,
            tiles='OpenStreetMap'
        )
        
        # Create point-based visualization
        map_obj = self._create_zip_point_map(map_obj, noise_df, zip_counts)
        
        # Add layer control
        folium.LayerControl().add_to(map_obj)
        
        if save:
            output_path = self.output_dir / "noise_choropleth.html"
            map_obj.save(str(output_path))
            logger.info(f"Choropleth map saved to {output_path}")
        
        return map_obj
    
    def _create_zip_point_map(self, 
                            map_obj: folium.Map,
                            noise_df: pd.DataFrame,
                            zip_counts: pd.DataFrame) -> folium.Map:
        """Create point-based map for ZIP codes"""
        
        # Calculate centroids for each ZIP code
        if not noise_df.empty:
            zip_centroids = (noise_df.groupby('incident_zip')[['latitude', 'longitude']]
                           .mean().reset_index())
            
            # Merge with counts
            zip_data = zip_centroids.merge(zip_counts, left_on='incident_zip', right_on='zip_code')
            
            # Add circle markers
            for _, row in zip_data.iterrows():
                if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=min(50, max(5, row['complaint_count'] / 50)),
                        popup=f"ZIP: {row['incident_zip']}<br>Complaints: {row['complaint_count']}",
                        color='red',
                        fill=True,
                        opacity=0.7
                    ).add_to(map_obj)
        
        return map_obj
    
    def create_cluster_map(self, 
                          df: pd.DataFrame, 
                          save: bool = True) -> folium.Map:
        """Create clustered marker map for individual complaints"""
        
        logger.info("Creating cluster map")
        
        # Filter valid coordinates
        df_coords = df.dropna(subset=['latitude', 'longitude'])
        
        if df_coords.empty:
            logger.warning("No valid coordinates for cluster map")
            return None
        
        # Sample data if too large (performance optimization)
        if len(df_coords) > 10000:
            df_coords = df_coords.sample(n=10000, random_state=42)
            logger.info("Data sampled to 10,000 points for performance")
        
        # Create base map
        map_obj = folium.Map(
            location=self.map_center,
            zoom_start=self.map_zoom
        )
        
        # Create marker cluster
        marker_cluster = MarkerCluster().add_to(map_obj)
        
        # Add markers
        for _, row in df_coords.iterrows():
            popup_text = self._create_popup_text(row)
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(
                    color=self._get_marker_color(row['complaint_type']),
                    icon='volume-up',
                    prefix='fa'
                )
            ).add_to(marker_cluster)
        
        if save:
            output_path = self.output_dir / "complaint_clusters.html"
            map_obj.save(str(output_path))
            logger.info(f"Cluster map saved to {output_path}")
        
        return map_obj
    
    def _create_popup_text(self, row: pd.Series) -> str:
        """Generate popup text for map markers"""
        
        popup_lines = [
            f"<b>{row.get('complaint_type', 'Unknown')}</b>",
            f"Date: {row.get('created_date', 'Unknown')}",
            f"Borough: {row.get('borough', 'Unknown')}",
            f"ZIP: {row.get('incident_zip', 'Unknown')}",
            f"Agency: {row.get('agency_name', 'Unknown')}"
        ]
        
        if pd.notna(row.get('descriptor')):
            popup_lines.append(f"Description: {row['descriptor']}")
        
        return "<br>".join(popup_lines)
    
    def _get_marker_color(self, complaint_type: str) -> str:
        """Get marker color based on complaint type"""
        
        if pd.isna(complaint_type):
            return 'gray'
        
        complaint_type = str(complaint_type).upper()
        
        color_mapping = {
            'VEHICLE': 'red',
            'RESIDENTIAL': 'blue', 
            'COMMERCIAL': 'green',
            'HELICOPTER': 'purple',
            'CONSTRUCTION': 'orange',
            'STREET': 'darkblue'
        }
        
        for keyword, color in color_mapping.items():
            if keyword in complaint_type:
                return color
        
        return 'gray'
    
    def create_all_maps(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate all map visualizations"""
        
        logger.info("Generating complete map suite")
        
        maps_generated = {}
        
        try:
            # Heatmap
            heatmap = self.create_complaint_heatmap(df)
            if heatmap:
                maps_generated['heatmap'] = str(self.output_dir / "noise_heatmap.html")
            
            # Choropleth
            choropleth = self.create_choropleth_map(df)
            if choropleth:
                maps_generated['choropleth'] = str(self.output_dir / "noise_choropleth.html")
            
            # Cluster map
            cluster_map = self.create_cluster_map(df)
            if cluster_map:
                maps_generated['clusters'] = str(self.output_dir / "complaint_clusters.html")
            
            logger.info(f"Generated {len(maps_generated)} maps successfully")
            return maps_generated
            
        except Exception as e:
            logger.error(f"Error generating maps: {e}")
            raise