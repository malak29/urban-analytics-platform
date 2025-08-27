import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class NoiseAnalyticsVisualizer:
    """Production-ready visualization engine for noise analytics"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.data.OUTPUT_DIR / "visualizations"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set visualization defaults
        self._configure_plotting()
    
    def _configure_plotting(self):
        """Configure plotting defaults for production quality"""
        
        # Matplotlib configuration
        plt.rcParams.update({
            'figure.figsize': settings.visualization.FIGURE_SIZE,
            'figure.dpi': settings.visualization.DPI,
            'font.size': settings.visualization.FONT_SIZE,
            'axes.titlesize': settings.visualization.FONT_SIZE + 2,
        })
        
        # Seaborn configuration
        sns.set_palette(settings.visualization.COLOR_PALETTE)
    
    def create_complaint_distribution_chart(self, 
                                          df: pd.DataFrame, 
                                          top_n: int = 10,
                                          save: bool = True) -> go.Figure:
        """Create interactive pie chart for complaint type distribution"""
        
        logger.info(f"Creating complaint distribution chart (top {top_n})")
        
        # Get top complaint types
        complaint_counts = df['complaint_type'].value_counts().head(top_n)
        
        # Create interactive pie chart
        fig = go.Figure(data=[
            go.Pie(
                labels=complaint_counts.index,
                values=complaint_counts.values,
                hole=0.3,
                textinfo='label+percent',
                textposition='outside',
                marker=dict(
                    colors=px.colors.qualitative.Set3,
                    line=dict(color='white', width=2)
                )
            )
        ])
        
        fig.update_layout(
            title={
                'text': f'Top {top_n} Noise Complaint Types Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            showlegend=True,
            width=800,
            height=600,
            margin=dict(t=80, b=40, l=40, r=40)
        )
        
        if save:
            output_path = self.output_dir / "complaint_distribution.html"
            fig.write_html(str(output_path))
            logger.info(f"Chart saved to {output_path}")
        
        return fig
    
    def create_temporal_heatmap(self, 
                               df: pd.DataFrame, 
                               save: bool = True) -> go.Figure:
        """Create heatmap showing complaint patterns by hour and day"""
        
        logger.info("Creating temporal heatmap")
        
        # Prepare data
        df['day_of_week_num'] = df['created_date'].dt.dayofweek
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                    'Friday', 'Saturday', 'Sunday']
        
        # Create pivot table
        heatmap_data = df.groupby(['day_of_week_num', 'hour']).size().unstack(fill_value=0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=list(range(24)),
            y=day_names,
            colorscale='YlOrRd',
            showscale=True,
            hoveropacity=0.8,
            colorbar=dict(title="Number of Complaints")
        ))
        
        fig.update_layout(
            title={
                'text': 'Noise Complaints by Day and Hour',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            width=900,
            height=500,
            margin=dict(t=80, b=60, l=80, r=60)
        )
        
        if save:
            output_path = self.output_dir / "temporal_heatmap.html"
            fig.write_html(str(output_path))
            logger.info(f"Heatmap saved to {output_path}")
        
        return fig
    
    def create_response_time_analysis(self, 
                                    df: pd.DataFrame, 
                                    save: bool = True) -> go.Figure:
        """Create response time analysis by agency"""
        
        logger.info("Creating response time analysis")
        
        # Filter valid response times
        df_response = df.dropna(subset=['response_time_hours'])
        df_response = df_response[df_response['response_time_hours'] > 0]
        df_response = df_response[df_response['response_time_hours'] < 8760]  # Less than 1 year
        
        if df_response.empty:
            logger.warning("No valid response time data found")
            return None
        
        # Calculate average response time by agency
        agency_response = (df_response.groupby('agency_name')['response_time_hours']
                          .agg(['mean', 'median', 'count'])
                          .sort_values('mean', ascending=False)
                          .head(15))
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Average Response Time by Agency', 'Response Time Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Bar chart for agency response times
        fig.add_trace(
            go.Bar(
                x=agency_response['mean'],
                y=agency_response.index,
                orientation='h',
                name='Mean Response Time',
                marker_color='lightcoral'
            ),
            row=1, col=1
        )
        
        # Histogram for response time distribution
        fig.add_trace(
            go.Histogram(
                x=df_response['response_time_hours'],
                nbinsx=50,
                name='Distribution',
                marker_color='skyblue',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title={
                'text': 'Response Time Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            height=600,
            width=1200,
            showlegend=False,
            margin=dict(t=80, b=60, l=120, r=60)
        )
        
        if save:
            output_path = self.output_dir / "response_time_analysis.html"
            fig.write_html(str(output_path))
            logger.info(f"Response time analysis saved to {output_path}")
        
        return fig
    
    def create_yearly_trend_analysis(self, 
                                   df: pd.DataFrame, 
                                   save: bool = True) -> go.Figure:
        """Create year-over-year trend analysis"""
        
        logger.info("Creating yearly trend analysis")
        
        # Monthly aggregation by year
        monthly_data = (df.groupby([df['created_date'].dt.year, df['created_date'].dt.month])
                       .size().unstack(level=0, fill_value=0))
        
        fig = go.Figure()
        
        # Add line for each year
        for year in monthly_data.columns:
            fig.add_trace(go.Scatter(
                x=list(range(1, 13)),
                y=monthly_data[year].values,
                mode='lines+markers',
                name=str(year),
                line=dict(width=3),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title={
                'text': 'Year-over-Year Monthly Comparison of Noise Complaints',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title="Month",
            yaxis_title="Number of Complaints",
            width=1000,
            height=600,
            hovermode='x unified',
            margin=dict(t=80, b=60, l=60, r=60)
        )
        
        # Customize x-axis
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=month_names
        )
        
        if save:
            output_path = self.output_dir / "yearly_trend_analysis.html"
            fig.write_html(str(output_path))
            logger.info(f"Yearly trend analysis saved to {output_path}")
        
        return fig
    
    def create_dashboard_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create summary metrics for dashboard display"""
        
        total_complaints = len(df)
        
        # Time range
        date_range = (
            df['created_date'].min().strftime('%Y-%m-%d'),
            df['created_date'].max().strftime('%Y-%m-%d')
        )
        
        # Top complaint type
        top_complaint = df['complaint_type'].value_counts().index[0]
        top_complaint_count = df['complaint_type'].value_counts().iloc[0]
        
        # Peak hour
        peak_hour = df['hour'].value_counts().index[0]
        
        # Most affected borough
        top_borough = df['borough'].value_counts().index[0]
        
        # Average response time
        avg_response = df['response_time_hours'].mean() if 'response_time_hours' in df.columns else None
        
        return {
            'total_complaints': total_complaints,
            'date_range': date_range,
            'top_complaint_type': {
                'type': top_complaint,
                'count': int(top_complaint_count),
                'percentage': round((top_complaint_count / total_complaints) * 100, 1)
            },
            'peak_hour': int(peak_hour),
            'most_affected_borough': top_borough,
            'average_response_time_hours': round(avg_response, 2) if avg_response else None,
            'data_quality_score': self._calculate_data_quality_score(df)
        }
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-100)"""
        
        scores = []
        
        # Completeness scores
        for col in ['complaint_type', 'borough', 'created_date']:
            if col in df.columns:
                completeness = (1 - df[col].isna().mean()) * 100
                scores.append(completeness)
        
        # Coordinate completeness
        if all(col in df.columns for col in ['latitude', 'longitude']):
            coord_completeness = (1 - df[['latitude', 'longitude']].isna().any(axis=1).mean()) * 100
            scores.append(coord_completeness)
        
        # Valid zip codes
        if 'incident_zip' in df.columns:
            valid_zips = df['incident_zip'].str.match(r'^\d{5}$', na=False).mean() * 100
            scores.append(valid_zips)
        
        return round(sum(scores) / len(scores) if scores else 0, 1)
    
    def export_all_charts(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate and save all charts"""
        
        logger.info("Generating complete chart suite")
        
        exports = {}
        
        try:
            # Create all visualizations
            charts = {
                'complaint_distribution': self.create_complaint_distribution_chart(df),
                'temporal_heatmap': self.create_temporal_heatmap(df),
                'response_time_analysis': self.create_response_time_analysis(df),
                'yearly_trends': self.create_yearly_trend_analysis(df)
            }
            
            for chart_name, fig in charts.items():
                if fig is not None:
                    output_path = self.output_dir / f"{chart_name}.html"
                    fig.write_html(str(output_path))
                    exports[chart_name] = str(output_path)
            
            logger.info(f"All charts exported successfully to {self.output_dir}")
            return exports
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            raise