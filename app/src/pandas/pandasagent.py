"""
Enhanced PandasAgent with ReAct Framework for TBI Data Analysis
A robust, intelligent data analysis tool with automatic visualization and error recovery.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Dict, Any, List, Optional, Union
import os
import json
import traceback
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import asyncio
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database schema definition for TBI tables
DB_SCHEMA = """
Tables:
1. patients
   - patient_id (varchar, primary key): Unique identifier for each patient
   - first_name (text): Patient's first name
   - last_name (text): Patient's last name
   - date_of_birth (text): Patient's date of birth
   - postal_code (text): Patient's postal code
   - user_type (text): Type of user
   - registered_at (timestamptz): Registration timestamp
   - country (text): Patient's country
   - referral_group (text): Referral group information
   - veteran (text): Veteran status
   - ethnicity (text): Patient's ethnicity
   - race (text): Patient's race
   - city (text): Patient's city
   - state (text): Patient's state
   - dark_mode (bool): Dark mode preference
   - gender (text): Patient's gender
   - patient_type (text): Type of patient
   - patient_sub_type (text): Patient sub-type
   - head_hit_count (float): Count of head hits
   - has_tbi_before (text): Whether patient had TBI before
   - age (float): Patient's age

2. tbi_incidents
   - patient_id (varchar, foreign key): References patients.patient_id
   - tbi_incident_date (text): Date of TBI incident
   - injury_from (text): Cause of injury
   - head_hit_location (text): Location on head where hit occurred
   - num_head_hit_location (int): Number of hit locations
   - total_tbi (int): Total TBI count
   - immediate_symptoms_resulting (text): Immediate symptoms after injury
   - describe_event (text): Description of the incident

3. symptom_logs
   - patient_about_id (text): Patient reference
   - symptom_date (date): Date of symptom
   - logged_at (timestamptz): When the symptom was logged
   - severity (float): Severity of symptom
   - category (text): Symptom category
   - subcategory (text): Symptom subcategory
   - had_symptom (text): Whether patient had the symptom
   - factor (text): Related factor

4. worst_symptoms
   - patient_id (text): Patient reference
   - symptom_id (int): Symptom identifier
   - id (int): Record identifier
   - category (text): Symptom category
   - subcategory (text): Symptom subcategory
   - factor (text): Related factor

5. therapies
   - patient_id (text): Patient reference
   - therapies_id (int): Therapy identifier
   - therapies (text): Therapy description
   - category (text): Therapy category

6. social_determinants
   - patient_id (text): Patient reference
   - symptom_id (int): Related symptom
   - category (text): Social determinant category
   - subcategory (text): Social determinant subcategory
   - factor (text): Related factor

7. symptom_reference
   - patient_id (text): Patient reference
   - symptom_id (int): Symptom identifier
   - id (int): Record identifier
   - category (text): Symptom category
   - subcategory (text): Symptom subcategory
   - factor (text): Related factor
   - prime (json): Additional data in JSON format

Table Relationships:
- One patient can have multiple TBI incidents (one-to-many)
- One patient can have multiple symptom logs (one-to-many)
- One patient can have multiple worst symptoms (one-to-many)
- One patient can have multiple therapies (one-to-many)
- One patient can have multiple social determinants (one-to-many)
- One patient can have multiple symptom references (one-to-many)
"""

class SchemaAnalyzer:
    """Enhanced schema analysis for better data understanding"""
    
    def __init__(self, dataframes: Dict[str, pd.DataFrame]):
        self.dataframes = dataframes
        self.schema_info = {}
        self.relationships = {}
        self.data_quality = {}
        self._analyze_all()
    
    def _analyze_all(self):
        """Perform comprehensive schema analysis"""
        for df_name, df in self.dataframes.items():
            table_name = df_name.replace('df_', '')
            self.schema_info[table_name] = self._analyze_table(df)
            self.data_quality[table_name] = self._analyze_data_quality(df)
        
        self._identify_relationships()
    
    def _analyze_table(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze individual table structure"""
        analysis = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': {},
            'categorical_columns': [],
            'numerical_columns': [],
            'date_columns': [],
            'text_columns': [],
            'key_columns': []
        }
        
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'sample_values': df[col].dropna().head(3).tolist()
            }
            
            # Determine column type
            if df[col].dtype in ['object', 'category']:
                if df[col].nunique() / len(df) < 0.1:  # Low cardinality
                    analysis['categorical_columns'].append(col)
                else:
                    analysis['text_columns'].append(col)
                    
                # Check for ID columns
                if 'id' in col.lower() or col.lower().endswith('_id'):
                    analysis['key_columns'].append(col)
                    
            elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                analysis['numerical_columns'].append(col)
                col_info.update({
                    'min': df[col].min() if not df[col].isnull().all() else None,
                    'max': df[col].max() if not df[col].isnull().all() else None,
                    'mean': df[col].mean() if not df[col].isnull().all() else None,
                    'std': df[col].std() if not df[col].isnull().all() else None
                })
                
            elif 'datetime' in str(df[col].dtype) or 'date' in col.lower():
                analysis['date_columns'].append(col)
            
            analysis['columns'][col] = col_info
        
        return analysis
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality metrics"""
        return {
            'total_nulls': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'completeness_score': ((df.size - df.isnull().sum().sum()) / df.size) * 100,
            'columns_with_nulls': df.columns[df.isnull().any()].tolist(),
            'high_null_columns': df.columns[df.isnull().sum() / len(df) > 0.5].tolist()
        }
    
    def _identify_relationships(self):
        """Identify potential relationships between tables"""
        for table1, info1 in self.schema_info.items():
            for table2, info2 in self.schema_info.items():
                if table1 != table2:
                    common_cols = set(info1['columns'].keys()) & set(info2['columns'].keys())
                    key_cols = [col for col in common_cols if 'id' in col.lower()]
                    if key_cols:
                        self.relationships[f"{table1}-{table2}"] = key_cols
    
    def get_comprehensive_schema(self) -> str:
        """Generate comprehensive schema description"""
        schema_desc = ["# Comprehensive Database Schema Analysis\n"]
        
        for table, info in self.schema_info.items():
            schema_desc.append(f"## Table: {table}")
            schema_desc.append(f"- Rows: {info['row_count']:,}")
            schema_desc.append(f"- Columns: {info['column_count']}")
            schema_desc.append(f"- Data Quality Score: {self.data_quality[table]['completeness_score']:.1f}%")
            
            if info['categorical_columns']:
                schema_desc.append(f"- Categorical Columns: {', '.join(info['categorical_columns'])}")
            if info['numerical_columns']:
                schema_desc.append(f"- Numerical Columns: {', '.join(info['numerical_columns'])}")
            if info['date_columns']:
                schema_desc.append(f"- Date Columns: {', '.join(info['date_columns'])}")
            if info['key_columns']:
                schema_desc.append(f"- Key Columns: {', '.join(info['key_columns'])}")
            
            schema_desc.append("")
        
        if self.relationships:
            schema_desc.append("## Table Relationships")
            for rel, cols in self.relationships.items():
                schema_desc.append(f"- {rel}: {', '.join(cols)}")
        
        return "\n".join(schema_desc)

class VisualizationEngine:
    """Handle visualization generation"""
    
    def __init__(self):
        self.supported_chart_types = [
            'bar', 'line', 'scatter', 'histogram', 'box', 'pie', 
            'heatmap', 'violin', 'density', 'correlation'
        ]
    
    def create_visualization(self, data: pd.DataFrame, chart_type: str, 
                      title: str = "", x_col: str = None, y_col: str = None,
                      **kwargs) -> str:
        """Create visualization and return HTML, with additional checks to prevent template vars"""
        try:
            # Generate the visualization
            if chart_type == 'bar':
                html = self._create_bar_chart(data, title, x_col, y_col, **kwargs)
            elif chart_type == 'line':
                html = self._create_line_chart(data, title, x_col, y_col, **kwargs)
            elif chart_type == 'scatter':
                html = self._create_scatter_plot(data, title, x_col, y_col, **kwargs)
            elif chart_type == 'histogram':
                html = self._create_histogram(data, title, x_col, **kwargs)
            elif chart_type == 'box':
                html = self._create_box_plot(data, title, x_col, y_col, **kwargs)
            elif chart_type == 'pie':
                html = self._create_pie_chart(data, title, x_col, y_col, **kwargs)
            elif chart_type == 'heatmap':
                html = self._create_heatmap(data, title, **kwargs)
            elif chart_type == 'correlation':
                html = self._create_correlation_matrix(data, title, **kwargs)
            else:
                html = self._create_auto_chart(data, title, x_col, y_col)
            
            # Verify we have valid HTML output (not template vars)
            if isinstance(html, str):
                if '${' in html or '$vizContent' in html or '$data.visualization' in html:
                    # Log the issue
                    logger.error("Template variables detected in visualization output")
                    return ""
            
            # Ensure plotly is properly included
            if isinstance(html, str) and 'plotly' in html and 'cdn.plot.ly' not in html:
                html = f"""
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                {html}
                """
            
            return html
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            # Return empty string instead of error message to avoid showing anything
            return ""
    
    def _create_bar_chart(self, data: pd.DataFrame, title: str, x_col: str, y_col: str, **kwargs):
        """Create bar chart with improved HTML output"""
        try:
            # Check if data has percentage column for enhanced hover info
            if 'Percentage' in data.columns and y_col in ['Count', 'count']:
                # Add color differentiation for multiple categories
                if len(data) > 1:
                    # Use color scale based on count values for better visual distinction
                    fig = px.bar(data, x=x_col, y=y_col, title=title, 
                                color=y_col,  # Color based on count values
                                color_continuous_scale='Viridis',  # Nice color palette
                                hover_name=x_col, **kwargs)
                    
                    # Enhanced hover template with percentage info
                    fig.update_traces(
                        hovertemplate="<b>%{x}</b><br>Count: %{y:,}<br>Percentage: %{customdata:.2f}%<extra></extra>",
                        customdata=data['Percentage']
                    )
                else:
                    fig = px.bar(data, x=x_col, y=y_col, title=title, **kwargs)
                    if 'Percentage' in data.columns:
                        fig.update_traces(
                            hovertemplate="<b>%{x}</b><br>Count: %{y:,}<br>Percentage: %{customdata:.2f}%<extra></extra>",
                            customdata=data['Percentage']
                        )
            else:
                # Standard bar chart without percentage info
                if len(data) > 1:
                    fig = px.bar(data, x=x_col, y=y_col, title=title, 
                                color=x_col, color_discrete_sequence=px.colors.qualitative.Set2, **kwargs)
                else:
                    fig = px.bar(data, x=x_col, y=y_col, title=title, **kwargs)
            
            # Adjust layout for better appearance
            fig.update_layout(
                showlegend=True, 
                height=500,
                font=dict(family="Arial, sans-serif", size=12),
                margin=dict(l=60, r=30, t=50, b=60)
            )
            
            # Add direct CDN reference to ensure plotly is loaded
            html = fig.to_html(include_plotlyjs='cdn', full_html=False)
            
            # Wrap in a container for better CSS control
            html = f"""
            <div style="width:100%; height:500px;">
                {html}
            </div>
            """
            
            return html
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            return ""
    
    def _create_line_chart(self, data: pd.DataFrame, title: str, x_col: str, y_col: str, **kwargs):
        """Create line chart"""
        # Add percentage info to hover if available
        if 'Percentage' in data.columns:
            fig = px.line(data, x=x_col, y=y_col, title=title, **kwargs)
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>Count: %{y:,}<br>Percentage: %{customdata:.2f}%<extra></extra>",
                customdata=data['Percentage']
            )
        else:
            fig = px.line(data, x=x_col, y=y_col, title=title, **kwargs)
        
        fig.update_layout(height=500)
        return fig.to_html(include_plotlyjs='cdn')
    
    def _create_scatter_plot(self, data: pd.DataFrame, title: str, x_col: str, y_col: str, **kwargs):
        """Create scatter plot"""
        fig = px.scatter(data, x=x_col, y=y_col, title=title, **kwargs)
        fig.update_layout(height=500)
        return fig.to_html(include_plotlyjs='cdn')
    
    def _create_histogram(self, data: pd.DataFrame, title: str, x_col: str, **kwargs):
        """Create histogram"""
        fig = px.histogram(data, x=x_col, title=title, **kwargs)
        fig.update_layout(height=500)
        return fig.to_html(include_plotlyjs='cdn')
    
    def _create_box_plot(self, data: pd.DataFrame, title: str, x_col: str, y_col: str = None, **kwargs):
        """Create box plot"""
        if y_col and x_col:
            # Grouped box plot (e.g., age by gender) - use color for different groups
            fig = px.box(data, x=x_col, y=y_col, color=x_col, title=title,
                        color_discrete_sequence=px.colors.qualitative.Set2, **kwargs)
        elif y_col:
            # Single box plot
            fig = px.box(data, y=y_col, title=title, **kwargs)
        else:
            # Single box plot with x_col as y
            fig = px.box(data, y=x_col, title=title, **kwargs)
        fig.update_layout(height=500)
        return fig.to_html(include_plotlyjs='cdn')
    
    def _create_pie_chart(self, data: pd.DataFrame, title: str, x_col: str, y_col: str, **kwargs):
        """Create pie chart"""
        # Limit pie chart to top 6 categories for readability
        if len(data) > 6:
            data = data.head(6)
        
        # Pie charts naturally show percentages, but we'll enhance with count info
        if 'Percentage' in data.columns:
            # Create custom labels showing both count and percentage
            labels = [f"{row[x_col]}<br>({row[y_col]:,} - {row['Percentage']}%)" 
                     for _, row in data.iterrows()]
            
            fig = px.pie(data, names=x_col, values=y_col, title=title, **kwargs)
            fig.update_traces(
                textinfo='label+percent',
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>"
            )
        else:
            fig = px.pie(data, names=x_col, values=y_col, title=title, **kwargs)
            fig.update_traces(textinfo='label+percent+value')
        
        fig.update_layout(height=500)
        return fig.to_html(include_plotlyjs='cdn')
    
    def _create_heatmap(self, data: pd.DataFrame, title: str, **kwargs):
        """Create heatmap"""
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return "<p>No numerical data available for heatmap</p>"
        
        fig = px.imshow(numeric_data.corr(), title=title, aspect="auto",
                       color_continuous_scale='RdBu_r')
        fig.update_layout(height=500)
        return fig.to_html(include_plotlyjs='cdn')
    
    def _create_correlation_matrix(self, data: pd.DataFrame, title: str, **kwargs):
        """Create correlation matrix"""
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return "<p>No numerical data available for correlation matrix</p>"
        
        corr_matrix = numeric_data.corr()
        fig = px.imshow(corr_matrix, 
                       x=corr_matrix.columns, 
                       y=corr_matrix.columns,
                       title=title,
                       color_continuous_scale='RdBu_r',
                       aspect="auto")
        fig.update_layout(width=600, height=500)
        return fig.to_html(include_plotlyjs='cdn')
    
    def _create_auto_chart(self, data: pd.DataFrame, title: str, x_col: str = None, y_col: str = None):
        """Automatically determine best chart type"""
        if len(data.columns) == 2:
            col1, col2 = data.columns
            
            # Check for categorical vs numerical
            col1_is_categorical = data[col1].dtype == 'object' or data[col1].nunique() < 20
            col2_is_numerical = pd.api.types.is_numeric_dtype(data[col2])
            col1_is_numerical = pd.api.types.is_numeric_dtype(data[col1])
            col2_is_categorical = data[col2].dtype == 'object' or data[col2].nunique() < 20
            
            if col1_is_categorical and col2_is_numerical:
                # Perfect for bar chart (categories vs counts/values)
                return self._create_bar_chart(data, title, col1, col2)
            elif col2_is_categorical and col1_is_numerical:
                # Reverse: numerical vs categories - still use bar chart
                return self._create_bar_chart(data, title, col2, col1)
            elif col1_is_numerical and col2_is_numerical:
                # Both numerical - use scatter plot
                return self._create_scatter_plot(data, title, col1, col2)
            elif col1_is_categorical and col2_is_categorical:
                # Both categorical - not ideal, but try bar chart
                return self._create_bar_chart(data, title, col1, col2)
        
        elif len(data.columns) >= 3:
            # Check if we have the standard pattern: Category, Count, Percentage
            if 'Percentage' in data.columns and any(col in ['Count', 'count'] for col in data.columns):
                # Find the categorical column (not Count/Percentage)
                cat_cols = [col for col in data.columns if col not in ['Count', 'count', 'Percentage']]
                count_cols = [col for col in data.columns if col in ['Count', 'count']]
                
                if cat_cols and count_cols:
                    return self._create_bar_chart(data, title, cat_cols[0], count_cols[0])
            
            # Fallback to first two columns
            col1, col2 = data.columns[0], data.columns[1]
            return self._create_bar_chart(data, title, col1, col2)
        
        elif len(data.columns) == 1:
            # Single column analysis
            col = data.columns[0]
            if pd.api.types.is_numeric_dtype(data[col]):
                return self._create_histogram(data, title, col)
            else:
                # Categorical data - show value counts as bar chart with percentages
                value_counts = data[col].value_counts().head(10).reset_index()
                value_counts.columns = [col, 'Count']
                # Add percentage column for better visualization
                value_counts['Percentage'] = (value_counts['Count'] / value_counts['Count'].sum() * 100).round(2)
                return self._create_bar_chart(value_counts, title, col, 'Count')
        
        # Default fallback
        first_col = data.columns[0]
        return self._create_histogram(data, title, first_col)

class EnhancedPandasAgent:
    """Enhanced PandasAgent with ReAct framework, schema understanding, and visualization"""
    
    def __init__(self, gemini_api_key: str = None):
        self.gemini_api_key = gemini_api_key
        self.dataframes = {}
        self.schema_analyzer = None
        self.visualization_engine = VisualizationEngine()
        self.db_connection = None
        self.execution_history = []
        
        if gemini_api_key:
            self._configure_gemini(gemini_api_key)
    
    def _configure_gemini(self, api_key: str):
        """Configure Gemini AI"""
        genai.configure(api_key=api_key)
        
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        logger.info("Gemini AI configured successfully")
    
    def connect_to_database(self, db_config: Dict[str, str] = None) -> bool:
        """Connect to database and load data"""
        try:
            if db_config is None:
                load_dotenv()
                db_config = {
                    'user': os.getenv("user") or os.getenv("DB_USER"),
                    'password': os.getenv("password") or os.getenv("DB_PASSWORD"),
                    'host': os.getenv("host") or os.getenv("DB_HOST"),
                    'port': os.getenv("port") or os.getenv("DB_PORT", "5432"),
                    'dbname': os.getenv("dbname") or os.getenv("DB_NAME")
                }
            
            required_keys = ['user', 'password', 'host', 'port', 'dbname']
            missing_keys = [key for key in required_keys if not db_config.get(key)]
            
            if missing_keys:
                logger.error(f"Missing database configuration keys: {missing_keys}")
                return False
            
            self.db_connection = psycopg2.connect(**db_config)
            logger.info("Successfully connected to the database")
            
            return self.load_data_from_database()
            
        except Exception as e:
            logger.exception(f"Failed to connect to database: {str(e)}")
            return False
    
    def load_data_from_database(self) -> bool:
        """Load data from database"""
        if not self.db_connection:
            logger.error("Database connection not established")
            return False
        
        try:
            tables = [
                'patients', 'tbi_incidents', 'symptom_logs', 
                'worst_symptoms', 'therapies', 'social_determinants', 
                'symptom_reference'
            ]
            
            dataframes = {}
            for table in tables:
                try:
                    query = f"SELECT * FROM {table};"
                    df = pd.read_sql_query(query, self.db_connection)
                    dataframes[table] = df
                    logger.info(f"Loaded table '{table}' with {len(df)} rows")
                except Exception as e:
                    logger.warning(f"Could not load table '{table}': {str(e)}")
            
            return self.load_dataframes(dataframes)
            
        except Exception as e:
            logger.exception(f"Error loading data from database: {str(e)}")
            return False
    
    def load_dataframes(self, dataframes: Dict[str, pd.DataFrame]) -> bool:
        """Load dataframes and perform schema analysis"""
        try:
            # Store dataframes with prefixes
            for table_name, df in dataframes.items():
                self.dataframes[f"df_{table_name}"] = df
                logger.info(f"Loaded dataframe '{table_name}' with {len(df)} rows")
            
            # Perform comprehensive schema analysis
            self.schema_analyzer = SchemaAnalyzer(self.dataframes)
            logger.info("Schema analysis completed")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error loading dataframes: {str(e)}")
            return False
    
    async def gemini_chat_completion(self, prompt: str, user_query: str) -> str:
        """Generate chat completion using Gemini"""
        try:
            if not self.gemini_api_key:
                return "Error: Gemini API key not configured"
            
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config={"temperature": 0.1, "top_p": 0.95, "top_k": 10},
                safety_settings=self.safety_settings
            )
            
            chat = model.start_chat(history=[])
            chat.send_message(prompt)
            response = chat.send_message(user_query)
            
            response_text = response.text
            
            # Extract code if wrapped in code blocks
            if "```python" in response_text:
                code = response_text.split("```python")[1].split("```")[0].strip()
            elif "```" in response_text:
                code = response_text.split("```")[1].split("```")[0].strip()
            else:
                code = response_text.strip()
                
            return code
            
        except Exception as e:
            logger.exception(f"Error with Gemini chat completion: {str(e)}")
            return f"Error with Gemini API: {str(e)}"
    
    async def react_process_query(self, query: str) -> Dict[str, Any]:
        """Process query using ReAct framework: Reason, Act, Observe"""
        if not self.dataframes or not self.schema_analyzer:
            return {"answer": "Error: No data loaded. Please load data first.", 
                   "visualization": None, "metadata": []}
        
        try:
            # REASON: Analyze the query and plan approach
            reasoning_result = await self._reason_about_query(query)
            
            # ACT: Generate and execute code
            action_result = await self._act_on_reasoning(query, reasoning_result)
            
            # OBSERVE: Analyze results and create visualization if appropriate
            observation_result = await self._observe_and_visualize(query, action_result)
            
            return observation_result
            
        except Exception as e:
            logger.exception(f"Error in ReAct process: {str(e)}")
            return {"answer": f"Error processing query: {str(e)}", 
                   "visualization": None, "metadata": []}
    
    async def _reason_about_query(self, query: str) -> Dict[str, Any]:
        """REASON: Analyze query and determine approach"""
        reasoning_prompt = f"""
        You are analyzing a natural language query about TBI (Traumatic Brain Injury) data.
        
        Query: "{query}"
        
        DETAILED DATABASE SCHEMA:
        {DB_SCHEMA}
        
        ACTUAL DATA STRUCTURE:
        {self.schema_analyzer.get_comprehensive_schema()}
        
        Please analyze this query and provide:
        1. Which tables are needed
        2. What type of analysis is required (descriptive, comparative, correlation, etc.)
        3. Whether visualization would be helpful and what type
        4. Any potential data quality issues to consider
        5. The key columns and operations needed
        
        Respond in JSON format with keys: tables_needed, analysis_type, visualization_type, data_considerations, key_operations
        """
        
        try:
            reasoning_response = await self.gemini_chat_completion(reasoning_prompt, "Analyze this query")
            
            # Try to parse JSON response
            try:
                reasoning_data = json.loads(reasoning_response)
            except:
                # If not valid JSON, create a basic reasoning structure
                reasoning_data = {
                    "tables_needed": ["patients", "tbi_incidents"],
                    "analysis_type": "descriptive",
                    "visualization_type": "bar",
                    "data_considerations": [],
                    "key_operations": []
                }
            
            return reasoning_data
            
        except Exception as e:
            logger.error(f"Error in reasoning step: {str(e)}")
            return {
                "tables_needed": ["patients"],
                "analysis_type": "descriptive", 
                "visualization_type": None,
                "data_considerations": [],
                "key_operations": []
            }
    
    async def _act_on_reasoning(self, query: str, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """ACT: Generate and execute code based on reasoning"""
        
        # Enhanced system prompt with schema information
        system_prompt = f"""
        You are a data analyst generating Python pandas code for TBI data analysis.
        
        DETAILED DATABASE SCHEMA:
        {DB_SCHEMA}
        
        COMPREHENSIVE SCHEMA INFORMATION:
        {self.schema_analyzer.get_comprehensive_schema()}
        
        REASONING ANALYSIS:
        - Tables needed: {', '.join(reasoning.get('tables_needed', []))}
        - Analysis type: {reasoning.get('analysis_type', 'descriptive')}
        - Visualization suggested: {reasoning.get('visualization_type', 'none')}
        
        IMPORTANT CODE REQUIREMENTS:
        1. Use only dataframes that exist: {list(self.dataframes.keys())}
        2. Always check for null values and handle them appropriately
        3. Use .copy() when modifying dataframes to avoid warnings
        4. For text analysis, use .str.contains() with na=False and always make df[col].str.lower() to search any textual data
        5. Always use proper error handling for operations
        6. Store final result in variable called 'analysis_result'
        7. Store data for visualization in 'viz_data' (if applicable)
        8. Set 'chart_type' variable if visualization is recommended
        
        CODE STRUCTURE REQUIREMENTS:
        - Write DIRECT pandas operations, NOT functions
        - Do NOT create function definitions like def analyze_data()
        - Write simple, executable statements that directly set variables
        - Example: analysis_result = df_table.groupby('col').count()
        - NOT: def analyze_data(): return result
        
        CRITICAL: Generate simple, direct pandas code without function wrappers.
        
        SMART QUERY UNDERSTANDING:
        - If query asks for "most common", "top", "highest", "frequent", "popular", etc. → AUTOMATICALLY limit to TOP 10 results
        - If query asks for "distribution by" → Use appropriate grouping (box plot, histogram, or grouped bar)
        - If query asks for "relationship between" → Use scatter plot or correlation analysis
        - If query asks for "percentage" or "proportion" → Calculate and show percentages
        - Always consider the data size - limit large results to top 10-15 for better visualization
        
        VISUALIZATION RULES:
        1. For categorical frequency queries → Use 'bar' chart with top 10 results
        2. For age/numerical distribution by category → Use 'box' chart or grouped bar
        3. For single numerical distribution → Use 'histogram'
        4. For correlation/relationship → Use 'scatter' plot
        5. For proportions → Use 'pie' chart (only if ≤ 6 categories)
        6. Always limit visualization data to top 10-15 items for clarity
        
        EXAMPLE PATTERNS:
        
        # For "most common" queries (AUTO-LIMIT TO TOP 10):
        ```python
        result_df = df_table['column'].value_counts().head(10).reset_index()
        result_df.columns = ['Category', 'Count']
        result_df['Percentage'] = (result_df['Count'] / df_table['column'].value_counts().sum() * 100).round(2)
        analysis_result = result_df
        viz_data = result_df
        chart_type = 'bar'
        ```
        
        # For distribution by category (like age by gender):
        ```python
        # Create grouped summary
        grouped_data = df_patients.groupby('gender')['age'].agg(['mean', 'median', 'std', 'count']).reset_index()
        grouped_data.columns = ['Gender', 'Mean_Age', 'Median_Age', 'Std_Age', 'Count']
        analysis_result = grouped_data
        
        # For visualization - use box plot data
        viz_data = df_patients[['gender', 'age']].dropna()
        chart_type = 'box'
        ```
        
        # For merging tables:
        ```python
        merged_df = df_patients.merge(df_tbi_incidents, on='patient_id', how='inner')
        # Then apply TOP 10 rule if asking for "most common"
        if "most" in query.lower() or "common" in query.lower() or "frequent" in query.lower():
            result_df = merged_df['column'].value_counts().head(10).reset_index()
        ```
        
        # For therapy/treatment queries (ALWAYS TOP 10):
        ```python
        therapy_counts = df_therapies['therapies'].value_counts().head(10).reset_index()
        therapy_counts.columns = ['Therapy', 'Count']
        therapy_counts['Percentage'] = (therapy_counts['Count'] / df_therapies['therapies'].value_counts().sum() * 100).round(2)
        analysis_result = therapy_counts
        viz_data = therapy_counts
        chart_type = 'bar'
        ```
        
        # For text analysis with symptoms:
        ```python
        all_symptoms = []
        for symptoms_str in df_tbi_incidents['immediate_symptoms_resulting'].dropna():
            if isinstance(symptoms_str, str):
                symptoms = re.split(r',\s*', symptoms_str)
                all_symptoms.extend([s.strip().lower() for s in symptoms if s.strip()])
        
        # Count and get TOP 10
        symptom_counts = pd.Series(all_symptoms).value_counts().head(10).reset_index()
        symptom_counts.columns = ['Symptom', 'Count']
        symptom_counts['Percentage'] = (symptom_counts['Count'] / len(all_symptoms) * 100).round(2)
        analysis_result = symptom_counts
        viz_data = symptom_counts
        chart_type = 'bar'
        ```
        
        CRITICAL: Always apply the TOP 10 rule for queries asking about "most", "common", "frequent", "highest", "popular", "top" etc.
        Generate clean, executable Python code that directly answers the user's query with proper top-N limiting.
        """
        
        try:
            code_response = await self.gemini_chat_completion(system_prompt, f"Query: {query}")
            
            # Execute the generated code
            execution_result = await self._execute_code_safely(code_response)
            
            return {
                "generated_code": code_response,
                "execution_result": execution_result,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"Error in action step: {str(e)}")
            return {
                "generated_code": "",
                "execution_result": {"error": str(e)},
                "reasoning": reasoning
            }
    
    async def _execute_code_safely(self, code: str) -> Dict[str, Any]:
        """Execute code with comprehensive error handling"""
        try:
            # Clean the code
            code = code.strip().strip('```python').strip('```').strip()
            
            # Debug: Print the generated code
            logger.info(f"Executing generated code:\n{code}")
            
            # Set up execution environment
            exec_globals = {
                **self.dataframes,
                'pd': pd,
                'np': np,
                'Counter': Counter,
                'defaultdict': defaultdict,
                're': re,
                'analysis_result': None,
                'viz_data': None,
                'chart_type': None
            }
            
            # Execute code
            exec(code, exec_globals)
            
            # Debug: Check what was produced
            analysis_result = exec_globals.get('analysis_result')
            viz_data = exec_globals.get('viz_data')
            chart_type = exec_globals.get('chart_type')
            
            logger.info(f"Execution results - analysis_result type: {type(analysis_result)}, viz_data type: {type(viz_data)}, chart_type: {chart_type}")
            
            if analysis_result is not None:
                logger.info(f"Analysis result shape/content: {getattr(analysis_result, 'shape', 'no shape')} / {str(analysis_result)[:200]}...")
            
            # Extract results
            result = {
                'analysis_result': analysis_result,
                'viz_data': viz_data,
                'chart_type': chart_type,
                'error': None,
                'executed_code': code
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Code execution error: {str(e)}")
            logger.error(f"Code that failed:\n{code}")
            
            # Try to fix common issues and re-execute
            fixed_result = await self._attempt_code_fixes(code, str(e))
            if fixed_result:
                return fixed_result
            
            return {
                'analysis_result': None,
                'viz_data': None,
                'chart_type': None,
                'error': str(e),
                'executed_code': code
            }
    
    async def _attempt_code_fixes(self, original_code: str, error_msg: str) -> Optional[Dict[str, Any]]:
        """Attempt to fix common code issues"""
        fixes_applied = []
        fixed_code = original_code
        
        try:
            # Fix 1: Variable name issues
            if "not defined" in error_msg:
                var_match = re.search(r"name '([^']*)' is not defined", error_msg)
                if var_match:
                    missing_var = var_match.group(1)
                    # Try to find similar variable names
                    all_vars = re.findall(r'(\w+)\s*=', fixed_code)
                    similar_vars = [v for v in all_vars if v.lower() in missing_var.lower() or missing_var.lower() in v.lower()]
                    if similar_vars:
                        fixed_code = fixed_code.replace(missing_var, similar_vars[0])
                        fixes_applied.append(f"Replaced {missing_var} with {similar_vars[0]}")
            
            # Fix 2: DataFrame method issues
            if "has no attribute" in error_msg:
                # Common pandas fixes
                fixed_code = fixed_code.replace('.str.contains(', '.str.contains(na=False, ')
                fixes_applied.append("Added na=False to str.contains")
            
            # Fix 3: Add missing imports or methods
            if "Counter" in fixed_code and "from collections import Counter" not in fixed_code:
                fixed_code = "from collections import Counter\n" + fixed_code
                fixes_applied.append("Added Counter import")
            
            if fixes_applied:
                logger.info(f"Applied fixes: {', '.join(fixes_applied)}")
                return await self._execute_code_safely(fixed_code)
            
        except Exception as e:
            logger.error(f"Error in code fixing: {str(e)}")
        
        return None
    
    async def _observe_and_visualize(self, query: str, action_result: Dict[str, Any]) -> Dict[str, Any]:
        """OBSERVE: Analyze results and create visualizations with improved response formatting"""
        
        execution_result = action_result.get('execution_result', {})
        analysis_result = execution_result.get('analysis_result')
        viz_data = execution_result.get('viz_data')
        chart_type = execution_result.get('chart_type')
        error = execution_result.get('error')
        generated_code = action_result.get('generated_code', '')
        
        # Debug logging
        logger.info(f"Observe phase - error: {error}, analysis_result: {type(analysis_result)}, viz_data: {type(viz_data)}")
        
        if error:
            return {
                "answer": f"Error executing analysis: {error}\n\nGenerated code:\n```python\n{generated_code}\n```",
                "visualization": None,
                "metadata": {
                    "generated_code": generated_code,
                    "reasoning": action_result.get('reasoning', {}),
                    "error": error
                }
            }
        
        # Check if we got results
        if analysis_result is None:
            # Try to provide some basic analysis as fallback
            fallback_answer = self._create_fallback_analysis(query)
            return {
                "answer": f"Generated code executed but no results were produced. Here's what we can determine:\n\n{fallback_answer}\n\nGenerated code:\n```python\n{generated_code}\n```",
                "visualization": None,
                "metadata": {
                    "generated_code": generated_code,
                    "reasoning": action_result.get('reasoning', {}),
                    "fallback_used": True
                }
            }
        
        # Format the analysis result with better readability
        formatted_answer = await self._format_analysis_result(query, analysis_result)
        
        # Create visualization if data is available
        visualization_html = None
        if viz_data is not None and isinstance(viz_data, pd.DataFrame) and not viz_data.empty:
            try:
                if chart_type and len(viz_data.columns) >= 2:
                    x_col = viz_data.columns[0]
                    y_col = viz_data.columns[1] if len(viz_data.columns) > 1 else None
                    
                    visualization_html = self.visualization_engine.create_visualization(
                        viz_data, 
                        chart_type, 
                        title=f"Visualization for: {query}",
                        x_col=x_col,
                        y_col=y_col
                    )
                    
                    # Fix any template variables in visualization HTML
                    if visualization_html and isinstance(visualization_html, str):
                        # Check for template variables
                        if '${' in visualization_html or '$vizContent' in visualization_html:
                            logger.warning("Template variables detected in visualization HTML. Skipping visualization.")
                            visualization_html = None
                        
                        # Make sure Plotly CDN is included
                        if 'plotly-' in visualization_html and 'cdn.plot.ly' not in visualization_html:
                            logger.info("Adding Plotly CDN reference to visualization")
                            visualization_html = f"""
                            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                            {visualization_html}
                            """
                else:
                    # Auto-determine chart type
                    visualization_html = self.visualization_engine._create_auto_chart(
                        viz_data,
                        f"Analysis Results: {query}"
                    )
                    
                    # Same template variable check
                    if visualization_html and isinstance(visualization_html, str):
                        if '${' in visualization_html or '$vizContent' in visualization_html:
                            logger.warning("Template variables detected in visualization HTML. Skipping visualization.")
                            visualization_html = None
                            
            except Exception as e:
                logger.error(f"Error creating visualization: {str(e)}")
                visualization_html = None
        
        # Improve formatting of the answer text
        formatted_answer = self._enhance_text_formatting(formatted_answer)
        
        return {
            "answer": formatted_answer,
            "visualization": visualization_html,
            "metadata": {
                "generated_code": generated_code,
                "reasoning": action_result.get('reasoning', {}),
                "data_shape": viz_data.shape if viz_data is not None else None,
                "chart_type": chart_type
            }
        }

    
    def _create_fallback_analysis(self, query: str) -> str:
        """Create fallback analysis when main execution fails"""
        try:
            query_lower = query.lower()
            
            # Basic data overview
            if 'patients' in query_lower or 'population' in query_lower:
                if 'df_patients' in self.dataframes:
                    df = self.dataframes['df_patients']
                    total_patients = len(df)
                    return f"Database contains {total_patients:,} total patients in the patients table."
            
            # TBI causes
            if 'cause' in query_lower and 'tbi' in query_lower:
                if 'df_tbi_incidents' in self.dataframes:
                    df = self.dataframes['df_tbi_incidents']
                    if 'injury_from' in df.columns:
                        top_causes = df['injury_from'].value_counts().head(5)
                        causes_text = "\n".join([f"- {cause}: {count} cases" for cause, count in top_causes.items()])
                        return f"Top 5 TBI causes from {len(df):,} incidents:\n{causes_text}"
            
            # Age/demographics
            if 'age' in query_lower:
                if 'df_patients' in self.dataframes:
                    df = self.dataframes['df_patients']
                    if 'age' in df.columns:
                        age_stats = df['age'].describe()
                        return f"Age statistics: Mean = {age_stats['mean']:.1f}, Median = {age_stats['50%']:.1f}, Range = {age_stats['min']:.0f}-{age_stats['max']:.0f}"
            
            return "Unable to process query automatically. Please check the generated code above."
            
        except Exception as e:
            return f"Fallback analysis also failed: {str(e)}"
        

    def _enhance_text_formatting(self, text: str) -> str:
        """Enhance the formatting of text for better readability"""
        try:
            # Convert raw pipe tables to HTML tables
            if '|' in text and not '<table' in text:
                lines = text.split('\n')
                table_lines = []
                non_table_lines = []
                in_table = False
                table_html = ''
                
                for line in lines:
                    if line.strip().startswith('|') and line.strip().endswith('|'):
                        if not in_table:
                            in_table = True
                            table_html = '<div class="table-container"><table class="markdown-table">'
                            # Check if this is the header row
                            cells = [c.strip() for c in line.strip('|').split('|')]
                            table_html += '<thead><tr>'
                            for cell in cells:
                                table_html += f'<th>{cell}</th>'
                            table_html += '</tr></thead><tbody>'
                        else:
                            # Skip separator rows like |-----|-----|
                            if not re.match(r'^\|\s*[-:\s]+\|', line):
                                cells = [c.strip() for c in line.strip('|').split('|')]
                                table_html += '<tr>'
                                for cell in cells:
                                    table_html += f'<td>{cell}</td>'
                                table_html += '</tr>'
                    else:
                        if in_table:
                            table_html += '</tbody></table></div>'
                            table_lines.append(table_html)
                            in_table = False
                        non_table_lines.append(line)
                
                if in_table:
                    table_html += '</tbody></table></div>'
                    table_lines.append(table_html)
                
                # Rebuild text with HTML tables
                if table_lines:
                    # Find where to insert the table
                    table_marker = next((i for i, line in enumerate(non_table_lines) 
                                    if 'table' in line.lower() or 'result' in line.lower()), 
                                    len(non_table_lines) - 1)
                                    
                    result_lines = non_table_lines[:table_marker+1]
                    result_lines.extend(table_lines)
                    result_lines.extend(non_table_lines[table_marker+1:])
                    text = '\n'.join(result_lines)
            
            # Improve bullet point formatting
            if '•' in text and '<ul>' not in text:
                bullet_pattern = r'^\s*•\s+(.*?)$'
                lines = text.split('\n')
                in_list = False
                result_lines = []
                
                for i, line in enumerate(lines):
                    bullet_match = re.match(bullet_pattern, line)
                    if bullet_match:
                        content = bullet_match.group(1)
                        if not in_list:
                            in_list = True
                            result_lines.append('<ul>')
                        result_lines.append(f'<li>{content}</li>')
                    else:
                        if in_list:
                            in_list = False
                            result_lines.append('</ul>')
                        result_lines.append(line)
                
                if in_list:
                    result_lines.append('</ul>')
                
                text = '\n'.join(result_lines)
            
            # Format "Key Findings" as a visually distinct element
            if 'Key Findings:' in text and '<div class="key-findings">' not in text:
                text = text.replace('Key Findings:', '<div class="key-findings"><strong>Key Findings:</strong>')
                
                # Find where to close the div
                lines = text.split('\n')
                for i, line in enumerate(lines):
                    if '<div class="key-findings">' in line:
                        # Look for the next empty line or the end
                        for j in range(i+1, len(lines)):
                            if not lines[j].strip():
                                lines[j] = '</div>' + lines[j]
                                break
                        else:
                            # If no empty line found, add div close at the end of text
                            lines.append('</div>')
                        break
                
                text = '\n'.join(lines)
            
            return text
        except Exception as e:
            logger.error(f"Error enhancing text formatting: {e}")
            return text
    
    async def _format_analysis_result(self, query: str, result: Any) -> str:
        """Format the analysis result for user presentation with improved table formatting"""
        if result is None:
            return "No results found for your query."
        
        try:
            if isinstance(result, pd.DataFrame):
                if result.empty:
                    return "No data found matching your criteria."
                elif len(result) > 20:
                    # For large tables, create a proper HTML table with headers
                    # First 10 rows only
                    header_row = '<tr>' + ''.join([f'<th>{col}</th>' for col in result.columns]) + '</tr>'
                    data_rows = []
                    
                    # Format each row
                    for _, row in result.head(10).iterrows():
                        row_html = '<tr>'
                        for val in row:
                            # Format numbers nicely
                            if isinstance(val, (int, float)):
                                if isinstance(val, float):
                                    # Format with commas for thousands and 2 decimal places for floats
                                    cell_content = f"{val:,.2f}"
                                else:
                                    # Format with commas for thousands for integers
                                    cell_content = f"{val:,}"
                            else:
                                cell_content = str(val)
                            row_html += f'<td>{cell_content}</td>'
                        row_html += '</tr>'
                        data_rows.append(row_html)
                    
                    # Combine into table
                    table_html = f'''
                    <div class="table-container">
                    <table class="markdown-table">
                        <thead>{header_row}</thead>
                        <tbody>{''.join(data_rows)}</tbody>
                    </table>
                    </div>
                    '''
                    
                    return f"## Analysis Results\n\nShowing first 10 of {len(result)} results:\n\n{table_html}\n\n*Note: Results truncated for readability*"
                else:
                    # For smaller tables, still use HTML table
                    header_row = '<tr>' + ''.join([f'<th>{col}</th>' for col in result.columns]) + '</tr>'
                    data_rows = []
                    
                    # Format each row
                    for _, row in result.iterrows():
                        row_html = '<tr>'
                        for val in row:
                            # Format numbers nicely
                            if isinstance(val, (int, float)):
                                if isinstance(val, float):
                                    # Format with commas for thousands and 2 decimal places for floats
                                    cell_content = f"{val:,.2f}"
                                else:
                                    # Format with commas for thousands for integers
                                    cell_content = f"{val:,}"
                            else:
                                cell_content = str(val)
                            row_html += f'<td>{cell_content}</td>'
                        row_html += '</tr>'
                        data_rows.append(row_html)
                    
                    # Combine into table
                    table_html = f'''
                    <div class="table-container">
                    <table class="markdown-table">
                        <thead>{header_row}</thead>
                        <tbody>{''.join(data_rows)}</tbody>
                    </table>
                    </div>
                    '''
                    
                    return f"## Analysis Results\n\n{table_html}"
            
            elif isinstance(result, pd.Series):
                if result.empty:
                    return "No data found matching your criteria."
                else:
                    # Convert Series to DataFrame for better display
                    df = pd.DataFrame({result.name or "Value": result})
                    
                    # Create HTML table
                    header_row = '<tr><th>Index</th><th>Value</th></tr>'
                    data_rows = []
                    
                    # Format each row
                    for idx, val in result.items():
                        # Format numbers nicely
                        if isinstance(val, (int, float)):
                            if isinstance(val, float):
                                # Format with commas and 2 decimal places
                                val_formatted = f"{val:,.2f}"
                            else:
                                # Format with commas
                                val_formatted = f"{val:,}"
                        else:
                            val_formatted = str(val)
                            
                        data_rows.append(f'<tr><td>{idx}</td><td>{val_formatted}</td></tr>')
                    
                    # Combine into table
                    table_html = f'''
                    <div class="table-container">
                    <table class="markdown-table">
                        <thead>{header_row}</thead>
                        <tbody>{''.join(data_rows)}</tbody>
                    </table>
                    </div>
                    '''
                    
                    return f"## Analysis Results\n\n{table_html}"
            
            elif isinstance(result, (int, float)):
                return f"## Analysis Result\n\n**Answer:** {result:,.2f}" if isinstance(result, float) else f"**Answer:** {result:,}"
            
            elif isinstance(result, str):
                return f"## Analysis Result\n\n{result}"
            
            elif isinstance(result, dict):
                formatted_dict = "\n".join([f"• **{k}:** {v}" for k, v in result.items()])
                return f"## Analysis Results\n\n{formatted_dict}"
            
            elif isinstance(result, list):
                if len(result) > 10:
                    formatted_list = "\n".join([f"• {item}" for item in result[:10]])
                    return f"## Analysis Results\n\nShowing first 10 of {len(result)} items:\n\n{formatted_list}\n\n*Results truncated for readability*"
                else:
                    formatted_list = "\n".join([f"• {item}" for item in result])
                    return f"## Analysis Results\n\n{formatted_list}"
            
            else:
                return f"## Analysis Result\n\n{str(result)}"
                
        except Exception as e:
            logger.error(f"Error formatting result: {str(e)}")
            return f"Analysis completed, but error formatting results: {str(e)}"
    
    def get_comprehensive_schema_info(self) -> str:
        """Get comprehensive schema information"""
        if not self.schema_analyzer:
            return "Schema analysis not available. Please load data first."
        
        return self.schema_analyzer.get_comprehensive_schema()
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Main entry point for processing queries"""
        logger.info(f"Processing query: {query}")
        
        # Use ReAct framework for processing
        result = await self.react_process_query(query)
        
        # Log the interaction
        self.execution_history.append({
            'query': query,
            'result': result,
            'timestamp': pd.Timestamp.now()
        })
        
        return result
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get history of executed queries"""
        return self.execution_history

# Helper function to generate query descriptions using LLM
async def generate_query_description(agent, query: str) -> str:
    """Generate a description for a query based on TBI data schema"""
    try:
        description_prompt = f"""
        You are analyzing a natural language query about TBI (Traumatic Brain Injury) research data.
        
        Query: "{query}"
        
        Available TBI Database Schema:
        {DB_SCHEMA}
        
        Based on this query and the available TBI data, generate a brief, technical description of what this query is asking for. 
        Focus on:
        1. Which tables/columns would be involved
        2. What type of analysis this represents
        3. What insights it might provide
        
        Keep the description concise (1-2 sentences) and technical.
        
        Example format: "Analysis of [specific columns] from [tables] to determine [insight type]"
        """
        
        description = await agent.gemini_chat_completion(description_prompt, "Generate a technical description for this query")
        
        # Clean up the description
        description = description.strip().strip('"').strip("'")
        if not description or len(description) < 10:
            description = f"Analysis of TBI data to answer: {query}"
        
        return description
        
    except Exception as e:
        logger.warning(f"Could not generate description for query: {str(e)}")
        return f"TBI data analysis query: {query[:50]}..."

# Main function to test with your TBI data
async def main():
    """Main function to test Enhanced PandasAgent with your TBI database"""
    
    print("🧠 Enhanced PandasAgent - TBI Data Analysis")
    print("=" * 60)
    
    # Initialize agent with Gemini API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("⚠️  GEMINI_API_KEY not found in environment variables")
        print("Please set it with: export GEMINI_API_KEY=your_api_key")
        return
    
    agent = EnhancedPandasAgent(gemini_api_key=gemini_api_key)
    
    # Connect to your TBI database
    print("🔌 Connecting to TBI database...")
    success = agent.connect_to_database()
    
    if not success:
        print("❌ Failed to connect to database. Please check your database configuration.")
        print("Required environment variables:")
        print("  - user (or DB_USER)")
        print("  - password (or DB_PASSWORD)")
        print("  - host (or DB_HOST)")
        print("  - port (or DB_PORT)")  
        print("  - dbname (or DB_NAME)")
        return
    
    print("✅ Successfully connected and loaded TBI data!")
    
    # Create visualizations directory
    viz_dir = "tbi_visualizations"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
        print(f"📁 Created visualization directory: {viz_dir}/")
    
    # Show schema information
    print("\n" + "="*60)
    print("📊 DATABASE SCHEMA ANALYSIS")
    print("="*60)
    print(agent.get_comprehensive_schema_info())
    
    # Define real TBI-related test queries (just the queries now)
    tbi_queries = [
        "What are the most common causes of TBI in our patient population?",
        "Show the age distribution of TBI patients by gender",
        "What are the most frequent immediate symptoms after TBI?",
        "Which states have the highest number of TBI patients?",
        "What's the relationship between injury location and symptom severity?",
        "What therapies are most commonly prescribed for TBI patients?",
        "How does veteran status relate to TBI causes?",
        "What percentage of patients had TBI before their current incident?",
        "Show symptom categories by frequency and average severity",
        "What are the top social determinants affecting TBI patients?"
    ]
    
    print(f"\n{'='*60}")
    print("🧪 RUNNING TBI DATA ANALYSIS QUERIES")
    print("="*60)
    print("Choose an option:")
    print("1. Run all test queries automatically")
    print("2. Run specific queries interactively")
    print("3. Interactive mode - enter your own queries")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        # Run all queries automatically
        for i, query in enumerate(tbi_queries, 1):
            print(f"\n{'-'*50}")
            print(f"🔍 QUERY {i}: {query}")
            print(f"{'-'*50}")
            
            # Generate description using LLM
            print("🤖 Generating query description...")
            description = await generate_query_description(agent, query)
            print(f"📝 Description: {description}")
            
            try:
                result = await agent.process_query(query)
                
                print("\n📊 ANALYSIS RESULT:")
                print(result['answer'])
                
                if result['visualization']:
                    print(f"\n📈 VISUALIZATION: Available (HTML content - {len(result['visualization'])} characters)")
                    # Save visualization to folder
                    filename = f"tbi_analysis_{i}.html"
                    filepath = os.path.join(viz_dir, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(result['visualization'])
                    print(f"💾 Visualization saved to: {filepath}")
                    print("💡 Open the HTML file in your browser to view the interactive chart")
                
                if result['metadata'].get('reasoning'):
                    print(f"\n🧠 REASONING: {result['metadata']['reasoning']}")
                
                print(f"\n✅ Query {i} completed successfully")
                
            except Exception as e:
                print(f"\n❌ Error processing query {i}: {str(e)}")
            
            # Wait for user to continue
            if i < len(tbi_queries):
                input("\nPress Enter to continue to next query...")
    
    elif choice == "2":
        # Interactive query selection
        print("\n🔍 Generating descriptions for available queries...")
        query_descriptions = []
        
        for i, query in enumerate(tbi_queries):
            description = await generate_query_description(agent, query)
            query_descriptions.append(description)
            print(f"   {i+1}. Processing query {i+1}...")
        
        print("\nAvailable TBI Analysis Queries:")
        for i, (query, description) in enumerate(zip(tbi_queries, query_descriptions), 1):
            print(f"{i:2}. {description}")
        
        while True:
            try:
                selection = input(f"\nSelect query number (1-{len(tbi_queries)}) or 'exit': ").strip()
                
                if selection.lower() == 'exit':
                    break
                
                query_idx = int(selection) - 1
                if 0 <= query_idx < len(tbi_queries):
                    selected_query = tbi_queries[query_idx]
                    selected_description = query_descriptions[query_idx]
                    
                    print(f"\n🔍 Running: {selected_description}")
                    print(f"Query: \"{selected_query}\"")
                    
                    result = await agent.process_query(selected_query)
                    
                    print("\n📊 ANALYSIS RESULT:")
                    print(result['answer'])
                    
                    if result['visualization']:
                        filename = f"tbi_query_{query_idx+1}.html"
                        filepath = os.path.join(viz_dir, filename)
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(result['visualization'])
                        print(f"\n📈 Visualization saved to: {filepath}")
                else:
                    print("Invalid selection. Please try again.")
                    
            except ValueError:
                print("Please enter a valid number.")
            except Exception as e:
                print(f"Error: {str(e)}")
    
    elif choice == "3":
        # Interactive mode for custom queries
        print("\n" + "="*60)
        print("💬 INTERACTIVE TBI DATA ANALYSIS")
        print("="*60)
        print("Enter your natural language queries about the TBI data.")
        print("Available tables: patients, tbi_incidents, symptom_logs, worst_symptoms, therapies, social_determinants, symptom_reference")
        print("Type 'schema' to see detailed table structure, or 'exit' to quit")
        
        while True:
            try:
                user_query = input("\n🔍 Enter your TBI analysis query: ").strip()
                
                if user_query.lower() == 'exit':
                    print("👋 Goodbye!")
                    break
                elif user_query.lower() == 'schema':
                    print(agent.get_comprehensive_schema_info())
                    continue
                elif not user_query:
                    continue
                
                # Generate description for the user query
                print("🤖 Analyzing query...")
                description = await generate_query_description(agent, user_query)
                print(f"📝 Query Understanding: {description}")
                
                print(f"\n⚙️  Processing: {user_query}")
                result = await agent.process_query(user_query)
                
                print("\n📊 ANALYSIS RESULT:")
                print(result['answer'])
                
                if result['visualization']:
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"tbi_custom_{timestamp}.html"
                    filepath = os.path.join(viz_dir, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(result['visualization'])
                    print(f"\n📈 Visualization saved to: {filepath}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    else:
        print("Invalid choice. Exiting.")
    
    print(f"\n{'='*60}")
    print("🎯 TBI DATA ANALYSIS COMPLETED")
    print("="*60)
    
    # Show execution history and file summary
    history = agent.get_execution_history()
    if history:
        print(f"\n📈 Executed {len(history)} queries in this session")
        print("Query history saved in agent.execution_history")
    
    # Show generated files
    if os.path.exists(viz_dir):
        viz_files = [f for f in os.listdir(viz_dir) if f.endswith('.html')]
        if viz_files:
            print(f"\n📁 Generated {len(viz_files)} visualization files in '{viz_dir}/' folder:")
            for file in sorted(viz_files):
                print(f"   - {file}")
            print(f"\n💡 Open any HTML file in your browser to view interactive charts")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the main function
    asyncio.run(main())