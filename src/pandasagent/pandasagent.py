import pandas as pd
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database schema definition based on Supabase tables
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

# System prompt for the PandasAgent
PANDAS_AGENT_SYSTEM_PROMPT = f"""
You are a helpful assistant that converts natural language queries about TBI (Traumatic Brain Injury) patient data into Python pandas code.

The data is spread across multiple DataFrame objects with the following schema:

{DB_SCHEMA}

Your task is to generate Python code that performs the requested analysis by:
1. Using the appropriate DataFrame(s) based on the query
2. Joining tables when necessary to analyze across different aspects of the data
3. Applying appropriate filtering, grouping, and aggregation
4. Returning a concise, informative result

IMPORTANT: Variable names must be consistent throughout your code. Double-check that all variable names you reference later in the code are exactly the same as where they were first defined. Be especially careful with pluralization (count vs counts) and abbreviations.

Only output Python code (no explanations) that directly answers the user's query.
Do not include import statements or print statements.
The answer should be a string that can be returned to the user and stored in a variable called 'final_answer'.

For any query about listing items where the result might be very large (like listing all symptoms, all therapies, etc.), 
make sure to include appropriate processing in your code to summarize the data. For example:
- When listing categories, include counts or percentages
- For text data, count occurrences and show the top 10-15 most common items
- Group similar items when possible
- Use value_counts() and proper formatting to create readable summaries

Examples of operations your code might perform:
- Join patients with tbi_incidents tables to analyze demographic patterns in TBI causes
- Analyze symptom prevalence across different injury types
- Compare therapy effectiveness for different TBI types
- Investigate correlations between social determinants and TBI outcomes

Code Example 1:
```python
# Query: "What are the most common causes of TBI?"
tbi_counts = df_tbi_incidents['injury_from'].value_counts().reset_index()
tbi_counts.columns = ['Cause', 'Count']
tbi_counts['Percentage'] = (tbi_counts['Count'] / tbi_counts['Count'].sum() * 100).round(2)
final_answer = tbi_counts.to_markdown(index=False)
```

Code Example 2:
```python
# Query: "Show average patient age by TBI cause"
merged_df = df_patients.merge(df_tbi_incidents, on='patient_id')
avg_age = merged_df.groupby('injury_from')['age'].mean().reset_index()
avg_age.columns = ['TBI Cause', 'Average Age']
final_answer = avg_age.to_markdown(index=False)
```

Code Example 3:
```python
# Query: "What percentage of patients with TBI from sports had loss of consciousness?"
sports_tbi = df_tbi_incidents[df_tbi_incidents['injury_from'] == 'Sports']
loss_of_consciousness = sports_tbi[sports_tbi['immediate_symptoms_resulting'].str.contains('Loss of Consciousness', case=False, na=False)]
percentage = (len(loss_of_consciousness) / len(sports_tbi) * 100) if len(sports_tbi) > 0 else 0
final_answer = f"Percentage of sports-related TBI patients who experienced loss of consciousness: 'percentage:.2f'%"
```

Code Example 4:
```python
# Query: "What therapies are most commonly used?"
therapy_counts = df_therapies['therapies'].value_counts().reset_index()
therapy_counts.columns = ['Therapy', 'Count']
therapy_counts['Percentage'] = (therapy_counts['Count'] / therapy_counts['Count'].sum() * 100).round(2)
final_answer = therapy_counts.to_markdown(index=False)
```

Code Example 5:
```python
# Query: "How many user types are there?"
user_types_count = df_patients['user_type'].nunique()
final_answer = f"There are 'user_types_count' different user types in the patients database."
```

Code Example 6:
```python
# Query: "List all user types"
user_types = df_patients['user_type'].value_counts().reset_index()
user_types.columns = ['User Type', 'Count']
user_types['Percentage'] = (user_types['Count'] / user_types['Count'].sum() * 100).round(2)
final_answer = user_types.to_markdown(index=False)
```

Code Example 7:
```python
# Query: "What are the most common immediate symptoms?"
# Split the comma-separated symptoms and count them individually
import re
all_symptoms = []
for symptoms_str in df_tbi_incidents['immediate_symptoms_resulting'].dropna():
    if isinstance(symptoms_str, str):
        # Split by comma or comma+space
        symptoms = re.split(r',\s*', symptoms_str)
        all_symptoms.extend([s.strip() for s in symptoms if s.strip()])

# Count occurrences of each symptom
from collections import Counter
symptom_counts = Counter(all_symptoms)

# Get the top 15 most common symptoms
top_symptoms = pd.DataFrame(symptom_counts.most_common(15), columns=['Symptom', 'Count'])
top_symptoms['Percentage'] = (top_symptoms['Count'] / len(all_symptoms) * 100).round(2)

final_answer = f"Top 15 Most Common Immediate Symptoms (out of 'len(symptom_counts)' unique symptoms):\\n" + top_symptoms.to_markdown(index=False)
```
"""

class PandasAgent:
    """
    Class for processing natural language queries about TBI data using pandas operations
    and Google's Gemini model for code generation.
    """
    
    def __init__(self, gemini_api_key: str = None):
        """
        Initialize the PandasAgent
        
        Args:
            gemini_api_key (str, optional): API key for Google's Gemini model
        """
        self.dataframes = {}
        self.gemini_api_key = gemini_api_key
        self.db_connection = None
        
        # Configure Gemini if API key is provided
        if gemini_api_key:
            self._configure_gemini(gemini_api_key)
    
    def _configure_gemini(self, api_key: str):
        """
        Configure the Gemini AI model with API key and safety settings
        
        Args:
            api_key (str): Gemini API key
        """
        genai.configure(api_key=api_key)
        
        # Configure safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        logger.info("Gemini AI configured successfully with safety settings")
    
    def connect_to_database(self, db_config: Dict[str, str] = None):
        """
        Connect to the Supabase PostgreSQL database
        
        Args:
            db_config (Dict[str, str], optional): Database configuration parameters.
                If None, will try to load from environment variables.
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # If db_config not provided, try to load from environment
            if db_config is None:
                load_dotenv()
                db_config = {
                    'user': os.getenv("user"),
                    'password': os.getenv("password"),
                    'host': os.getenv("host"),
                    'port': os.getenv("port"),
                    'dbname': os.getenv("dbname")
                }
            
            # Check if all required config values are present
            required_keys = ['user', 'password', 'host', 'port', 'dbname']
            missing_keys = [key for key in required_keys if not db_config.get(key)]
            
            if missing_keys:
                logger.error(f"Missing database configuration keys: {missing_keys}")
                return False
            
            # Connect to the database
            self.db_connection = psycopg2.connect(
                user=db_config['user'],
                password=db_config['password'],
                host=db_config['host'],
                port=db_config['port'],
                dbname=db_config['dbname']
            )
            
            logger.info("Successfully connected to the database")
            
            # Load all necessary tables
            self.load_data_from_database()
            
            return True
            
        except Exception as e:
            logger.exception(f"Failed to connect to database: {str(e)}")
            return False
    
    def load_data_from_database(self):
        """
        Load all necessary tables from the connected database
        
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        if not self.db_connection:
            logger.error("Database connection not established")
            return False
        
        try:
            # Table names to load
            tables = [
                'patients', 
                'tbi_incidents', 
                'symptom_logs', 
                'worst_symptoms', 
                'therapies', 
                'social_determinants', 
                'symptom_reference'
            ]
            
            dataframes = {}
            
            # Load each table into a dataframe
            for table in tables:
                query = f"SELECT * FROM {table};"
                
                df = pd.read_sql_query(query, self.db_connection)
                dataframes[table] = df
                
                logger.info(f"Loaded table '{table}' with {len(df)} rows and {len(df.columns)} columns")
            
            # Store dataframes with standardized prefixes
            for table_name, df in dataframes.items():
                self.dataframes[f"df_{table_name}"] = df
            
            return True
            
        except Exception as e:
            logger.exception(f"Error loading data from database: {str(e)}")
            return False
    
    def load_dataframes(self, dataframes: Dict[str, pd.DataFrame]) -> bool:
        """
        Load multiple dataframes for analysis
        
        Args:
            dataframes (Dict[str, pd.DataFrame]): Dictionary of dataframes with table name as key
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate required tables
            required_tables = ['patients', 'tbi_incidents', 'symptom_logs', 
                               'worst_symptoms', 'therapies', 'social_determinants', 
                               'symptom_reference']
            
            missing_tables = [table for table in required_tables if table not in dataframes]
            if missing_tables:
                logger.warning(f"Missing tables: {missing_tables}. Some queries may not work properly.")
            
            # Store dataframes with standardized prefixes
            for table_name, df in dataframes.items():
                self.dataframes[f"df_{table_name}"] = df
                logger.info(f"Loaded dataframe '{table_name}' with {len(df)} rows and {len(df.columns)} columns")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error loading dataframes: {str(e)}")
            return False
    
    async def gemini_chat_completion(self, prompt: str, user_query: str) -> str:
        """
        Generate chat completion using Google's Gemini model
        
        Args:
            prompt (str): System prompt/instructions
            user_query (str): User's query
            
        Returns:
            str: Generated response
        """
        try:
            if not self.gemini_api_key:
                return "Error: Gemini API key not configured"
            
            # Initialize the model
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config={"temperature": 0.1, "top_p": 0.95, "top_k": 10},
                safety_settings=self.safety_settings
            )
            
            # Create chat session
            chat = model.start_chat(history=[])
            
            # Send system prompt
            system_message = {"role": "system", "content": prompt}
            chat.send_message(system_message["content"])
            
            # Send user query
            response = chat.send_message(user_query)
            
            # Extract the code from the response
            response_text = response.text
            
            # If response is wrapped in code blocks, extract just the code
            if "```python" in response_text and "```" in response_text:
                code = response_text.split("```python")[1].split("```")[0].strip()
            elif "```" in response_text:
                code = response_text.split("```")[1].split("```")[0].strip()
            else:
                code = response_text.strip()
                
            return code
            
        except Exception as e:
            logger.exception(f"Error with Gemini chat completion: {str(e)}")
            return f"Error with Gemini API: {str(e)}"
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query about TBI data using pandas operations
        
        Args:
            query (str): Natural language query about the TBI data
            
        Returns:
            Dict[str, Any]: Dictionary containing the answer and metadata
        """
        if not self.dataframes:
            return {"answer": "Error: No data loaded. Please load data first.", "metadata": []}
        
        if not self.gemini_api_key:
            return {"answer": "Error: Gemini API key not configured.", "metadata": []}
        
        try:
            # Get the code from Gemini
            system_msg = PANDAS_AGENT_SYSTEM_PROMPT
            user_msg = f"Question: {query}\n\nPlease ensure all variable names are consistent throughout your code."
            
            llm_response = await self.gemini_chat_completion(system_msg, user_msg)
            
            # Clean up LLM response
            code = llm_response.strip("```python").strip("```").strip()
            logger.info(f"Generated code: {code}")
            
            # Preprocess the code to fix common variable issues
            code = self._preprocess_code(code)
            
            final_answer = None
            
            try:
                # Setup globals with all available dataframes
                exec_globals = {**self.dataframes, "final_answer": final_answer, "pd": pd}
                
                # Find all variable definitions in the code
                import re
                var_defs = re.findall(r'(\w+)\s*=', code)
                
                # Pre-define all variables to prevent NameError
                for var in var_defs:
                    exec_globals[var] = None
                
                # Execute the preprocessed code
                try:
                    exec(code, exec_globals)
                    final_answer = exec_globals.get("final_answer", None)
                    
                    if final_answer is None:
                        final_answer = "There was no valid response generated for your query."
                    
                except NameError as e:
                    # Handle variable name errors
                    error_var_match = re.search(r"name '([^']*)' is not defined", str(e))
                    if error_var_match:
                        missing_var = error_var_match.group(1)
                        
                        # Try to find the closest variable name
                        closest_var = self._find_closest_variable(missing_var, var_defs)
                        
                        if closest_var:
                            # Fix the code by replacing the missing variable with the closest match
                            fixed_code = code.replace(missing_var, closest_var)
                            
                            # Try executing the fixed code
                            exec_fixed_globals = {**self.dataframes, "final_answer": final_answer, "pd": pd}
                            for var in var_defs:
                                exec_fixed_globals[var] = None
                                
                            try:
                                exec(fixed_code, exec_fixed_globals)
                                final_answer = exec_fixed_globals.get("final_answer", None)
                                
                                if final_answer is None:
                                    final_answer = "There was no valid response generated for your query."
                            except Exception as e2:
                                final_answer = f"Error in pandas operation: {str(e2)}"
                        else:
                            final_answer = f"Error in pandas operation: {str(e)}"
                    else:
                        final_answer = f"Error in pandas operation: {str(e)}"
                
                except Exception as e:
                    final_answer = f"Error in pandas operation: {str(e)}"
                
                # Format the final answer
                if isinstance(final_answer, pd.DataFrame):
                    if final_answer.empty:
                        final_answer = "No results found for your query."
                    else:
                        final_answer = final_answer.to_markdown(index=False)
                elif isinstance(final_answer, pd.Series):
                    if final_answer.empty:
                        final_answer = "No results found for your query."
                    elif len(final_answer) == 1:
                        final_answer = str(final_answer.iloc[0])
                    else:
                        final_answer = final_answer.to_markdown()
                
            except Exception as e:
                logger.exception("Error executing pandas code.")
                return {"answer": f"Error executing code: {str(e)}", "metadata": []}
            
            logger.info(f"Final answer before parsing: {final_answer}")
            
            # Format the response
            formatted_response = await self.parse_response(query, final_answer)
            return {"answer": formatted_response, "metadata": []}
            
        except Exception as e:
            logger.exception(f"Error in process_query: {str(e)}")
            return {"answer": f"Error processing query: {str(e)}", "metadata": []}
    
    def _preprocess_code(self, code: str) -> str:
        """
        Preprocess the code to fix common issues with variable names
        
        Args:
            code (str): Original code
            
        Returns:
            str: Preprocessed code
        """
        import re
        
        # Find all variable definitions
        var_defs = re.findall(r'(\w+)\s*=', code)
        
        # Find all variable usages (excluding definitions)
        var_usages = []
        for line in code.splitlines():
            # Skip lines where the variable is being defined
            if '=' in line:
                continue
                
            # Find all words in the line
            for word in re.findall(r'\b(\w+)\b', line):
                if word not in ['df', 'pd', 'if', 'else', 'for', 'in', 'and', 'or', 'not', 'None', 'True', 'False'] and not word.startswith('df_'):
                    var_usages.append(word)
        
        # Check for variables that are used but not defined
        undefined_vars = [var for var in var_usages if var not in var_defs and var != 'final_answer']
        
        # For each undefined variable, find the most similar defined variable
        replacements = {}
        for undefined in undefined_vars:
            closest = self._find_closest_variable(undefined, var_defs)
            if closest:
                replacements[undefined] = closest
        
        # Apply replacements
        new_code = code
        for old_var, new_var in replacements.items():
            new_code = re.sub(r'\b' + old_var + r'\b', new_var, new_code)
        
        # Special case: check if the first line defines a variable with a missing first letter
        # (like 'bi_counts' instead of 'tbi_counts')
        lines = new_code.splitlines()
        if len(lines) > 0 and '=' in lines[0]:
            first_def = re.findall(r'(\w+)\s*=', lines[0])
            if first_def:
                first_var = first_def[0]
                
                # Look for similar variables in subsequent lines
                for i in range(1, len(lines)):
                    for word in re.findall(r'\b(\w+)\b', lines[i]):
                        if word != first_var and self._is_similar_with_prefix(word, first_var):
                            # Replace all occurrences of the similar variable with the first defined one
                            for j in range(i, len(lines)):
                                lines[j] = re.sub(r'\b' + word + r'\b', first_var, lines[j])
        
        return '\n'.join(lines)
    
    def _find_closest_variable(self, var_name: str, defined_vars: List[str]) -> str:
        """
        Find the closest variable name from the defined variables
        
        Args:
            var_name (str): Variable name to find a match for
            defined_vars (List[str]): List of defined variable names
            
        Returns:
            str: Closest matching variable name, or empty string if no close match found
        """
        if not defined_vars:
            return ""
            
        import difflib
        
        # Find the closest match
        matches = difflib.get_close_matches(var_name, defined_vars, n=1, cutoff=0.6)
        
        if matches:
            return matches[0]
            
        # If no match found with default cutoff, try more specific matching strategies
        
        # Check for prefix/suffix matches (like 'tbi_count' vs 'tbi_counts')
        for defined in defined_vars:
            # Check if one is a prefix of the other
            if defined.startswith(var_name) or var_name.startswith(defined):
                return defined
                
            # Check if they differ only in plural/singular form
            if defined.endswith('s') and defined[:-1] == var_name or var_name.endswith('s') and var_name[:-1] == defined:
                return defined
        
        # Check for partial matches (missing first letter, like 'bi_counts' vs 'tbi_counts')
        for defined in defined_vars:
            if len(defined) > 1 and defined[1:] == var_name or len(var_name) > 1 and var_name[1:] == defined:
                return defined
        
        return ""
    
    def _is_similar_with_prefix(self, var1: str, var2: str) -> bool:
        """
        Check if two variable names are similar, accounting for missing prefixes
        
        Args:
            var1 (str): First variable name
            var2 (str): Second variable name
            
        Returns:
            bool: True if variables are similar, False otherwise
        """
        # Check if one is a prefix of the other
        if var1.startswith(var2) or var2.startswith(var1):
            return True
            
        # Check if one is missing a prefix (like 'bi_counts' vs 'tbi_counts')
        if len(var1) > 2 and len(var2) > 2:
            if var1[1:] == var2 or var2[1:] == var1:
                return True
                
        # Check for pluralization differences
        if var1.endswith('s') and var1[:-1] == var2 or var2.endswith('s') and var2[:-1] == var1:
            return True
            
        # Check for typical abbreviations
        prefixes = ['num_', 'count_', 'avg_', 'mean_', 'total_', 'sum_']
        for prefix in prefixes:
            if var1.startswith(prefix) and var2 == var1[len(prefix):] or var2.startswith(prefix) and var1 == var2[len(prefix):]:
                return True
                
        return False
    
    async def parse_response(self, query: str, raw_response: str) -> str:
        """
        Parse and format the raw response from pandas operations
        
        Args:
            query (str): Original query
            raw_response (str): Raw response from pandas operations
            
        Returns:
            str: Formatted response
        """
        if self.gemini_api_key:
            try:
                # Check for excessively long responses first
                is_too_long = False
                if isinstance(raw_response, str) and len(raw_response) > 1000:
                    # Large text response that likely needs summarization
                    is_too_long = True
                
                if is_too_long:
                    # Special prompt for summarizing long results
                    system_prompt = """
                    You are an assistant tasked with summarizing large data outputs from a query about TBI (Traumatic Brain Injury) data.
                    The raw response is too long to present directly to the user. Your job is to analyze it and create a useful summary.
                    
                    For lists of items:
                    1. Group similar items into categories
                    2. Provide counts or percentages when possible
                    3. Highlight notable patterns or outliers
                    4. Present the information in a clear, structured format with headings
                    
                    For numerical data:
                    1. Provide key statistics (min, max, mean, median)
                    2. Identify trends or patterns
                    3. Focus on the most important insights
                    
                    Keep your summary concise but informative. Use bullet points, tables, or other formatting to improve readability.
                    """
                    
                    user_msg = f"""
                    Original query: {query}
                    
                    The raw response is very long, containing {len(raw_response)} characters. Please analyze and summarize the key information in a format that would be helpful to the user.
                    
                    Here's a sample of the data to help you understand what it contains:
                    {raw_response[:500]}...
                    """
                    
                    formatted_response = await self.gemini_chat_completion(system_prompt, user_msg)
                    return formatted_response
                else:
                    # Standard formatting for normal-length responses
                    system_prompt = """
                    You are an assistant tasked with generating a user-friendly response to a query about TBI (Traumatic Brain Injury) data.
                    The raw response provided is the result of executing a pandas operation on TBI datasets.
                    Use the raw response to craft a clear, concise, and professional answer.
                    If the raw response contains tabular data, ensure it's preserved in a readable format.
                    If no results are found, inform the user appropriately.
                    """
                    
                    user_msg = f"""
                    Original query: {query}
                    
                    Raw response:
                    {raw_response}
                    
                    Please provide a user-friendly response.
                    """
                    
                    formatted_response = await self.gemini_chat_completion(system_prompt, user_msg)
                    return formatted_response
            
            except Exception as e:
                logger.exception(f"Error parsing response with Gemini: {str(e)}")
                return raw_response
        else:
            # If no Gemini API key, return the raw response
            return raw_response
    
    def get_schema_info(self) -> str:
        """
        Get information about the loaded dataframes and their schemas
        
        Returns:
            str: Database schema information
        """
        if not self.dataframes:
            return "No dataframes loaded yet."
        
        schema_info = ["# Database Schema Information"]
        
        for df_name, df in self.dataframes.items():
            table_name = df_name.replace("df_", "")
            schema_info.append(f"\n## {table_name}")
            schema_info.append(f"- Rows: {len(df)}")
            schema_info.append(f"- Columns: {len(df.columns)}")
            
            schema_info.append("\n### Columns:")
            for col in df.columns:
                dtype = df[col].dtype
                missing = df[col].isna().sum()
                missing_pct = (missing / len(df)) * 100
                
                schema_info.append(f"- {col}: {dtype} (Missing: {missing}, {missing_pct:.2f}%)")
        
        return "\n".join(schema_info)

# Example usage
async def main():
    """
    Main function to demonstrate the PandasAgent
    """
    # Initialize PandasAgent with Gemini API key
    agent = PandasAgent(gemini_api_key="your-gemini-api-key")  # Replace with actual API key
    
    # Load data from CSV files
    # agent.load_data_from_files("path/to/data", "csv")
    
    # Alternatively, load data from pandas DataFrames
    # Example dataframes (replace with actual data loading code)
    patients_df = pd.read_csv("patients.csv")
    tbi_incidents_df = pd.read_csv("tbi_incidents.csv")
    symptom_logs_df = pd.read_csv("symptom_logs.csv")
    worst_symptoms_df = pd.read_csv("worst_symptoms.csv")
    therapies_df = pd.read_csv("therapies.csv")
    social_determinants_df = pd.read_csv("social_determinants.csv")
    symptom_reference_df = pd.read_csv("symptom_reference.csv")
    
    dataframes = {
        "patients": patients_df,
        "tbi_incidents": tbi_incidents_df,
        "symptom_logs": symptom_logs_df,
        "worst_symptoms": worst_symptoms_df,
        "therapies": therapies_df, 
        "social_determinants": social_determinants_df,
        "symptom_reference": symptom_reference_df
    }
    
    agent.load_dataframes(dataframes)
    
    # Print schema information
    print(agent.get_schema_info())
    
    # Example queries
    example_queries = [
        "What are the most common causes of TBI in this dataset?",
        "What percentage of patients experienced loss of consciousness?",
        "What's the average age of patients with different types of injuries?",
        "Which symptoms are most commonly reported for sports-related TBIs?"
    ]
    
    for query in example_queries:
        print(f"\n--- QUERY: {query} ---")
        result = await agent.process_query(query)
        print(result["answer"])
    
    # Interactive mode
    print("\n--- INTERACTIVE MODE ---")
    print("Type 'exit' to quit")
    
    while True:
        user_query = input("\nEnter your query: ")
        if user_query.lower() == 'exit':
            break
        
        result = await agent.process_query(user_query)
        print("\nResult:")
        print(result["answer"])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())