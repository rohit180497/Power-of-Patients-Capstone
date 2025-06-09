import asyncio
import os
from dotenv import load_dotenv
from pandasagent import PandasAgent
import argparse

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Power of Patient PandasAgent queries")
    parser.add_argument("--api_key", help="Google Gemini API key")
    parser.add_argument("--query", help="Query to run (if not provided, interactive mode will start)")
    args = parser.parse_args()
    
    # Get API key from arguments or environment variable
    load_dotenv()
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Error: Gemini API key is required. Provide it as an argument or set GEMINI_API_KEY environment variable.")
        return
    
    # Initialize the agent
    agent = PandasAgent(gemini_api_key=api_key)
    
    # Connect to the database using environment variables from .env file
    print("Connecting to the database...")
    if not agent.connect_to_database():
        print("Failed to connect to the database. Please check your database configuration.")
        return
    
    print(f"Successfully connected to the database and loaded data!")
    
    # Print a brief summary of the loaded data
    for df_name, df in agent.dataframes.items():
        table_name = df_name.replace("df_", "")
        print(f"- {table_name}: {len(df)} rows, {len(df.columns)} columns")
    
    # Run a single query if provided
    if args.query:
        print(f"\nRunning query: {args.query}")
        result = await agent.process_query(args.query)
        print("\nResult:")
        print(result["answer"])
        return
    
    # Interactive mode
    print("\nEntering interactive mode. Type 'exit' to quit.\n")
    print("Example queries:")
    print("1. What are the most common causes of TBI in this dataset?")
    print("2. What's the average age of patients with different types of injuries?")
    print("3. Which symptoms have the highest average severity?")
    print("4. How many patients experienced loss of consciousness?")
    
    while True:
        try:
            user_query = input("\nEnter your query (or 'exit' to quit): ")
            if user_query.lower() in ['exit', 'quit', 'q']:
                break
            
            print("\nProcessing query... (this may take a few seconds)")
            result = await agent.process_query(user_query)
            print("\nResult:")
            print(result["answer"])
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Close the database connection when done
    if agent.db_connection:
        agent.db_connection.close()
        print("Database connection closed.")

if __name__ == "__main__":
    asyncio.run(main())