import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

db_user = os.getenv("db_user")
db_password = os.getenv("db_password")
db_host=os.getenv("db_host")
#db_warehouse=os.getenv("db_warehouse")
db_database=os.getenv("db_database")
db_port=os.getenv("db_port")
db_schema= os.getenv("db_schema")
adv_db_database = os.getenv("adv_db_database") 
adv_db_schema = os.getenv("adv_db_schema")
adv_db_schema_hr = os.getenv("adv_db_schema_hr")
adv_db_schema_pe = os.getenv("adv_db_schema_pe")
adv_db_schema_purchase = os.getenv("adv_db_schema_purchase")
adv_db_schema_sales = os.getenv("adv_db_schema_sales")
#table_details_prompt = os.getenv('TABLE_DETAILS_PROMPT')
# Change if your schema is different
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
# LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
# LANGCHAIN_ENDPOINT=os.getenv("LANGCHAIN_ENDPOINT")


from langchain_community.utilities.sql_database import SQLDatabase
#from langchain.agents import create_sql_agent
#from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.memory import ChatMessageHistory
from operator import itemgetter

from urllib.parse import quote_plus


from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from table_details import table_chain as select_table
from prompts import final_prompt, answer_prompt, few_shot_prompt
from table_details import get_table_details , get_tables , itemgetter , create_extraction_chain_pydantic, Table 
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import configure

# @cache_resource
def get_chain(question, _messages,selected_model, selected_subject = 'Adv-HumanResources(HR)'):
    llm = ChatOpenAI(model=selected_model, temperature=0)
    #db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")  
    table_details = get_table_details(selected_subject)
    # table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
    # The permissible tables names are listed below and must be strictly followed:

    # {table_details}

    # Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""
    table_details_set_prompt = os.getenv('TABLE_DETAILS_SET_PROMPT')
    table_details_prompt=table_details_set_prompt.format(table=table_details)
    
    print("From utils.py Table_details_prompt: ", table_details_prompt)
    table_chain = {"input": itemgetter("question")} | create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt) | get_tables
    


    chosen_tables=table_chain.invoke({"question": question})
    # chosen_tables=select_table.invoke({"question": question})
    #Added by Rama
    print("tables chosen for query gen are: ... " , question , " ---" , chosen_tables)
    print("Creating DB Connection ... ")
    print("whats in the dropdown: " , selected_subject)
    print(selected_subject.startswith('Adv'))
    # print(adv_db_database, "-", adv_db_schema)
    
    #Changed by Rama
    if selected_subject.startswith('Adv'):
        if selected_subject.endswith('(HR)'):
    # if configure.selected_subject.startswith('Adv'):
            db = SQLDatabase.from_uri(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{adv_db_database}'
                                ,schema=adv_db_schema_hr
                                ,include_tables= chosen_tables
                                , view_support=True
                                ,sample_rows_in_table_info=1
                                ,lazy_table_reflection=True
                                )
            print("DB Connection Done for Adventureworks----",db._schema)
        elif selected_subject.endswith('Person'):
            db = SQLDatabase.from_uri(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{adv_db_database}'
                    ,schema=adv_db_schema_pe
                    ,include_tables= chosen_tables
                    , view_support=True
                    ,sample_rows_in_table_info=1
                    ,lazy_table_reflection=True
                    )
            print("DB Connection Done for Adventureworks----",db._schema)
        elif selected_subject.endswith('Purchasing'):
            db = SQLDatabase.from_uri(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{adv_db_database}'
                    ,schema=adv_db_schema_purchase
                    ,include_tables= chosen_tables
                    , view_support=True
                    ,sample_rows_in_table_info=1
                    ,lazy_table_reflection=True
                    )
            print("DB Connection Done for Adventureworks----",db._schema)
        elif selected_subject.endswith('Sales'):
            db = SQLDatabase.from_uri(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{adv_db_database}'
                    ,schema=adv_db_schema_sales
                    ,include_tables= chosen_tables
                    , view_support=True
                    ,sample_rows_in_table_info=1
                    ,lazy_table_reflection=True
                    )
            print("DB Connection Done for Adventureworks----",db._schema)

        else:
            db = SQLDatabase.from_uri(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{adv_db_database}'
                                ,schema=adv_db_schema
                                ,include_tables= chosen_tables
                                , view_support=True
                                ,sample_rows_in_table_info=1
                                ,lazy_table_reflection=True
                                )
            print("DB Connection Done for Adventureworks----",db._schema)
            
    else:
        db = SQLDatabase.from_uri(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}'
                             ,schema=db_schema
                             ,include_tables= chosen_tables
                             , view_support=True
                             ,sample_rows_in_table_info=1
                             ,lazy_table_reflection=True
                              )
        print("DB Connection Done for PostGress---",db._schema)
        #for testing of synapse
    
        
    llm = ChatOpenAI(model=selected_model, temperature=0)
    print("Generate Query Starting")
    generate_query = create_sql_query_chain(llm, db, final_prompt)
    SQL_Statement = generate_query.invoke({"question": question, "messages": _messages})
    print(f"Generated SQL Statement before execution: {SQL_Statement}") 

    print("and messages..", _messages, "This is generated query..", SQL_Statement)
    execute_query = QuerySQLDataBaseTool(db=db)
    rephrase_answer = answer_prompt | llm | StrOutputParser()
    #COMMENTED BY KRUSHNA 
    # chain = (
    #     RunnablePassthrough.assign(table_names_to_use=table_chain) |
    #     RunnablePassthrough.assign(query=generate_query).assign(
    #         result=itemgetter("query") | execute_query
    #     ) | rephrase_answer
    # )
    #ADDED BY KRUSHNA AS PER THE NEW REQUIREMENTS
    chain = (
        RunnablePassthrough.assign(table_names_to_use=table_chain) |
        RunnablePassthrough.assign(query=generate_query).assign(
            result=itemgetter("query") 
        ) 
    )

    return chain, chosen_tables, SQL_Statement, db

def create_history(messages):
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history

def invoke_chain(question, messages, selected_model, selected_subject):
    try:
            print('model used is: ', selected_model)
            history = create_history(messages)
            chain, chosen_tables, SQL_Statement, db = get_chain(question, history.messages, selected_model, selected_subject)
            print(f"Generated SQL Statement: {SQL_Statement}")
            SQL_Statement = SQL_Statement.replace("SQL Query:", "").strip()
            response = chain.invoke({"question": question, "top_k": 3, "messages": history.messages})
            print("Printing the question: ... ", question)
            print("Printing the response: ... ", response)
            print("Printing the chosen tables: ... ", chosen_tables)
            if configure.selected_subject.startswith('Adv'):
                alchemyEngine = create_engine(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{adv_db_database}')
            else:
                alchemyEngine = create_engine(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}')

                
        #     tables_data = {
        #     table: pd.read_sql(sql=response["query"] + ";", con=alchemyEngine.connect().connection)
        #     for table in chosen_tables
        # }
            tables_data = {}
            for table in chosen_tables:
                query = response["query"] + ";"
                # result = db.run(query)
                print(f"Executing SQL Query: {query}")
                with alchemyEngine.connect() as conn:
                    df = pd.read_sql(
                        sql=query,
                        con=conn.connection
                    )
                # tables_data[table] = pd.DataFrame()
                tables_data[table] = df
                break 
            
            
            return response, chosen_tables, tables_data, db
    except Exception as e:
        #error(f"An error occurred: {e}")
        print("error   ",e)
        custom_message = "Insufficient information to generate SQL Query."
        # # #generated_query = custom_message
        return custom_message, [], {}, None
        #return None, [], {}, None
#This change was done on 23/7/24 to keep track of the user's feedback 
def escape_single_quotes(input_string):
    return input_string.replace("'", "''")
def insert_feedback(department,user_query, sql_query, table_name, data, feedback_type="user not reacted", feedback="user not given feedback"):
    engine = create_engine(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}')
    Session = sessionmaker(bind=engine)
    session = Session()
    #sql_query = sql_query.replace("'", "''")
    department = escape_single_quotes(department)
    user_query = escape_single_quotes(user_query)
    sql_query = escape_single_quotes(sql_query)
    table_name = escape_single_quotes(table_name)
    data = escape_single_quotes(data)
    feedback_type = escape_single_quotes(feedback_type)
    feedback = escape_single_quotes(feedback)
    insert_query = f"""
    INSERT INTO lz_feedbacks (department, user_query, sql_query, table_name, data, feedback_type, feedback)
    VALUES ('{department}', '{user_query}', '{sql_query}', '{table_name}', '{data}', '{feedback_type}', '{feedback}')
    """
    try:
        session.execute(insert_query)
        session.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        session.rollback()
    finally:
        session.close()
def load_votes(table_name):
    votes = {"upvotes": 0, "downvotes": 0}
    engine = create_engine(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}')
    Session = sessionmaker(bind=engine)
    session = Session()
    execute_query = f"""
    SELECT upvotes, downvotes 
    FROM lz_votes 
    WHERE table_name = '{table_name}'
    """
    
    result = session.execute(execute_query).fetchone()
    votes = {"upvotes": 0, "downvotes": 0}
    if result:
        votes["upvotes"] = result[0]
        votes["downvotes"] = result[1]
    
    session.close()
    return votes

# Function to save votes for a table
def save_votes(table_name, votes):
    engine = create_engine(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}')
    Session = sessionmaker(bind=engine)
    session = Session()
    execute_query = f"""
INSERT INTO lz_votes (table_name, upvotes, downvotes) 
VALUES ('{table_name}', {votes["upvotes"]}, {votes["downvotes"]})
ON CONFLICT (table_name) 
DO UPDATE SET 
    upvotes = EXCLUDED.upvotes,
    downvotes = EXCLUDED.downvotes
"""
    
    session.execute(execute_query)
    session.commit()
    session.close()
