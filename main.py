from fastapi import FastAPI, Form, HTTPException,Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.responses import JSONResponse
import os
import pandas as pd
from langchain.chains.openai_tools import create_extraction_chain_pydantic 
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from newlangchain_utils import *
from dotenv import load_dotenv
from state import session_state, session_lock
load_dotenv()  # Load environment variables from .env file
from typing import Optional
app = FastAPI()

# Set up static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize OpenAI API key and model
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
models = os.getenv('models').split(',')
databases = os.getenv('databases').split(',')
subject_areas1 = os.getenv('subject_areas1').split(',')
subject_areas2 = os.getenv('subject_areas2').split(',')
question_dropdown = os.getenv('Question_dropdown')
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)  # Adjust model as necessary
from table_details import get_table_details  # Importing the function

class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):

    # Extract table names dynamically
    tables = []

    # Pass dynamically populated dropdown options to the template
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": models,
        "databases": databases,  # Dynamically populated database dropdown
        "section": subject_areas2,
        "tables": tables,        # Table dropdown based on database selection
        "question_dropdown": question_dropdown.split(','),  # Static questions from env
    })
    
@app.get("/get_questions/")
async def get_questions(subject: str):
    """Fetch questions from the selected subject's CSV file."""
    csv_file = f"{subject}_questions.csv"
    if not os.path.exists(csv_file):
        return JSONResponse(
            content={"error": f"The file `{csv_file}` does not exist."}, status_code=404
        )
    
    try:
        # Read the questions from the CSV
        questions_df = pd.read_csv(csv_file)
        if "question" in questions_df.columns:
            questions = questions_df["question"].tolist()
        else:
            questions = questions_df.iloc[:, 0].tolist()
        return {"questions": questions}
    except Exception as e:
        return JSONResponse(
            content={"error": f"An error occurred while reading the file: {str(e)}"}, status_code=500
        )
@app.get("/get-tables/")
async def get_tables(selected_section: str):
    # Fetch table details for the selected section
    table_details = get_table_details(selected_section)
    # Extract table names dynamically
    tables = [line.split("Table Name:")[1].strip() for line in table_details.split("\n") if "Table Name:" in line]
    # Return tables as JSON
    return {"tables": tables}

@app.post("/submit")
async def submit_query(
    request: Request,
    section: str = Form(...),
    role: str = Form(...),
    example_question: str = Form(...),
    user_query: str = Form(...),
    page: int = Query(1),  # Default to page 1
    records_per_page: int = Query(5),  # Default to 10 records per page
    model : Optional[str] = Form("gpt-4o-mini")
):
    selected_subject = section
    session_state['user_query'] = user_query
    if role == "user":
        prompt = user_query if user_query else example_question

        # Initialize session state if not already done
        if 'messages' not in session_state:
            session_state['messages'] = []

        session_state['messages'].append({"role": "user", "content": prompt})

        try:
            response, chosen_tables, tables_data, agent_executor = invoke_chain(
                prompt, session_state['messages'],"gpt-4o-mini",selected_subject
            )

            if isinstance(response, str):
                session_state['generated_query'] = response
            else:
                session_state['chosen_tables'] = chosen_tables
                session_state['tables_data'] = tables_data
                session_state['generated_query'] = response.get("query", "")

            # Prepare table data for rendering
            tables_html = []
            for table_name, data in tables_data.items():
                # Pagination details
                total_records = len(data)
                total_pages = (total_records + records_per_page - 1) // records_per_page

                # Render styled table for the current page
                html_table = display_table_with_styles(data, table_name, page, records_per_page)

                tables_html.append({
                    "table_name": table_name,
                    "table_html": html_table,
                    "pagination": {
                        "current_page": page,
                        "total_pages": total_pages,
                        "records_per_page": records_per_page,
                    }
                })

            response_data = {
                "user_query": session_state['user_query'],
                "query": session_state['generated_query'],
                "tables": tables_html,
            }
            return JSONResponse(content=response_data)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing the prompt: {str(e)}")
    else:
        return {"message": "Admin functionality not implemented yet."}

# Table data display endpoint
def display_table_with_styles(data, table_name, page_number, records_per_page):
    start_index = (page_number - 1) * records_per_page
    end_index = start_index + records_per_page
    page_data = data.iloc[start_index:end_index]
    styled_table = page_data.style.set_table_attributes('style="border: 2px solid black; border-collapse: collapse;"') \
        .set_table_styles(
            [{
                'selector': 'th',
                'props': [('background-color', '#333'), ('color', 'white'), ('font-weight', 'bold'), ('font-size', '16px')]
            },
            {
                'selector': 'td',
                'props': [('border', '2px solid black'), ('padding', '5px')]
            }]
        ).to_html(escape=False)
    return styled_table

@app.get("/get_table_data/")
async def get_table_data(
    table_name: str = Query(...),
    page_number: int = Query(1),
    records_per_page: int = Query(10),
):
    """Fetch paginated and styled table data."""
    try:
        # Check if the requested table exists in session state
        if "tables_data" not in session_state or table_name not in session_state["tables_data"]:
            raise HTTPException(status_code=404, detail=f"Table {table_name} data not found.")

        # Retrieve the data for the specified table
        data = session_state["tables_data"][table_name]
        total_records = len(data)
        total_pages = (total_records + records_per_page - 1) // records_per_page

        # Ensure valid page number
        if page_number < 1 or page_number > total_pages:
            raise HTTPException(status_code=400, detail="Invalid page number.")

        # Slice data for the requested page
        start_index = (page_number - 1) * records_per_page
        end_index = start_index + records_per_page
        page_data = data.iloc[start_index:end_index]

        # Style the table as HTML
        styled_table = (
            page_data.style.set_table_attributes('style="border: 2px solid black; border-collapse: collapse;"')
            .set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#333'), ('color', 'white'), ('font-weight', 'bold'), ('font-size', '16px')]},
                {'selector': 'td', 'props': [('border', '2px solid black'), ('padding', '5px')]},
            ])
            .to_html(escape=False)  # Render as HTML
        )

        return {
            "table_html": styled_table,
            "page_number": page_number,
            "total_pages": total_pages,
            "total_records": total_records,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating table data: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
