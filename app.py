
import streamlit as st
import google.generativeai as genai
import mysql.connector
import pandas as pd
import io  # To use BytesIO for Excel export
from mysql.connector import Error
import networkx as nx
import plotly.express as px
import speech_recognition as sr
import  tempfile
import os
from gtts import gTTS
import requests

# Configure the API Key
GOOGLE_API_KEY = "AIzaSyDX8os0xVmRfWDXfKR-rnaoks3_GfwtwFg"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')


YOUTUBE_API_KEY = "AIzaSyD2k2-kMObtxWrX_RQVQhFhUhXzS3ZbVk8"  

# Function to get topic from Gemini API
def get_sql_topic_from_gemini(query):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Replace with the desired model
        response = model.generate_content(query)
        topic = response.text.strip()  # Get the SQL topic from Gemini's response
        return topic
    except Exception as e:
        st.error(f"Error getting topic from Gemini API: {e}")
        return None

    
# Function to search YouTube videos based on the query
def search_youtube_videos(api_key, search_query):
    youtube_search_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": f"SQL {search_query} tutorial",  # More descriptive search query
        "type": "video",
        "key": api_key,
        "maxResults": 7,
    }

    try:
        response = requests.get(youtube_search_url, params=params)
        response.raise_for_status()
        response_data = response.json()

        # Extract video details
        video_results = []
        for video in response_data.get("items", []):
            video_id = video["id"]["videoId"]
            title = video["snippet"]["title"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            video_results.append({"title": title, "url": video_url})
        
        return video_results
    except requests.exceptions.RequestException as e:
        st.error(f"Error with YouTube API: {e}")
        return []
    except KeyError as e:
        st.error(f"Unexpected response format: {e}")
        return []


# Function to determine the modified table (can be enhanced based on your use case)
def get_modified_table_from_query(query):
    # Identify the modified table based on query keywords (e.g., INSERT INTO, UPDATE, DELETE FROM)
    query_lower = query.strip().lower()
    if query_lower.startswith("insert into"):
        return query.split()[2]  # Assuming table name follows 'INSERT INTO'
    elif query_lower.startswith("update"):
        return query.split()[1]  # Assuming table name follows 'UPDATE'
    elif query_lower.startswith("delete from"):
        return query.split()[2]  # Assuming table name follows 'DELETE FROM'
    elif query_lower.startswith(("create", "alter", "drop")):
        # For DDL queries, the table is usually specified after the command
        return query.split()[2] if len(query.split()) > 2 else None
    return None


# Function to execute SQL query
def execute_query(query, db_params):
    try:
        if not all(db_params.values()) and not query.strip().lower().startswith("create database"):
            return {"status": "error", "message": "Database connection details are missing."}

        # Special handling for CREATE, DROP, or DELETE DATABASE queries
        if query.strip().lower().startswith("create database"):
            conn = mysql.connector.connect(
                host=db_params.get("host", "localhost"),
                user=db_params.get("user", "root"),
                password=db_params.get("password", ""),
            )
            cursor = conn.cursor()
            cursor.execute(query)
            conn.close()
            return {"status": "success", "message": "Database created successfully."}

        elif query.strip().lower().startswith("drop database"):
            conn = mysql.connector.connect(
                host=db_params.get("host", "localhost"),
                user=db_params.get("user", "root"),
                password=db_params.get("password", ""),
            )
            cursor = conn.cursor()
            cursor.execute(query)
            conn.close()
            return {"status": "success", "message": "Database dropped successfully."}

        elif query.strip().lower().startswith("delete database"):
            conn = mysql.connector.connect(
                host=db_params.get("host", "localhost"),
                user=db_params.get("user", "root"),
                password=db_params.get("password", ""),
            )
            cursor = conn.cursor()
            cursor.execute(query)
            conn.close()
            return {"status": "success", "message": "Database deleted successfully."}

        # Handle other queries
        conn = mysql.connector.connect(**db_params)
        cursor = conn.cursor()

        # Execute the query
        cursor.execute(query)

        # Check for DDL queries like CREATE, ALTER, DROP
        if query.strip().lower().startswith(("create", "alter", "drop")):
            conn.commit()
            # After executing DDL queries, fetch all tables and their status
            tables = fetch_tables(db_params)
            conn.close()
            return {"status": "success", "message": "DDL query executed.", "data": tables}

        elif query.strip().lower().startswith("select"):
            columns = [col[0] for col in cursor.description]
            data = cursor.fetchall()
            conn.close()
            return {"status": "success", "data": pd.DataFrame(data, columns=columns)}

        else:
            conn.commit()
            # After executing DML queries (INSERT, UPDATE, DELETE), fetch the modified table data
            modified_table = get_modified_table_from_query(query)
            if modified_table:
                table_data = fetch_table_data(db_params, modified_table)
                conn.close()
                return {"status": "success", "data": table_data}
            conn.close()
            return {"status": "success", "data": pd.DataFrame()}

    except Error as e:
        return {"status": "error", "message": f"Error executing query: {str(e)}"}


# Function to process SQL file content
def process_sql_file(file_content):
    queries = file_content.strip().split(";")
    return [query.strip() for query in queries if query.strip()]


def transcribe_audio_to_text():
    """
    Captures audio from the microphone and transcribes it into text.
    Returns the transcribed text or an error message.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak into the microphone.")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.info("Processing the audio...")
            text = recognizer.recognize_google(audio)  # Using Google Web Speech API
            return text
        except sr.WaitTimeoutError:
            return "Error: Timeout. Please speak louder or check your microphone."
        except sr.UnknownValueError:
            return "Error: Could not understand the audio."
        except sr.RequestError as e:
            return f"Error: Could not request results from the speech recognition service; {e}"

# Function to generate explanation for SQL query
def generate_explanation(sql_query):
    explanation_template = """
        Explain the SQL Query snippet:
        {sql_query}
        Please provide the simplest explanation:
    """
    explanation_formatted = explanation_template.format(sql_query=sql_query)
    explanation_response = model.generate_content(explanation_formatted)
    explanation = explanation_response.text.strip()

    return explanation

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        tts.save(tmp_file.name)
        tmp_file.seek(0)  # Move pointer to the start of the file
        return tmp_file.name

# Function to generate expected output for SQL query
def generate_expected_output(sql_query):
    expected_output_template = """
        What would be the expected output of the SQL Query snippet:
        {sql_query}
        Provide a sample tabular response with no explanation:
    """
    expected_output_formatted = expected_output_template.format(sql_query=sql_query)
    expected_output_response = model.generate_content(expected_output_formatted)
    expected_output = expected_output_response.text.strip()

    return expected_output

# Function to fetch all tables in the connected database
def fetch_tables(db_params):
    try:
        conn = mysql.connector.connect(**db_params)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        conn.close()
        return pd.DataFrame(tables, columns=["Table Name"])  # Return as a DataFrame for display
    except Error as e:
        return {"status": "error", "message": f"Error fetching tables: {str(e)}"}

# Function to fetch data from a specific table
def fetch_table_data(db_params, table_name):
    try:
        conn = mysql.connector.connect(**db_params)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
        columns = [col[0] for col in cursor.description]
        data = cursor.fetchall()
        conn.close()
        return pd.DataFrame(data, columns=columns)
    except Error as e:
        return {"status": "error", "message": f"Error fetching data from table {table_name}: {str(e)}"}

def generate_pdf_report(report_data):
    import io
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add title
    pdf.cell(200, 10, txt="SQL Query Report", ln=True, align="C")
    pdf.ln(10)

    # Iterate through the data and add it to the PDF
    for _, row in report_data.iterrows():
        pdf.set_font("Arial", style="B", size=12)
        pdf.multi_cell(0, 10, f"Prompt:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, row["prompt"])

        pdf.set_font("Arial", style="B", size=12)
        pdf.multi_cell(0, 10, f"Query:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, row["query"])

        pdf.set_font("Arial", style="B", size=12)
        pdf.multi_cell(0, 10, f"Explanation:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, row["explanation"])

        pdf.set_font("Arial", style="B", size=12)
        pdf.multi_cell(0, 10, f"Expected Output:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, row["output"])
        

        # Include query results if available
        if "results" in row and row["results"]:
            pdf.set_font("Arial", style="B", size=12)
            pdf.multi_cell(0, 10, f"Query Results:")
            pdf.set_font("Arial", size=12)

            # Convert query results to a readable string format
            results = row["results"]
            if isinstance(results, list):
                results_text = "\n".join(str(res) for res in results)
            elif isinstance(results, str):
                results_text = results
            else:
                results_text = str(results)

            pdf.multi_cell(0, 10, results_text)
        pdf.ln(5)

    # Save the PDF content to a BytesIO object
    pdf_output = io.BytesIO()
    pdf_content = pdf.output(dest="S").encode("latin1")  # Get PDF content as a string
    pdf_output.write(pdf_content)  # Write the string content to BytesIO
    pdf_output.seek(0)  # Reset the buffer's pointer to the start
    return pdf_output



def main():
    
    st.set_page_config(page_title="SQL Query Generator", page_icon="🔍", layout="wide")
  # App Header
    st.markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <h1>SQL Query Generator 🤖</h1>
            <h3>Generate SQL queries effortlessly ✨</h3>
            <h4>Get explanations, expected outputs, and optional execution 📚</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for Database Connection
    st.sidebar.header("Database Connection (Optional)")
    db_params = {
        "host": st.sidebar.text_input("Host", "localhost"),
        "user": st.sidebar.text_input("Username", "root"),
        "password": st.sidebar.text_input("Password", type="password"),
        "database": st.sidebar.text_input("Database Name", "mydb"),
    }

    # Sidebar - Test Connection
    if st.sidebar.button("Test Connection"):
        try:
            conn = mysql.connector.connect(**db_params)
            conn.close()
            st.sidebar.success("Database connection successful!")
        except Error as e:
            st.sidebar.error(f"Connection failed: {e}")

    # Show tables in the sidebar if database connection details are provided
    if all(db_params.values()):
        tables = fetch_tables(db_params)
        if isinstance(tables, pd.DataFrame):
            st.sidebar.subheader("Tables in Database")
            selected_table = st.sidebar.selectbox("Select a table to view:", tables["Table Name"].tolist())
            if selected_table:
                table_data = fetch_table_data(db_params, selected_table)
                if isinstance(table_data, pd.DataFrame):
                    st.sidebar.write(f"Contents of `{selected_table}`:")
                    st.sidebar.dataframe(table_data)
                else:
                    st.sidebar.error(table_data["message"])
        else:
            st.sidebar.error(tables["message"])

    # Sidebar for Uploading Files
    st.sidebar.header("Upload SQL Files")
    uploaded_files = st.sidebar.file_uploader("Upload one or more .sql files", type=["sql"], accept_multiple_files=True)

    generated_data = []  # To store generated prompts, queries, and outputs

    tabs = st.selectbox(
        "Choose a feature",
        [
            "Generate Query from English",
            "Upload SQL Files",
            "Visualize Data",
            "Database Schema",
            "Learn SQL"
            
        ]
    )

    # Handling Tabs
    if tabs == "Upload SQL Files":
        # Handling Uploaded SQL Files
        if uploaded_files:
            st.info("Processing uploaded SQL files...")
            for uploaded_file in uploaded_files:
                st.markdown(f"## File: {uploaded_file.name}")
                file_content = uploaded_file.read().decode("utf-8")
                queries = process_sql_file(file_content)

                for idx, query in enumerate(queries, start=1):
                    st.markdown(f"### Query {idx}")
                    st.code(query, language="sql")

                    explanation = generate_explanation(query)

                    # Generate expected output even when a database connection is provided
                    expected_output = generate_expected_output(query)

                    st.success("Explanation:")
                    st.markdown(explanation)

                    st.success("Expected Output:")
                    st.markdown(expected_output)

                    # Add to generated_data
                    generated_data.append({
                        "prompt": f"Query {idx} from file {uploaded_file.name}",
                        "query": query,
                        "explanation": explanation,
                        "output": expected_output
                    })

        else:
            st.warning("Please upload a SQL file to get started.")

    elif tabs == "Database Schema":
        schema = fetch_tables(db_params)
        if isinstance(schema, pd.DataFrame):
            G = nx.DiGraph()
            for table in schema["Table Name"]:
                table_data = fetch_table_data(db_params, table)
                G.add_node(table)
                for col in table_data.columns:
                    G.add_edge(table, col)
            st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())


    elif tabs == "Visualize Data":
     query = st.text_area("Enter a SELECT query to visualize data:")
     chart_type = st.selectbox("Select chart type", ["Bar", "Line", "Pie", "Scatter", "Histogram", "Box", "Area"])

    # Initialize session state for columns
     if "all_columns" not in st.session_state:
        st.session_state.all_columns = []
     if "numeric_columns" not in st.session_state:
        st.session_state.numeric_columns = []

    # Step 1: Fetch columns using a separate button
     if st.button("Fetch Columns"):
        result = execute_query(query, db_params)
        if result["status"] == "success" and "data" in result:
            data = result["data"]
            if data.empty:
                st.warning("Query returned no data.")
            else:
                st.dataframe(data)  # Show the data to the user
                # Update session state with column information
                st.session_state.all_columns = data.columns.tolist()
                st.session_state.numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
                st.success("Columns fetched successfully. Select X and Y axes below.")
        else:
            error_message = result.get("message", "Unknown error occurred.")
            st.error(error_message)

    # Step 2: Allow user to select X and Y axes
     x_axis = st.selectbox("Select X-axis column", st.session_state.all_columns)
     y_axis = st.selectbox("Select Y-axis column (optional)", st.session_state.numeric_columns)

    # Step 3: Visualize the data
     if st.button("Visualize Query"):
        result = execute_query(query, db_params)
        if result["status"] == "success" and "data" in result:
            data = result["data"]
            if data.empty:
                st.warning("Query returned no data to visualize.")
            else:
                st.dataframe(data)  # Display the data again

                # Generate the appropriate chart
                if chart_type == "Bar":
                    fig = px.bar(data, x=x_axis, y=y_axis, title="Bar Chart")
                elif chart_type == "Line":
                    fig = px.line(data, x=x_axis, y=y_axis, title="Line Chart")
                elif chart_type == "Pie":
                    fig = px.pie(data, names=x_axis, values=y_axis, title="Pie Chart")
                elif chart_type == "Scatter":
                    fig = px.scatter(data, x=x_axis, y=y_axis, title="Scatter Plot")
                elif chart_type == "Histogram":
                    fig = px.histogram(data, x=x_axis, title="Histogram")
                elif chart_type == "Box":
                    fig = px.box(data, x=x_axis, y=y_axis, title="Box Plot")
                elif chart_type == "Area":
                    fig = px.area(data, x=x_axis, y=y_axis, title="Area Chart")
                else:
                    st.error("Unsupported chart type selected.")

                st.plotly_chart(fig)
        else:
            error_message = result.get("message", "Unknown error occurred.")
            st.error(error_message)

    elif tabs == "Learn SQL":
      st.header("Learn SQL with AI and Videos 🎥")

   # Input field for SQL-related question in plain English
      query = st.text_input("Ask a question about SQL (e.g., 'Explain SQL JOIN')")
       # Add a "Generate Videos" button to control when the search occurs
      if st.button("Generate") and query:
        # Step 1: Get SQL topic from Gemini
         sql_topic = get_sql_topic_from_gemini(query)
         if sql_topic:
            st.write(f"{sql_topic}")

            # Step 2: Search YouTube videos based on the topic
            videos = search_youtube_videos(YOUTUBE_API_KEY, sql_topic)
            if videos:
                st.write("Here are some tutorial videos for you:")
                for video in videos:
                    st.markdown(f"[{video['title']}]({video['url']})")  # Display YouTube video links
            else:
                st.write("No videos found. Try another query.")
         else:
            st.write("Unable to retrieve SQL topic. Please try again.")


    elif tabs == "Generate Query from English":
    # Handling Plain English Query Input
      st.header("Generate SQL Query from Plain English")
    
    # Initialize session state for voice input
      if "voice_input" not in st.session_state:
        st.session_state["voice_input"] = ""

    # Text input area
      text_input = st.text_area(
        "Enter your SQL query in plain English:",
        placeholder="E.g., Show all employees in the IT department",
        value=st.session_state.get("voice_input", "")
    )

    # Voice input button
      if st.button("Record Audio 🎙️"):
        voice_text = transcribe_audio_to_text()  # Call the function to transcribe audio
        if "Error" not in voice_text:
            st.session_state["voice_input"] = voice_text  # Save the transcribed text
            st.success(f"Transcribed Text: {voice_text}")
            text_input = voice_text  # Use voice input as the query text
        else:
            st.error(voice_text)

    # Generate SQL Query Button
      if st.button("Generate SQL Query"):
        if text_input.strip() == "":
            st.warning("Please enter a valid plain English query.")
            return

        with st.spinner("Generating SQL Query..."):
            try:
                # Generate SQL Query
                template = """
                    Create a SQL Query snippet using the below text:
                    {text_input}
                    I just want a SQL Query.
                """
                formatted_template = template.format(text_input=text_input)
                response = model.generate_content(formatted_template)
                sql_query = response.text.strip().lstrip("```sql").rstrip("```")

                # Generate Explanation and Expected Output
                explanation = generate_explanation(sql_query)
                expected_output = generate_expected_output(sql_query)

                # Display Results
                st.success("SQL Query Generated Successfully! Here is your query:")
                st.code(sql_query, language="sql")

                st.success("Explanation of the SQL Query:")
                st.markdown(explanation)


                # Play the explanation using gTTS
                if explanation:
                    st.audio(text_to_speech(explanation), format="audio/mp3")


                if expected_output:
                    st.success("Expected Output of the SQL Query:")
                    st.markdown(expected_output)

                # Execute Query if Database Connection Details Provided
                if all(db_params.values()):
                    st.info("Executing query...")
                    result = execute_query(sql_query, db_params)
                    if result["status"] == "success":
                        if "data" in result and not result["data"].empty:
                            st.success("Query executed successfully! Displaying results:")
                            st.dataframe(result["data"])
                        else:
                            st.success(result["message"])
                    else:
                        st.error(f"Error executing query: {result['message']}")

                # Enable Download of Generated Query
                st.download_button(
                    label="Download SQL Query",
                    data=sql_query,
                    file_name="generated_query.sql",
                    mime="text/sql"
                )

                # Add to generated_data
                generated_data.append({
                    "prompt": "Plain English Query",
                    "query": sql_query,
                    "explanation": explanation,
                    "output": expected_output
                })

            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Selectbox for choosing download format
    download_format = st.selectbox(
        "Choose download format for the query report:",
        ("JSON", "CSV", "Text", "PDF")
    )

    if generated_data:
        report_data = pd.DataFrame(generated_data)

        if download_format == "CSV":
            st.download_button(
                label="Download CSV",
                data=report_data.to_csv(index=False),
                file_name="sql_query_report.csv",
                mime="text/csv"
            )
        elif download_format == "JSON":
            st.download_button(
                label="Download JSON",
                data=report_data.to_json(orient="records"),
                file_name="sql_query_report.json",
                mime="application/json"
            )
        elif download_format == "Text":
            text_content = ""
            for _, row in report_data.iterrows():
                text_content += f"Prompt:\n{row['prompt']}\n\n"
                text_content += f"Query:\n{row['query']}\n\n"
                text_content += f"Explanation:\n{row['explanation']}\n\n"
                text_content += f"Expected Output:\n{row['output']}\n\n"
                text_content += "-" * 50 + "\n"

            st.download_button(
                label="Download Text",
                data=text_content,
                file_name="sql_query_report.txt",
                mime="text/plain"
            )
        elif download_format == "PDF":
            pdf_content = generate_pdf_report(report_data)
            st.download_button(
                label="Download PDF",
                data=pdf_content,
                file_name="sql_query_report.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("No queries were generated yet. Please generate some queries first.")

if __name__ == "__main__":
    main()

