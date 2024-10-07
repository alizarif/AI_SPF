import os
import pandas as pd
import time
import signal
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_current_date():
    today = datetime.now()
    return today.strftime("%b %d %Y")

def get_current_quarter():
    today = datetime.now()
    return f"{today.year} Q{(today.month - 1) // 3 + 1}"

def get_future_quarters(num_quarters):
    today = datetime.now()
    quarters = []
    for i in range(num_quarters):
        future_date = today + timedelta(days=91*i)
        quarters.append(f"{future_date.year} Q{(future_date.month - 1) // 3 + 1}")
    return quarters

def create_forecast_queries(variable_data, num_future_quarters=4):
    column_explanations = {
        'cpi': "Consumer Price Index",
        'gdp': "Gross Domestic Product",
        # ... (other variables)
    }

    current_date = get_current_date()
    current_quarter = get_current_quarter()
    future_quarters = get_future_quarters(num_future_quarters)

    query = (f"We are currently on {current_date}, which falls in {current_quarter}. "
             f"Please provide your best numeric forecasts for the following variables: {', '.join(variable_data['definition'])}. "
             f"Do this for the current quarter and the following future quarters: {', '.join(future_quarters)}. "
             f"Also provide annual forecasts for this year and the next {num_future_quarters} years. "
             f"Use available information up to today, your professional judgement, and experience. "
             f"Consider current economic conditions and Federal Reserve actions. "
             f"Your forecast is anonymous. Provide the forecasts as a sequence of numerical values only. "
             f"Please provide your forecasts in the format: (current quarter, {', '.join([f'Q+{i+1}' for i in range(num_future_quarters)])}, this year's average, {', '.join([f'year+{i+1} average' for i in range(num_future_quarters)])}). ")

    return query, current_date, current_quarter

def signal_handler(signum, frame):
    global should_exit
    logger.info("Received interrupt signal. Finishing current task before exiting...")
    should_exit = True

def save_progress(assistant_name, batch_count, last_processed_idx):
    progress = {
        'assistant_name': assistant_name,
        'batch_count': batch_count,
        'last_processed_idx': last_processed_idx
    }
    with open('progress.json', 'w') as f:
        json.dump(progress, f)
    logger.info(f"Progress saved: {progress}")

def load_progress():
    if os.path.exists('progress.json'):
        with open('progress.json', 'r') as f:
            progress = json.load(f)
        logger.info(f"Progress loaded: {progress}")
        return progress
    return None

def format_results(results):
    formatted_results = []
    for row in results:
        name = row[0]
        current_date = row[1]
        current_quarter = row[2]
        labeled_forecasts = row[10]
        
        # Parse the labeled forecasts
        for forecast in labeled_forecasts.split(';'):
            variable, values = forecast.split(':')
            values = values.strip()[1:-1].split(',')  # Remove parentheses and split
            
            formatted_row = [name, current_date, current_quarter, variable.strip()]
            formatted_row.extend([float(v.strip()) for v in values])
            formatted_results.append(formatted_row)
    
    return formatted_results

def main():
    global should_exit
    should_exit = False

    # Clear any cached environment variables
    load_dotenv('api.env')
    api_key = os.getenv('API_KEY1')
    client = OpenAI(api_key=api_key)

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Define output folder
    output_folder = 'future_forecasts'
    os.makedirs(output_folder, exist_ok=True)

    # Load variable data
    variable_data = pd.read_csv('inflation_def.csv', encoding='latin1')

    # Load forecaster data
    forecaster_data = pd.read_csv('forecasters.csv')

    # Create forecast query
    forecast_query, current_date, current_quarter = create_forecast_queries(variable_data)

    # Define assistants
    assistants = {
        "Assistant_SPF5_Run1": "asst_BfXuSsa5HJSPokfNcW0ihdEk",
    }

    # Process results in batches and save to a new file each time
    batch_size = 10
    
    all_results = []
    for assistant_name, assistant_id in assistants.items():
        results = []
        batch_count = 0
        
        # Resume from last processed index if applicable
        progress = load_progress()
        last_processed_idx = -1
        if progress and progress['assistant_name'] == assistant_name:
            last_processed_idx = progress['last_processed_idx']
            batch_count = progress['batch_count']
        
        for idx, row in forecaster_data.iterrows():
            if idx <= last_processed_idx:
                continue
            
            if should_exit:
                logger.info("Exiting gracefully...")
                save_progress(assistant_name, batch_count, idx - 1)
                return

            name = row['name']
            result_row = [name, current_date, current_quarter]
            
            try:
                my_thread = client.beta.threads.create()
                
                # Initial instruction
                personal_info = f"You are a participant on a panel of Survey of Professional Forecasters. {name} "
                company_location = f"Your organization is based in {row['company_location']}. " if pd.notna(row['company_location']) else ""
                country_origin_text = f"You are originally from {row['country_origin']}. " if pd.notna(row['country_origin']) else ""
                social_media = f"{row['social_media_status']} " if pd.notna(row['social_media_status']) else ""
                
                initial_instruction = personal_info + company_location + country_origin_text + social_media + row['persona']
                initial_prompt = "Do you confirm and understand your role?"
                
                my_thread_message = client.beta.threads.messages.create(
                    thread_id=my_thread.id,
                    role="user",
                    content=f"{initial_instruction}\n\n{initial_prompt}"
                )
                
                my_run = client.beta.threads.runs.create(
                    thread_id=my_thread.id,
                    assistant_id=assistant_id,
                    instructions="Please confirm if you understand your role based on the given information. Please just type 'yes' or 'no'."
                )
                
                start_time = time.time()
                while True:
                    keep_retrieving_run = client.beta.threads.runs.retrieve(
                        thread_id=my_thread.id,
                        run_id=my_run.id
                    )
                    if keep_retrieving_run.status == "completed":
                        all_messages = client.beta.threads.messages.list(
                            thread_id=my_thread.id
                        )
                        response = all_messages.data[0].content[0].text.value
                        result_row.extend([initial_instruction, initial_prompt, "Please confirm if you understand your role based on the given information. Please just type 'yes' or 'no'.", response])
                        break
                    if time.time() - start_time > 60:  # Timeout after 60 seconds
                        logger.warning(f"Timeout waiting for initial confirmation for {name}")
                        result_row.extend([initial_instruction, initial_prompt, "Please confirm if you understand your role based on the given information. Please just type 'yes' or 'no'.", "Timeout"])
                        break
                    time.sleep(1)
                
                # Forecast query
                my_thread_message = client.beta.threads.messages.create(
                    thread_id=my_thread.id,
                    role="user",
                    content=forecast_query
                )
                
                my_run = client.beta.threads.runs.create(
                    thread_id=my_thread.id,
                    assistant_id=assistant_id,
                    instructions="Take a deep breath and provide your best numeric forecasts based on the given information and instructions. Please only provide your forecasts in the requested format. Do not use any alphabet under any circumstance (with the exception of the variable name) and only provide the numbers in the requested format."
                )
                
                start_time = time.time()
                while True:
                    keep_retrieving_run = client.beta.threads.runs.retrieve(
                        thread_id=my_thread.id,
                        run_id=my_run.id
                    )
                    if keep_retrieving_run.status == "completed":
                        all_messages = client.beta.threads.messages.list(
                            thread_id=my_thread.id
                        )
                        follow_up_response = all_messages.data[0].content[0].text.value
                        
                        # Split the response into forecasts, considering both \n and commas as delimiters
                        forecast_lines = [line.strip() for line in follow_up_response.replace('\n', ',').split(',') if line.strip()]
                        
                        # Ensure the number of forecasts matches the number of variables
                        labeled_forecasts = []
                        if len(forecast_lines) == len(variable_data['var']):
                            for forecast, (var, definition) in zip(forecast_lines, variable_data[['var', 'definition']].itertuples(index=False)):
                                try:
                                    forecast_value = float(forecast.strip())
                                    labeled_forecasts.append(f"{definition}: ({forecast.strip()})")
                                except ValueError:
                                    labeled_forecasts.append(f"{definition}")
                        else:
                            logger.warning(f"Warning: Number of forecast lines ({len(forecast_lines)}) does not match number of variables ({len(variable_data['var'])}).")
                            labeled_forecasts = [f"{definition}" for definition in variable_data['definition']]
                        
                        result_row.append("; ".join(labeled_forecasts))
                        result_row.extend([forecast_query, "Take a deep breath, and provide your best numeric forecasts based on the given information and instructions. Please only provide your forecasts in the requested format. Do not use any alphabet and only provide the numbers in the requested format.", follow_up_response])
                        break
                    if time.time() - start_time > 60:  # Timeout after 60 seconds
                        logger.warning(f"Timeout waiting for follow-up response for {name}")
                        result_row.append("; ".join([f"{definition}" for definition in variable_data['definition']]))
                        result_row.extend([forecast_query, "Take a deep breath, and provide your best numeric forecasts based on the given information and instructions. Please only provide your forecasts in the requested format. Do not use any alphabet and only provide the numbers in the requested format.", "Timeout"])
                        break
                    time.sleep(1)
                
                # Reasoning query
                my_thread_message = client.beta.threads.messages.create(
                    thread_id=my_thread.id,
                    role="user",
                    content="Please provide the reasoning behind your forecast. Please limit your response to 1-2 sentences only."
                )
                
                my_run = client.beta.threads.runs.create(
                    thread_id=my_thread.id,
                    assistant_id=assistant_id,
                    instructions="Please provide a brief explanation for your forecast in 1-2 sentences."
                )
                
                start_time = time.time()
                while True:
                    keep_retrieving_run = client.beta.threads.runs.retrieve(
                        thread_id=my_thread.id,
                        run_id=my_run.id
                    )
                    if keep_retrieving_run.status == "completed":
                        all_messages = client.beta.threads.messages.list(
                            thread_id=my_thread.id
                        )
                        reasoning_response = all_messages.data[0].content[0].text.value
                        result_row.extend(["Please provide the reasoning behind your forecast. Please have your response in 1-2 sentences.", "Please provide a brief explanation for your forecast in 1-2 sentences.", reasoning_response])
                        break
                    if time.time() - start_time > 60:  # Timeout after 60 seconds
                        logger.warning(f"Timeout waiting for reasoning response for {name}")
                        result_row.extend(["Please provide the reasoning behind your forecast. Please have your response in 1-2 sentences.", "Please provide a brief explanation for your forecast in 1-2 sentences.", "Timeout"])
                        break
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing forecast for {name}: {e}")
                result_row.extend(["Error"] * (14 - len(result_row)))  # Ensure consistent number of columns
            
            results.append(result_row)
            logger.info(f"Assistant: {assistant_name}, Name: {name}: Processed")

            if (idx + 1) % batch_size == 0 or idx == len(forecaster_data) - 1:
                # Save the current batch to a new Excel file
                batch_count += 1
                columns = ['Name', 'Current_Date', 'Current_Quarter', 'Initial Instruction', 'Initial Prompt', 'Initial Instruction Response', 'Initial Response', 'Forecast Prompt', 'Forecast Instruction', 'Forecast Response', 'Labeled Forecasts', 'Reasoning Prompt', 'Reasoning Instruction', 'Reasoning Response']
                df = pd.DataFrame(results, columns=columns)
                output_filename = os.path.join(output_folder, f'future_forecasts_{assistant_name}_batch_{batch_count}.xlsx')
                df.to_excel(output_filename, index=False)
                logger.info(f"Results for {assistant_name} batch {batch_count} saved to {output_filename}")
                
                # Save progress
                save_progress(assistant_name, batch_count, idx)
                
                # Clear the results list and take a short break
                all_results.extend(results)
                results = []
                time.sleep(10)  # Adjust the sleep time as needed

    # Format the results
    formatted_results = format_results(all_results)

    # Save formatted results
    columns = ['Name', 'Current_Date', 'Current_Quarter', 'Variable', 'Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Y0', 'Y1', 'Y2', 'Y3', 'Y4']
    df = pd.DataFrame(formatted_results, columns=columns)
    output_filename = os.path.join(output_folder, f'formatted_forecasts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    df.to_csv(output_filename, index=False)
    logger.info(f"Formatted results saved to {output_filename}")

if __name__ == "__main__":
    main()