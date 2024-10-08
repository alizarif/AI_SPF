import os
import pandas as pd
import time
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

def get_current_date():
    return datetime.now().strftime("%b %d %Y")

def get_current_quarter():
    today = datetime.now()
    return f"{today.year} Q{(today.month - 1) // 3 + 1}"

def create_forecast_query(variable_data):
    current_date = get_current_date()
    current_quarter = get_current_quarter()
    query = (f"We are currently on {current_date}, which falls in {current_quarter}. "
             f"Please provide your best numeric forecasts for the following variables: {', '.join(variable_data['definition'])}. "
             f"Do this for the current quarter and the next 4 quarters. "
             f"Also provide annual forecasts for this year and the next 4 years. "
             f"Use available information up to today, your professional judgement, and experience. "
             f"Your forecast is anonymous. Provide the forecasts as a sequence of numerical values only. "
             f"Please provide your forecasts in this exact format for each variable: "
             f"Variable Name: (current quarter, Q+1, Q+2, Q+3, Q+4, this year's average, year+1 average, year+2 average, year+3 average, year+4 average)")
    return query, current_date, current_quarter

def process_forecast(client, assistant_id, name, persona, forecast_query):
    thread = client.beta.threads.create()
    
    messages = [
        {"role": "user", "content": f"You are {name}. {persona}\n\nDo you understand your role?"},
        {"role": "user", "content": forecast_query},
        {"role": "user", "content": "Please provide a brief explanation for your forecast in 1-2 sentences."}
    ]
    
    for message in messages:
        client.beta.threads.messages.create(thread_id=thread.id, **message)
        run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)
        
        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run_status.status == 'completed':
                break
            time.sleep(1)
    
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    responses = [msg.content[0].text.value for msg in reversed(messages.data) if msg.role == "assistant"]
    return responses

def parse_forecasts(forecast_string):
    forecasts = []
    for line in forecast_string.split('\n'):
        if ':' in line:
            variable, values = line.split(':', 1)
            values = values.strip()
            if values.startswith('(') and values.endswith(')'):
                values = values[1:-1]  # Remove parentheses
            values = [v.strip() for v in values.split(',')]
            if len(values) == 10:  # Ensure we have the correct number of values
                forecasts.append([variable.strip()] + values)
    return forecasts

def main():
    load_dotenv('api.env')
    client = OpenAI(api_key=os.getenv('API_KEY1'))

    output_folder = 'future_forecasts'
    os.makedirs(output_folder, exist_ok=True)

    variable_data = pd.read_csv('inflation_def.csv', encoding='latin1')
    forecaster_data = pd.read_csv('forecasters.csv')

    forecast_query, current_date, current_quarter = create_forecast_query(variable_data)
    assistant_id = "asst_BfXuSsa5HJSPokfNcW0ihdEk"

    all_forecasts = []
    for _, row in forecaster_data.iterrows():
        name = row['name']
        persona = row['persona']
        print(f"Processing forecast for {name}...")
        
        responses = process_forecast(client, assistant_id, name, persona, forecast_query)
        
        if len(responses) >= 2:
            forecasts = parse_forecasts(responses[1])
            for forecast in forecasts:
                all_forecasts.append([name, current_date, current_quarter] + forecast)
        else:
            print(f"Incomplete responses for {name}")

    # Save formatted results
    columns = ['Name', 'Current_Date', 'Current_Quarter', 'Variable', 'Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Y0', 'Y1', 'Y2', 'Y3', 'Y4']
    formatted_df = pd.DataFrame(all_forecasts, columns=columns)
    formatted_filename = os.path.join(output_folder, f'formatted_forecasts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    formatted_df.to_csv(formatted_filename, index=False)
    print(f"Formatted results saved to {formatted_filename}")

if __name__ == "__main__":
    main()
