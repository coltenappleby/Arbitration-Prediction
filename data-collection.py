import pandas as pd
import numpy as np

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import requests

# https://developers.google.com/sheets/api/quickstart/python
# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


# The ID and range of a sample spreadsheet.


def collect_arb_data():
	"""Shows basic usage of the Sheets API.
	Prints values from a sample spreadsheet.
	"""

	SAMPLE_SPREADSHEET_ID_ARB = "1qxfZP9F05K7mIO4Kwv5D6hEbvtmAutIc807GSfoCQ1A"
	MLB_SALARY_DATA = "12XSXOQpjDJDCJKsA4xC1e_9FlS11aeioZy_p1nqpclg"
	ARB_SHEATNAME = "MLB-2023 Arb by WARP!A4:L"
	api_responses = []

	start_year = 2010
	end_year = 2023

	creds = None
	# The file token.json stores the user's access and refresh tokens, and is
	# created automatically when the authorization flow completes for the first
	# time.
	if os.path.exists("token.json"):
		creds = Credentials.from_authorized_user_file("token.json", SCOPES)

	# If there are no (valid) credentials available, let the user log in.
	if not creds or not creds.valid:
		if creds and creds.expired and creds.refresh_token:
			creds.refresh(Request())
		else:
			flow = InstalledAppFlow.from_client_secrets_file(
				"credentials.json", SCOPES
			)
			creds = flow.run_local_server(port=0)
		# Save the credentials for the next run
		with open("token.json", "w") as token:
			token.write(creds.to_json())

	try:
		service = build("sheets", "v4", credentials=creds)

		for year in range(start_year, end_year + 1):
			# Call the Sheets API
			sheet = service.spreadsheets()
			result = (
				sheet.values()
				.get(spreadsheetId=MLB_SALARY_DATA, range=f"{year}.xls!A3:D")
				.execute()
			)
			values = result.get("values", [])

			if not values:
				print("No data found.")
				return
			for row in values:
				row.append(year)

			api_responses.append(values)

		return api_responses

	except HttpError as err:
		print(err)


def get_fangraphs_data(
		start_year=2010,
		end_year=2023,
		position="all",
		stats="bat",  # pit
		sort_stat="WAR",
		qual=5,  # qualified is "y"
):
	years = []

	for year in range(start_year, end_year + 1):
		url = f"https://www.fangraphs.com/api/leaders/major-league/data?age=&pos={position}&stats={stats}&lg=all&qual={qual}&season={year}&season1={year}&startdate=&enddate=&month=0&hand=&team=0&pageitems=2000000000&pagenum=1&ind=0&rost=0&players=&type=8&postseason=&sortdir=default&sortstat=WAR"
		response = requests.get(url)
		response.raise_for_status()
		json_data = response.json()
		df = pd.DataFrame(json_data['data'])
		years.append(df)
	merged_df = pd.concat(years, ignore_index=True)
	merged_df.to_csv(f'./data/mlb-stats-{stats}-{start_year}-{end_year}.csv', index=False)


def clean_arb_date(file_name):
	"""
	Clean the data from the arb data
	"""
	pass






# for result in results:
#   print(result, end="\n"*2)


if __name__ == "__main__":
	# collect_arb_data()
	# raw_data = collect_arb_data()
	# # print(raw_data.shape)
	# flat_list = [item for sublist in raw_data for item in sublist]
	# arb_data_columns = ['Player', 'Season', 'Club', 'Position', 'MLS', 'Year WAR/WARP', 'Career WARP', 'Year Salary',
	# 					'Player Request', 'Club Offer', 'Next Year Salary', 'Estimate/Notes']
	# mlb_salary_columns = ['Player', 'Position', 'MLS', 'Salary', 'Year']
	# cleaned_data = pd.DataFrame(flat_list, columns=mlb_salary_columns)
	# print(cleaned_data.head())
	# cleaned_data.to_csv('./data/mlb-salaries_2010-2023.csv', index=False)

	get_fangraphs_data(stats='pit')


