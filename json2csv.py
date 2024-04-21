# Python program to convert
# JSON file to CSV


import json
import csv

#STRING.replace("': '", '": "').replace("', '", '", "').replace("{'", '{"').replace("'}", '"}').replace("': \"", '": "').replace("', \"", '", "').replace("\", '", '", "').replace("'", '\\"')
# Opening JSON file and loading the data
# into the variable data
with open('D:\dissertation\dataset\dblp.v10\dblp-ref\dblp-ref-3.json') as json_file:
	data = json.load(json_file)

employee_data = data['data']

# now we will open a file for writing
data_file = open('D:\dissertation\dataset\dblp.v10\dblp-ref\dblp-ref-3.csv', 'w')

# create the csv writer object
csv_writer = csv.writer(data_file)

# Counter variable used for writing
# headers to the CSV file
count = 0

for emp in employee_data:
	if count == 0:

		# Writing headers of CSV file
		header = emp.keys()
		csv_writer.writerow(header)
		count += 1

	# Writing data of CSV file
	csv_writer.writerow(emp.values())

data_file.close()
