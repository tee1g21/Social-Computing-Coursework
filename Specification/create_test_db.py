# !/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# (c) Copyright University of Southampton, 2021
#
# Copyright in this software belongs to University of Southampton,
# Highfield, University Road, Southampton SO17 1BJ
#
# Created By : Stuart E. Middleton
# Created Date : 2021/02/11
# Project : Teaching
#
######################################################################

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, codecs, json, math, time, warnings, logging, os, shutil, subprocess, sqlite3, traceback, random

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logging started')

'''
test
# setup a ECS VM (which will run RedHat Enterprise 8 and therefore need Python 3.9)
# copy comp3208_100k_train_withratings.csv file to the same folder as db_example.py

sudo yum -y install gcc gcc-c++ python39-devel
sudo yum install python39 python39-pip
sudo python3.9 -m pip install --upgrade pip
sqlite3 comp3208.db
	CREATE TABLE IF NOT EXISTS example_table (UserID INT, ItemID INT, Rating FLOAT, PredRating FLOAT);
	.quit
python3.9 db_example.py

'''

if __name__ == '__main__':

	logger.info( 'loading training set and creating sqlite3 database' )

	conn = sqlite3.connect('../../data/dataset2/test_20M.db') 
	c = conn.cursor()

	# Create the table if it doesn't exist
	c.execute('''
		CREATE TABLE IF NOT EXISTS example_table (
			UserID INT,
			ItemID INT,
			TimeStamp INT,
			PredRating FLOAT DEFAULT NULL
		)
	''')
	conn.commit()

	# Optional: Clear existing data in the table to avoid duplicates
	c.execute('DELETE FROM example_table')
	conn.commit()

	# Read the CSV file
	with codecs.open('../../data/dataset2/test_20M_withoutratings.csv', 'r', 'utf-8', errors='replace') as file:
		lines = file.readlines()

	# Insert data into the database
	for line in lines:  # Skip header row if your CSV has one
		parts = line.strip().split(',')
		if len(parts) == 3:  # Ensure there are only three elements (UserID, ItemID, TimeStamp)
			user_id, item_id, timestamp = map(int, parts)  # Convert strings to integers
			# Insert data into the table
			c.execute('INSERT INTO example_table (UserID, ItemID, TimeStamp) VALUES (?, ?, ?)',
						(user_id, item_id, timestamp))
		else:
			print(f"Skipping line due to incorrect format: {line}")

	conn.commit()

	# Optionally, create an index to improve query performance
	c.execute('CREATE INDEX IF NOT EXISTS idx_userid_itemid ON example_table (UserID, ItemID)')
	conn.commit()

	# Close the database connection
	c.close()
	conn.close()

