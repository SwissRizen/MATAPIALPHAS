# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 20:20:03 2021

@author: nils
"""
from flask import Flask, render_template
import sqlite3 as sq

    #%%############################################################################################################################## App definition
app = Flask(__name__)

conn = sq.connect('flares.db')
c = conn.cursor()
#%%############################################################################################################################## Home Page

@app.route('/')
def index():
    return render_template('index.html')

#%%############################################################################################################################## Flare Page

@app.route('/flares')
def flares():
    conn = sq.connect('flares.db')
    c = conn.cursor()
    c.execute('''SELECT * from flares ORDER BY ID DESC''')
    all_flares = c.fetchall()
    return render_template('flares.html', all_flares=all_flares)


#%%############################################################################################################################## run app plus create database

if __name__ == '__main__':
    app.run( port=8888)