# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 14:14:29 2019

@author: Brandon
"""

from yahoo_fin import stock_info as si

# Get live price of Apple
aaplPrice = si.get_live_price("aapl")
# Get live price of Amazon
amznPrice = si.get_live_price("amzn")

# Get quote table back as a dataframe
aaplQuoteTable = si.get_quote_table("aapl", dict_result = False)
# Get quote table back as dictionary
amznQuoteTable = si.get_quote_table("amzn")