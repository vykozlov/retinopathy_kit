#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 18:07:17 2018

@author: valentin
"""
import os, sys
import pandas as pd


def main(argv):

    subm_file = str(sys.argv[1])

    df = pd.read_csv(subm_file)

    output_dir = os.path.dirname(subm_file)
    df[:5000].to_csv(os.path.join(output_dir, 'subSubmission00-05k.csv'), index=False)
    df[5000:10000].to_csv(os.path.join(output_dir, 'subSubmission05-10k.csv'), index=False)
    df[10000:15000].to_csv(os.path.join(output_dir, 'subSubmission10-15k.csv'), index=False)
    df[15000:20000].to_csv(os.path.join(output_dir, 'subSubmission15-20k.csv'), index=False)    
    df[20000:25000].to_csv(os.path.join(output_dir, 'subSubmission20-25k.csv'), index=False)
    df[25000:30000].to_csv(os.path.join(output_dir, 'subSubmission25-30k.csv'), index=False)
    df[30000:35000].to_csv(os.path.join(output_dir, 'subSubmission30-35k.csv'), index=False)
    df[35000:40000].to_csv(os.path.join(output_dir, 'subSubmission35-40k.csv'), index=False)    
    df[40000:45000].to_csv(os.path.join(output_dir, 'subSubmission40-45k.csv'), index=False)
    df[45000:50000].to_csv(os.path.join(output_dir, 'subSubmission45-50k.csv'), index=False)
    df[50000:].to_csv(os.path.join(output_dir, 'subSubmission50-End.csv'), index=False)

    
if __name__ == "__main__":
   main(sys.argv[1:])