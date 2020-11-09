import numpy as np
import pandas as pd 
import cv2
import os, sys
from tensorflow.keras import utils as np_utils
import sklearn.model_selection as model_selection


#Set CD to files 
data_path='/crop_part1'
os.chdir(data_path)
categories = os.listdir(data_path)

#Place directory in Pandas DF
df = pd.DataFrame(categories)
df.columns = ['File_Names']


#Split values into columns 
df['File_Stem'] = df.File_Names.str.replace('.jpg.chip.jpg','')
df[['Age','Gender','Race','DateTime']] = df.File_Stem.str.split(pat='_',expand=True)
df.head()




#Disable false pandas warning
pd.options.mode.chained_assignment = None

#Apply mask no_mask split
mask,no_mask = model_selection.train_test_split(df,test_size=0.5, random_state=75)

#Where File_names value is the same as train, set Labels value to 1
df['Labels'] = np.where(df['File_Names'].isin(mask['File_Names']),1,0)
df['new_names'] = 5

#Confirm 0 and 1 have equal race and gender breakdowns 
#df.loc[df['Labels']==0,"Race"].value_counts()

#Create new names
for i in range(0, len(df)):
    #Line to make the new names
    df.new_names[i] = str(df.Labels[i]) + "_" +  str(df.DateTime[i]) + ".jpg"

df.head()



#Re-name files
for category in categories:  
    img_names = os.path.join(data_path, category)  
    new_name = df.loc[df['File_Names']==category,'new_names'].to_string(index=False)
    new_name.strip()
    new_names = os.path.join(data_path, new_name)  
    os.rename(img_names, new_names)
