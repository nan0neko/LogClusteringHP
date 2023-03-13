import datetime
import pandas as pd

log_data=open("output3.csv",'r')

log_entries = sorted(log_data, key=lambda x: datetime.datetime.strptime(x.split(',')[0], '%Y-%m-%d %H:%M:%S'))

groups = []
current_group = []
prev_timestamp = datetime.datetime.strptime(log_entries[0].split(',')[0], '%Y-%m-%d %H:%M:%S')
k=datetime.timedelta(minutes=5)

for log_entry in log_entries:
    
    
    timestamp = datetime.datetime.strptime(log_entry.split(',')[0], '%Y-%m-%d %H:%M:%S') 
  
    
    if (timestamp - prev_timestamp) <= k :
        current_group.append(log_entry)
    else:
    
        if current_group:
            groups.append(current_group)
        current_group = [log_entry]
        log = datetime.datetime.strptime(log_entry.split(',')[0], '%Y-%m-%d %H:%M:%S') 
        while (log-prev_timestamp)>k :
             prev_timestamp = prev_timestamp + datetime.timedelta(minutes=5)
  
types=[]
groupss=[]
logss=[]
types_logs=[]
lnm=0
for i, group in enumerate(groups):
    
    grp_type=[]
    for j in group:
        if  j.split(',')[1] not in grp_type:
            grp_type.append(j.split(',')[1])
    types.append(group[0].split(',')[0])
    groupss.append(len(group))
    logss.append(grp_type)
    
    types_logs.append(len(grp_type))
    
    
    

        
dict = {'group_no':types,'number':groupss,'types':logss,'types_logs':types_logs}
df = pd.DataFrame(dict)
df.to_csv('answer.csv',index=False)    
