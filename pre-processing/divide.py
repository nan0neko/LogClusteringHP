import re
import pandas as pd


logfile = open("E:\HPE-CTY\logfile.log","r")

date_time = r'(?P<date>\d{4}-\d{2}-\d{2})\s(?P<time>\d{2}:\d{2}:\d{2})\s-\s(?P<component>[\w/.]+)(\[(?P<pid>[#]+)\])?:\s%(?P<source>\w+-[#]*)[\:\- ]*(?P<message>.*)'

dates = []
times = []
components = []
pids = []
sources = []
messages = []
for log in logfile:
    match = re.match(date_time,log)
    if match:
        date = match.group('date')
        time = match.group('time')
        component = match.group('component')
        pid = match.group('pid')
        source = match.group('source')
        message = match.group('message')
        dates.append(date)
        times.append(time)
        components.append(component)
        pids.append(pid)
        sources.append(source)
        messages.append(message)
    
dict = {'Date':dates, 'Time' : times, 'Components' : components, 'PID' : pids, 'Source of Origin' : sources, 'Message' : messages}
df = pd.DataFrame(dict)
df.to_csv('output.csv', index=False)