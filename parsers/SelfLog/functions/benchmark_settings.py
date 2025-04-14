import re

from tqdm import tqdm

benchmark_settings = {
    # 'HDFS': {
    #     'log_file': 'HDFS/HDFS_2k.log',
    #     'buffer_file': 'HDFS/HDFS_buffer.log',
    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    #     'out_file': 'HDFS_2k.log_parsed.csv',
    #     'dpp_eval': '_results/2000_HDFS_result.csv'
    #     # 'out_file': 'HDFS_full_window.log_parsed.csv' # online
    # },

    # 'Hadoop': {
    #     'log_file': 'Hadoop/Hadoop_2k.log',
    #     'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
    #     'regex': [r'(\d+\.){3}\d+'],
    #     'out_file': 'Hadoop_2k.log_parsed.csv',
    #     'dpp_eval': 'reproduce_results/2000_Hadoop_result.csv'
    # },

    # 'Spark': {
    #     'log_file': 'Spark/Spark_2k.log',
    #     'buffer_file': 'Spark/Spark_buffer.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    #     'out_file': 'Spark_2k.log_parsed.csv',
    #     'dpp_eval': 'reproduce_results/2000_Spark_result.csv',
    #     # 'out_file': 'Spark_full_window.log_parsed.csv'
    # },

    # 'Zookeeper': {
    #     'log_file': 'Zookeeper/Zookeeper_2k.log',
    #     'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
    #     'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
    #     'out_file': 'Zookeeper_2k.log_parsed.csv',
    #     'dpp_eval': 'reproduce_results/2000_Zookeeper_result.csv'
    # },

    # 'BGL': {
    #     'log_file': 'BGL/BGL_2k.log',
    #     'buffer_file': 'BGL/BGL_buffer.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'out_file': 'BGL_2k.log_parsed.csv',
    #     'dpp_eval': 'reproduce_results/2000_BGL_result.csv',
    #     # 'out_file': 'BGL_full_window.log_parsed.csv'
    # },

    # 'HPC': {
    #     'log_file': 'HPC/HPC_2k.log',
    #     'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
    #     'regex': [r'=\d+'],
    #     'out_file': 'HPC_2k.log_parsed.csv',
    #     'dpp_eval': 'reproduce_results/2000_HPC_result.csv'
    # },
    # #
    # 'Thunderbird': {
    #     'log_file': 'Thunderbird/Thunderbird_2k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
    #     'regex': [r'(\d+\.){3}\d+'],
    #     'out_file': 'Thunderbird_2k.log_parsed.csv',
    #     'dpp_eval': 'reproduce_results/2000_Thunderbird_result.csv'
    # },

    # 'Windows': {
    #     'log_file': 'Windows/Windows_2k.log',
    #     'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
    #     'regex': [r'0x.*?\s'],
    #     'out_file': 'Windows_2k.log_parsed.csv',
    #     'dpp_eval': 'reproduce_results/2000_Windows_result.csv'
    # },

    # 'Linux': {
    #     'log_file': 'Linux/Linux_2k.log',
    #     'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],
    #     'out_file': 'Linux_2k.log_parsed.csv',
    #     'dpp_eval': 'reproduce_results/2000_Linux_result.csv'
    # },

    # 'Andriod': {
    #     'log_file': 'Andriod/Andriod_2k.log',
    #     'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
    #     'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
    #     'out_file': 'Andriod_2k.log_parsed.csv',
    #     'dpp_eval': 'reproduce_results/2000_Andriod_result.csv'
    # },

    # 'HealthApp': {
    #     'log_file': 'HealthApp/HealthApp_2k.log',
    #     'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
    #     'regex': [],
    #     'out_file': 'HealthApp_2k.log_parsed.csv',
    #     'dpp_eval': 'reproduce_results/2000_HealthApp_result.csv'
    # },

    # 'Apache': {
    #     'log_file': 'Apache/Apache_2k.log',
    #     'log_format': '\[<Time>\] \[<Level>\] <Content>',
    #     'regex': [r'(\d+\.){3}\d+'],
    #     'out_file': 'Apache_2k.log_parsed.csv',
    #     'dpp_eval': 'reproduce_results/2000_Apache_result.csv'
    # },

    # 'Proxifier': {
    #     'log_file': 'Proxifier/Proxifier_2k.log',
    #     'log_format': '\[<Time>\] <Program> - <Content>',
    #     'regex': [r'<\d+\s?sec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
    #     'out_file': 'Proxifier_2k.log_parsed.csv',
    #     'dpp_eval': 'reproduce_results/2000_Proxifier_result.csv'
    # },
    # #
    # 'OpenSSH': {
    #     'log_file': 'OpenSSH/OpenSSH_2k.log',
    #     'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
    #     'regex': [r'([\w-]+\.){2,}[\w-]+', r'(\d+\.){3}\d+'],
    #     'out_file': 'OpenSSH_2k.log_parsed.csv',
    #     'dpp_eval': 'reproduce_results/2000_OpenSSH_result.csv'
    # },

    # 'OpenStack': {
    #     'log_file': 'OpenStack/OpenStack_2k.log',
    #     'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
    #     'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\s\d+\s'],
    #     'out_file': 'OpenStack_2k.log_parsed.csv',
    #     'dpp_eval': 'reproduce_results/2000_OpenStack_result.csv'
    # },
    # #
    # 'Mac': {
    #     'log_file': 'Mac/Mac_2k.log',
    #     'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
    #     'regex': [r'([\w-]+\.){2,}[\w-]+'],
    #     'out_file': 'Mac_2k.log_parsed.csv',
    #     'dpp_eval': 'reproduce_results/2000_Mac_result.csv'
    # }
}

def generate_logformat_regex(logformat):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex

def load_logs(log_file, regex, headers):
    """ Function to transform log file to dataframe
    """
    log_messages = dict()
    linecount = 0
    with open(log_file, 'r') as fin:
        for line in tqdm(fin.readlines(), desc='load data'):
            try:
                linecount += 1
                match = regex.search(line.strip())
                message = dict()
                for header in headers:
                    message[header] = match.group(header)
                message['LineId'] = linecount
                log_messages[linecount] = message
            except Exception as e:
                pass
    return log_messages