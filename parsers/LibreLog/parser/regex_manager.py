import re
import string
from datetime import datetime


def verify_one_regex_to_match_whole_log(log, regex):
    log = log.replace(",", "")
    regex = regex.replace(",", "")
    regex = f'^{regex}$'
    try:
        if re.search(regex, log):

            return True
        else:
            return False
    except re.error as e:
        return False
    

def is_punctuation_or_space(s):
    allowed_chars = string.punctuation + ' '
    filtered_string = ''.join(char for char in s if char not in allowed_chars)
    return all(char in allowed_chars for char in s) or len(filtered_string) < 3


class RegexTemplateManager:
    def __init__(self):
        self.regex_templates = []
        self.regex_template_set = set()
        self.total_time = 0.0

    def add_regex_template(self, regex_template,log=False):
        if is_punctuation_or_space(regex_template):
            return False
        if regex_template in self.regex_template_set:
            return
        else:
            if verify_one_regex_to_match_whole_log(log, regex_template):
                self.regex_template_set.add(regex_template)
            else:
                return 
        word_count = regex_template.count(' ') + 1
        regex_template_tuple = (word_count, regex_template)

        if not self.regex_templates:
            self.regex_templates.append(regex_template_tuple)
            print(f"\nRegex Template is added into manager: {regex_template_tuple}", flush=True)
            return

        insert_index = self.get_index_by_length(word_count)

        while insert_index < len(self.regex_templates) and self.regex_templates[insert_index][0] >= word_count:
            insert_index += 1

        self.regex_templates.insert(insert_index, regex_template_tuple)
        print(f"Regex Template is added into manager: {insert_index}, {regex_template_tuple}", flush=True)

    def add_regex_templates(self, regex_templates):
        for regex_template in regex_templates:
            self.add_regex_template(regex_template)

    def print_regex_templates(self):
        for word_count, regex_regex_template in self.regex_templates:
            print(f"Word Count: {word_count}, Regex regex_template: {regex_regex_template}", flush=True)

    def get_index_by_length(self, max_length):
        left, right = 0, len(self.regex_templates) - 1
        target_index = len(self.regex_templates)  

        while left <= right:
            mid = (left + right) // 2
            if self.regex_templates[mid][0] <= max_length:
                target_index = mid  
                right = mid - 1
            else:
                left = mid + 1

        return target_index

    def get_regex_templates_by_length(self, max_length):
        target_index = self.get_index_by_length(max_length=max_length + 1)  
        to_index = self.get_index_by_length(max_length=max_length - 1)
        return self.regex_templates[target_index:to_index]

    def find_matched_regex_template(self, log):
        start_time = datetime.now()
        log_word_count = log.count(' ') + 1
        regex_to_match = self.get_regex_templates_by_length(log_word_count)
        for regex in regex_to_match:
            try:
                if verify_one_regex_to_match_whole_log(log, regex[1]):
                    time_taken = datetime.now() - start_time
                    self.total_time += time_taken.total_seconds()  
                    return regex[1]
            except:
                pass
        time_taken = datetime.now() - start_time
        self.total_time += time_taken.total_seconds()  
        return False

    def print_time(self):
        print("[Regex matching time taken: {!s}]".format(self.total_time), flush=True)


if __name__ == '__main__':
    manager = RegexTemplateManager()
    manager.add_regex_template("jk2_init\(\) Found child (.*?) in scoreboard slot (.*?)$")
    manager.add_regex_template("workerEnv\.init\(\) ok (.*?)$")
    manager.add_regex_template("mod_jk child workerEnv in error state (.*?)$")
    manager.print_regex_templates()
    a = manager.get_regex_templates_by_length(6)
    print(a)
    x = manager.find_matched_regex_template("mod_jk child workerEnv in error sta1te 6")
    print(x)
