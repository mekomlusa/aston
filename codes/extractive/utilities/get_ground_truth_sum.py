import re
import os.path


def get_ground_truth_sum(file_path):
    with open(file_path, 'r') as rf:
        data = rf.read()
        # first count how many @highlights
        sentence_count = len(re.findall(r'@highlight', data)) * -1
        # get the summary and the original text
        return [sent.strip() for sent in data.split('@highlight')[sentence_count:]]


if __name__ == '__main__':
    dir_name = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(dir_name, os.pardir, os.pardir, os.pardir))
    stories_dir = os.path.join(root_dir, 'cnn_stories')
    story_path = os.path.join(stories_dir, '0a0adc84ccbf9414613e145a3795dccc4828ddd4.story')
    sums = get_ground_truth_sum(story_path)
    for s in sums:
        print(s)