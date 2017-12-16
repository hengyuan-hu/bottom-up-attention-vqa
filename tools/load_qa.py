import json


if __name__ == '__main__':
    train_answer_file = 'data/v2_mscoco_train2014_annotations.json'
    train_answers = json.load(open(train_answer_file))['annotations']

    val_answer_file = 'data/v2_mscoco_val2014_annotations.json'
    val_answers = json.load(open(val_answer_file))['annotations']

    train_question_file = 'data/v2_OpenEnded_mscoco_train2014_questions.json'
    train_questions = json.load(open(train_question_file))['questions']

    val_question_file = 'data/v2_OpenEnded_mscoco_val2014_questions.json'
    val_questions = json.load(open(val_question_file))['questions']
