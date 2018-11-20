import load_data
from numpy import concatenate


# from LoadDataNBC2 import main
def over_all_data():
    subjects = ["Data/ExtractedData/Subject_04799",
                    "Data/ExtractedData/Subject_04820",
                    "Data/ExtractedData/Subject_04847",
                    "Data/ExtractedData/Subject_05680",
                    "Data/ExtractedData/Subject_05675",
                    "Data/ExtractedData/Subject-05710"]

    # testing_subjects = ["Data/ExtractedData/Subject-05710"]
    # info1, trails1 = main(training_subjects[0])
    # info2, trails2 = main(training_subjects[1])
    # info3, trails3 = main(training_subjects[2])
    # info4, trails4 = main(training_subjects[3])
    # info5, trails5 = main(training_subjects[4])

    examples, labels = load_data.get_all_subjects_data(subjects)

    # print(freq)

    training_examples = concatenate((examples[:192][:], examples[240:432][:]))
    training_labels = concatenate((labels[:192], labels[240:432]))

    print("getting test data")

    testing_examples = concatenate((examples[192:240][:], examples[432:][:]))
    testing_labels = concatenate((labels[192:240], labels[432:]))

    from GNB import GaussianNB
    clf = GaussianNB()

    print("Fitting..")
    clf.fit(training_examples, training_labels)

    print("predicting")
    score = clf.score(testing_examples, testing_labels)
    print(score)


def over_one_subject():
    subjects = ["Data/ExtractedData/Subject_04799",
                "Data/ExtractedData/Subject_04820",
                "Data/ExtractedData/Subject_04847",
                "Data/ExtractedData/Subject_05680",
                "Data/ExtractedData/Subject_05675",
                "Data/ExtractedData/Subject-05710"]

    avg_score = []

    for subject in subjects:
        examples, labels = load_data.get_subject_data(subject)

        # print(freq)

        training_examples = concatenate((examples[:32][:], examples[40:72][:]))
        training_labels = concatenate((labels[:32], labels[40:72]))

        print("getting test data")

        testing_examples = concatenate((examples[32:40][:], examples[72:][:]))
        testing_labels = concatenate((labels[32:40], labels[72:]))

        from GNB import GaussianNB
        clf = GaussianNB()

        print("Fitting..")
        clf.fit(training_examples, training_labels)

        print("predicting")
        score = clf.score(testing_examples, testing_labels)
        print(score)
        avg_score.append(score)

    print('avg: ', sum(avg_score)/len(avg_score)*100, '%')


def pos_vs_neg_sen_per_subject():
    subjects = ["Data/ExtractedData/Subject_04799",
                "Data/ExtractedData/Subject_04820",
                "Data/ExtractedData/Subject_04847",
                "Data/ExtractedData/Subject_05680",
                "Data/ExtractedData/Subject_05675",
                "Data/ExtractedData/Subject-05710"]

    for subject in subjects:
        examples, labels = load_data.get_subject_data_sentence(subject)

        # print(freq)

        training_examples = concatenate((examples[:32][:], examples[40:72][:]))
        training_labels = concatenate((labels[:32], labels[40:72]))

        print("getting test data")

        testing_examples = concatenate((examples[32:40][:], examples[72:][:]))
        testing_labels = concatenate((labels[32:40], labels[72:]))

        from GNB import GaussianNB
        clf = GaussianNB()

        print("Fitting..")
        clf.fit(training_examples, training_labels)

        print("predicting")
        score = clf.score(testing_examples, testing_labels)
        print(score)

over_one_subject()
# over_all_data()
# pos_vs_neg_sen_per_subject()