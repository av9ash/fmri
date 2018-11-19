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

    examples, labels, freq = load_data.get_all_subjects_data(subjects)

    print(freq)

    training_examples = concatenate((examples[:100][:], examples[120:220][:]))
    training_labels = concatenate((labels[:100], labels[120:220]))

    print("getting test data")

    testing_examples = concatenate((examples[100:120][:], examples[220:][:]))
    testing_labels = concatenate((labels[100:120], labels[220:]))

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

    for subject in subjects:
        examples, labels = load_data.get_subject_data(subject)

        # print(freq)

        training_examples = concatenate((examples[:8][:], examples[10:18][:]))
        training_labels = concatenate((labels[:8], labels[10:18]))

        print("getting test data")

        testing_examples = concatenate((examples[8:10][:], examples[18:][:]))
        testing_labels = concatenate((labels[8:10], labels[18:]))

        from GNB import GaussianNB
        clf = GaussianNB()

        print("Fitting..")
        clf.fit(training_examples, training_labels)

        print("predicting")
        score = clf.score(testing_examples, testing_labels)
        print(score)


over_one_subject()
# over_all_data()