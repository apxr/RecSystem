clear;
train_data = readtable("/Users/apurvpriyam/Documents/Sem2/CSE6643/project/code/train_old.csv");
train_data = train_data{:,:};
test_data = readtable("/Users/apurvpriyam/Documents/Sem2/CSE6643/project/code/test_old.csv");
test_data = test_data{:,:};

save('/Users/apurvpriyam/Documents/Sem2/CSE6643/project/code/movielens_old')

