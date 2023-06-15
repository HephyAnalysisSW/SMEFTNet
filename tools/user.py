import os

if os.environ['USER'] in ['robert.schoefbeck']:
    plot_directory         = "/groups/hephy/cms/robert.schoefbeck/www/pytorch/"
    model_directory        = "/groups/hephy/cms/robert.schoefbeck/NN/models/"
    data_directory         = "/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/"
elif os.environ['USER'] in ['suman.chatterjee']:
    plot_directory = './plots/'
    model_directory= "/groups/hephy/cms/suman.chatterjee/ML-pytorch/models/"
    #data_directory = "groups/hephy/cms/suman.chatterjee/ML-pytorch/data/"
    data_directory         = "/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/"
elif os.environ['USER'] in ['sesanche']:
    plot_directory = './plots/'
    model_directory= "./models/"
    data_directory = "./data/"
