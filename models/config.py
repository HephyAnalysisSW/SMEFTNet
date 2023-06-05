import os

if os.environ["USER"] in ["robert.schoefbeck"]:
    data_directory = "/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v5/"
else:
    raise NotImplementedError("Add yourselves.")
